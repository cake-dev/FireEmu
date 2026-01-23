import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32
import config

@cuda.jit
def compute_reaction_and_fuel_kernel(fuel_density, fuel_moisture, 
                                     n_ep_received, incoming_x, incoming_y, incoming_z,
                                     centroid_x, centroid_y, centroid_z, ep_history,
                                     time_since_ignition, reaction_rate, ep_counts, 
                                     dt, cm, t_burnout, h_wood, vol, c_rad_loss, eep,
                                     cp_wood, t_crit, t_ambient):
    i, j, k = cuda.grid(3)
    nx, ny, nz = fuel_density.shape
    
    if i < nx and j < ny and k < nz:
        
        # --- 1. Sub-grid Centroid Update (Same as before) ---
        n_new = n_ep_received[i, j, k]
        
        if n_new > 0:
            avg_in_x = incoming_x[i, j, k] / n_new
            avg_in_y = incoming_y[i, j, k] / n_new
            avg_in_z = incoming_z[i, j, k] / n_new
            
            n_hist = ep_history[i, j, k]
            current_cx = centroid_x[i, j, k]
            current_cy = centroid_y[i, j, k]
            current_cz = centroid_z[i, j, k]
            
            total_count = n_hist + n_new
            if total_count > 1000:
                 total_count = 1000
                 n_hist = 1000 - n_new
            
            new_cx = (current_cx * n_hist + avg_in_x * n_new) / total_count
            new_cy = (current_cy * n_hist + avg_in_y * n_new) / total_count
            new_cz = (current_cz * n_hist + avg_in_z * n_new) / total_count
            
            centroid_x[i, j, k] = max(0.0, min(1.0, new_cx))
            centroid_y[i, j, k] = max(0.0, min(1.0, new_cy))
            centroid_z[i, j, k] = max(0.0, min(1.0, new_cz))
            
            ep_history[i, j, k] = total_count

        # --- 2. Reaction Physics (Same as before) ---
        rho_f = fuel_density[i, j, k]
        rr = 0.0
        
        if rho_f > 1e-4:
            has_energy = n_ep_received[i, j, k] > 0
            is_burning = time_since_ignition[i, j, k] > 0
            
            if has_energy or is_burning:
                if time_since_ignition[i, j, k] == 0:
                    time_since_ignition[i, j, k] = 0.01
                
                time_since_ignition[i, j, k] += dt
                
                if time_since_ignition[i, j, k] <= t_burnout:
                    rho_o2 = 0.25
                    psi = 1.0
                    sigma = 1.0
                    lambda_s = 1.0
                    rr = cm * rho_f * rho_o2 * psi * sigma * lambda_s
        
        reaction_rate[i, j, k] = rr
        
        if rr > 0:
            change = rr * dt
            fuel_density[i, j, k] = max(0.0, fuel_density[i, j, k] - change)
            
            # Pre-heating cost (Eq. 9)
            sensible_heat_cost = cp_wood * (t_crit - t_ambient)
            effective_h = h_wood - sensible_heat_cost
            if effective_h < 0: effective_h = 0.0
            
            q_net = rr * effective_h * vol * dt
            expected_n_ep = (q_net * (1.0 - c_rad_loss)) / eep
            n_ep_base = int(expected_n_ep)
            remainder = expected_n_ep - n_ep_base
            
            if (i + j + k) % 100 < (remainder * 100):
                n_ep_base += 1
                
            ep_counts[i, j, k] = n_ep_base
        else:
            ep_counts[i, j, k] = 0

@cuda.jit
def transport_eps_kernel(ep_counts, 
                         n_ep_received, incoming_x, incoming_y, incoming_z,
                         centroid_x, centroid_y, centroid_z, 
                         u, v, w, elevation, rng_states, dx, dy, dz, dt, slope_factor, jump_hack=False, mod_dt=False):
    """
    Sub-grid Transport with Bifurcation (Tower/Trough) Logic
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = ep_counts.shape
    
    if i < nx and j < ny and k < nz:
        count = ep_counts[i, j, k]
        
        if count > 0:
            rng_idx = (k * (nx * ny)) + (j * nx) + i
            
            # Source Location with Jitter
            jitter = (xoroshiro128p_uniform_float32(rng_states, rng_idx) - 0.5) * 0.2
            src_x = i + centroid_x[i, j, k] + jitter
            src_y = j + centroid_y[i, j, k] + jitter
            src_z = k + centroid_z[i, j, k] + jitter

            # --- CALCULATE SLOPE VECTOR ---
            # We calculate the gradient of the terrain at this (i, j) location
            # and add a bias vector pointing uphill.
            i_prev = max(0, i - 1)
            i_next = min(nx - 1, i + 1)
            j_prev = max(0, j - 1)
            j_next = min(ny - 1, j + 1)
            
            # Local gradient (Rise over Run)
            dz_dx = (elevation[i_next, j] - elevation[i_prev, j]) / (2.0 * dx)
            dz_dy = (elevation[i, j_next] - elevation[i, j_prev]) / (2.0 * dy)
            
            # Split counts
            n_wind = int(count * 0.7) 
            n_creeping = count - n_wind
            
            # --- 1. Wind Transport with Bifurcation ---
            if n_wind > 0:
                uc = u[i, j, k]
                vc = v[i, j, k]
                wc = w[i, j, k]
                
                # --- Bifurcation Physics (Section 2.5) ---
                # A. Estimate local flame updraft w* (Eq 18)
                # Intensity proxy: count (energy density)
                # Note: 0.05 is a tuning parameter to map counts to m/s. 
                # Ideally I = (count * EEP) / Area. Assuming Area ~ 4m^2
                w_star = 0.377 * (count * 0.1) 
                if w_star < 0.1: w_star = 0.1
                
                # B. Calculate Phi (Updraft Dominance Ratio) (Eq 21)
                u_horiz = math.sqrt(uc*uc + vc*vc)
                phi = w_star / (w_star + u_horiz + 1e-6)
                
                # C. Bifurcate Loop per EP
                
                # apply the paper's "Binary Choice" stochastically to the vector
                # Prob of going Vertical (Tower) = Phi
                rnd_bifurcation = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                
                is_tower = rnd_bifurcation < phi
                
                scale = 2.0
                up = xoroshiro128p_normal_float32(rng_states, rng_idx)
                vp = xoroshiro128p_normal_float32(rng_states, rng_idx)
                wp = xoroshiro128p_normal_float32(rng_states, rng_idx)

                dx_travel = 0.0
                dy_travel = 0.0
                dz_travel = 0.0

                if is_tower:
                    # TOWER MODE: Energy lofts vertically.
                    # High vertical component, low horizontal spread.
                    # This removes energy from the spreading front -> slows fire locally.
                    dx_travel = (uc * 0.2 + up * scale) 
                    dy_travel = (vc * 0.2 + vp * scale)
                    dz_travel = (w_star + wc + abs(wp) * scale) # Strong upward bias
                else:
                    # TROUGH MODE: Energy hugs the ground.
                    # Strong horizontal component (wind driven).
                    # This drives the fire forward rapidly.
                    dx_travel = (uc + up * scale)
                    dy_travel = (vc + vp * scale)
                    dz_travel = (wc + wp * scale) # Normal turbulence

                # --- APPLY SLOPE CORRECTION ---
                # Add uphill bias to the horizontal transport
                dx_travel += dz_dx * slope_factor
                dy_travel += dz_dy * slope_factor

                # jump_hack (optional)
                if jump_hack:
                    grid_dist_x = dx_travel / dx
                    grid_dist_y = dy_travel / dy
                    
                    # Calculate magnitude of the jump in 2D grid space
                    jump_mag = math.sqrt(grid_dist_x**2 + grid_dist_y**2)
                    
                    # Max Jump: maximum distance an EP can jump in one step
                    max_jump = 1.5 
                    
                    if jump_mag > max_jump:
                        factor = max_jump / jump_mag
                        grid_dist_x *= factor
                        grid_dist_y *= factor
                        # Note: We usually don't clamp Z as strictly to allow crowning
                    
                    # Apply the clamped transport
                    dest_x_glob = src_x + grid_dist_x
                    dest_y_glob = src_y + grid_dist_y
                    dest_z_glob = src_z + (dz_travel + 2.0) / dz
                else:
                    # Apply Transport
                    if mod_dt:
                        dx_travel *= dt
                        dy_travel *= dt
                        dz_travel *= dt
                    dest_x_glob = src_x + dx_travel / dx
                    dest_y_glob = src_y + dy_travel / dy
                    dest_z_glob = src_z + (dz_travel + 2.0) / dz # +2.0 base buoyancy
                
                # Destination update (Sub-grid logic)
                di = int(math.floor(dest_x_glob))
                dj = int(math.floor(dest_y_glob))
                dk = int(math.floor(dest_z_glob))
                
                if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                    off_x = dest_x_glob - di
                    off_y = dest_y_glob - dj
                    off_z = dest_z_glob - dk
                    
                    cuda.atomic.add(n_ep_received, (di, dj, dk), n_wind)
                    cuda.atomic.add(incoming_x, (di, dj, dk), off_x * n_wind)
                    cuda.atomic.add(incoming_y, (di, dj, dk), off_y * n_wind)
                    cuda.atomic.add(incoming_z, (di, dj, dk), off_z * n_wind)

            # --- 2. Creeping Transport (Same as before) ---
            if n_creeping > 0:
                theta = xoroshiro128p_uniform_float32(rng_states, rng_idx) * 2.0 * 3.14159
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                
                uc = u[i, j, k]
                vc = v[i, j, k]
                # Creeping also feels the slope!
                # Bias the random walk uphill
                uc += dz_dx * slope_factor * 2.0 # Stronger effect on creeping
                vc += dz_dy * slope_factor * 2.0
                
                wind_mag = math.sqrt(uc*uc + vc*vc)
                
                dot = 0.0
                if wind_mag > 1e-6:
                    dot = (uc * cos_theta + vc * sin_theta) / wind_mag
                
                l_creep_max = 2.0 * (1.0 - dot)
                
                rng_dist = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                d_actual = l_creep_max * (1.0 - math.sqrt(rng_dist))
                
                dest_x_glob = src_x + (cos_theta * d_actual) / dx
                dest_y_glob = src_y + (sin_theta * d_actual) / dy
                dest_z_glob = src_z + ((xoroshiro128p_uniform_float32(rng_states, rng_idx) - 0.5) * 0.5) / dz
                
                di = int(math.floor(dest_x_glob))
                dj = int(math.floor(dest_y_glob))
                dk = int(math.floor(dest_z_glob))
                
                if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                    off_x = dest_x_glob - di
                    off_y = dest_y_glob - dj
                    off_z = dest_z_glob - dk
                    
                    cuda.atomic.add(n_ep_received, (di, dj, dk), n_creeping)
                    cuda.atomic.add(incoming_x, (di, dj, dk), off_x * n_creeping)
                    cuda.atomic.add(incoming_y, (di, dj, dk), off_y * n_creeping)
                    cuda.atomic.add(incoming_z, (di, dj, dk), off_z * n_creeping)