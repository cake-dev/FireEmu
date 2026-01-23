import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

@cuda.jit
def compute_reaction_and_fuel_kernel(fuel_density, fuel_moisture, 
                                     n_ep_received, incoming_x, incoming_y, incoming_z,
                                     centroid_x, centroid_y, centroid_z, ep_history,
                                     time_since_ignition, reaction_rate, ep_counts, 
                                     dt, cm, t_burnout, h_wood, vol, c_rad_loss, eep,
                                     cp_wood, t_crit, t_ambient, h_h2o):
    i, j, k = cuda.grid(3)
    nx, ny, nz = fuel_density.shape
    
    if i < nx and j < ny and k < nz:
        
        # --- 1. Sub-grid Centroid Update (Lagrangian to Eulerian mapping) ---
        n_new = n_ep_received[i, j, k]
        
        if n_new > 0:
            # Update center of activity based on where EPs landed
            avg_in_x = incoming_x[i, j, k] / n_new
            avg_in_y = incoming_y[i, j, k] / n_new
            avg_in_z = incoming_z[i, j, k] / n_new
            
            n_hist = ep_history[i, j, k]
            current_cx = centroid_x[i, j, k]
            current_cy = centroid_y[i, j, k]
            current_cz = centroid_z[i, j, k]
            
            # Weighted average with history (decaying history effect)
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

        # --- 2. Physics: Moisture & Reaction ---
        rho_f = fuel_density[i, j, k]
        moist = fuel_moisture[i, j, k]
        rr = 0.0
        
        # Thresholds
        MIN_FUEL = 1e-4
        MOIST_EXTINCTION = 0.15 # If moisture > 15%, hard to ignite immediately
        
        if rho_f > MIN_FUEL:
            has_energy = n_ep_received[i, j, k] > 0
            is_burning = time_since_ignition[i, j, k] > 0
            
            # --- Moisture Evaporation (Paper Section 2.7) ---
            if has_energy and not is_burning:
                # Energy available to heat/dry fuel
                # Energy In [J] = (EPs * E_per_EP_per_sec * dt)
                # Note: EEP is Watts (J/s), so multiply by dt
                energy_in = float(n_ep_received[i, j, k]) * eep * dt
                
                # Mass of water in cell [kg]
                water_mass = moist * rho_f * vol
                
                # Energy required to vaporize all water [J]
                # Q = m * Lv (ignoring Cp*deltaT for simplicity, or we can add it)
                energy_to_dry = water_mass * h_h2o
                
                if energy_in >= energy_to_dry:
                    # Flash dry
                    fuel_moisture[i, j, k] = 0.0
                    # Remaining energy contributes to ignition
                    # For simplicity, if dried, we allow ignition in next step or now
                    if energy_in > energy_to_dry * 1.1: # 10% buffer
                        time_since_ignition[i, j, k] = 0.01 # Ignite
                else:
                    # Partial drying
                    evaporated_mass = energy_in / h_h2o
                    new_water_mass = max(0.0, water_mass - evaporated_mass)
                    # Update moisture ratio: mass_water / mass_dry_fuel
                    fuel_moisture[i, j, k] = new_water_mass / (rho_f * vol)

            # --- Combustion ---
            # Re-check burning status after potential ignition above
            is_burning = time_since_ignition[i, j, k] > 0
            
            if is_burning:
                if time_since_ignition[i, j, k] == 0:
                    time_since_ignition[i, j, k] = 0.01
                
                time_since_ignition[i, j, k] += dt
                
                if time_since_ignition[i, j, k] <= t_burnout:
                    # Reaction Rate Model (Paper Eq 1)
                    # d_rho/dt = -Cm * rho_f * rho_o2 * ...
                    # Assuming infinite oxygen mixing (0.25 approx) for fast running
                    rho_o2 = 0.25 
                    
                    # Psi: Fraction actively burning. Simplified to 1.0 if ignited.
                    psi = 1.0     
                    
                    # Sigma: Turbulence mixing factor. 
                    # Paper uses Smagorinsky, simplified here to 1.0 for GPU perf
                    sigma = 1.0   
                    
                    rr = cm * rho_f * rho_o2 * psi * sigma
        
        reaction_rate[i, j, k] = rr
        
        # Apply Mass Loss
        if rr > 0:
            change = rr * dt
            fuel_density[i, j, k] = max(0.0, fuel_density[i, j, k] - change)
            
            # Energy Release (Paper Eq 4)
            # Q_net = R * (H_wood - sensible_heat_cost)
            sensible_heat_cost = cp_wood * (t_crit - t_ambient)
            effective_h = h_wood - sensible_heat_cost
            if effective_h < 0: effective_h = 0.0
            
            q_net = rr * effective_h * vol 
            
            # Number of EPs to emit (Paper Eq 5)
            # n_ep = (Q_net * (1 - loss)) / E_ep
            expected_n_ep = (q_net * (1.0 - c_rad_loss)) / eep
            
            # Stochastic integer rounding
            n_ep_base = int(expected_n_ep)
            remainder = expected_n_ep - n_ep_base
            
            # Simple deterministic dither based on location
            if (i + j + k) % 100 < (remainder * 100):
                n_ep_base += 1
                
            ep_counts[i, j, k] = n_ep_base
        else:
            ep_counts[i, j, k] = 0

@cuda.jit
def transport_eps_kernel(ep_counts, 
                         n_ep_received, incoming_x, incoming_y, incoming_z,
                         centroid_x, centroid_y, centroid_z, 
                         u, v, w, elevation, rng_states, 
                         dx, dy, dz, dt, eep):
    i, j, k = cuda.grid(3)
    nx, ny, nz = ep_counts.shape
    
    if i < nx and j < ny and k < nz:
        count = ep_counts[i, j, k]
        
        if count > 0:
            rng_idx = (k * (nx * ny)) + (j * nx) + i
            
            # Source position (global coordinates)
            src_x = i + centroid_x[i, j, k]
            src_y = j + centroid_y[i, j, k]
            src_z = k + centroid_z[i, j, k]
            
            uc = u[i, j, k]
            vc = v[i, j, k]
            wc = w[i, j, k]
            u_horiz = math.sqrt(uc*uc + vc*vc)
            
            # --- Intensity & Updraft (Paper Eq 17 & 18) ---
            # Intensity = kW/m2 approx
            area_cell = dx * dy 
            intensity = (count * eep) / area_cell 
            
            # w* = 0.377 * I (Eq 18), but scaled for units consistency
            # Using power law approx for stability in low-res grids
            w_star = 0.377 * math.pow(intensity * 0.001, 0.4) 
            if w_star < 0.1: w_star = 0.1

            # --- Dominance Ratio (Phi) (Paper Eq 11) ---
            phi = w_star / (w_star + u_horiz + 1e-6)
            
            # --- Length Scale (Paper Eq 14 & 15) ---
            h_flame = dz + 0.0155 * math.pow(intensity, 0.4)
            
            # Froude stretch
            stretch = 1.0
            if w_star > 1e-6:
                stretch = math.sqrt((uc*uc + vc*vc)/(w_star*w_star))
                
            l_scale = h_flame * max(1.0, stretch)
            
            # Grid connectivity enforcement
            if u_horiz > 0.5:
                l_scale = max(l_scale, dx * 1.5)
            
            # Partition EPs into Wind-driven vs Creeping (Paper Section 2.5)
            # Creeping handles local spread, Wind handles spotting/convection
            n_wind = int(count * 0.7) 
            n_creeping = count - n_wind
            
            # --- Wind Transport ---
            if n_wind > 0:
                # Bifurcation (Tower/Trough) Logic
                rnd_bifurcation = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                is_tower = rnd_bifurcation < phi
                
                rnd_dist = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                dist_travel = l_scale * (1.0 - math.sqrt(1.0 - rnd_dist))
                
                # Turbulence perturbation
                tke_est = 0.1 * (u_horiz + w_star)
                pert_mag = math.sqrt(tke_est) * 0.5
                u_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                v_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                w_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                
                dx_travel = 0.0
                dy_travel = 0.0
                dz_travel = 0.0

                if is_tower:
                    # Tower: Vertical bias
                    total_w = w_star + wc + w_pert
                    total_u = uc * 0.2 + u_pert 
                    total_v = vc * 0.2 + v_pert
                else:
                    # Trough: Horizontal bias
                    total_w = wc + w_pert 
                    total_u = uc + u_pert
                    total_v = vc + v_pert

                mag = math.sqrt(total_u**2 + total_v**2 + total_w**2)
                if mag > 1e-6:
                    dx_travel = (total_u / mag) * dist_travel
                    dy_travel = (total_v / mag) * dist_travel
                    dz_travel = (total_w / mag) * dist_travel

                dest_x_glob = src_x + dx_travel / dx
                dest_y_glob = src_y + dy_travel / dy
                dest_z_glob = src_z + dz_travel / dz
                
                # Boundary Checks & Atomic Add
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

            # --- Creeping Transport (Paper Eq 12) ---
            if n_creeping > 0:
                l_creep = max(2.0, dx * 1.2) # Approx from paper description
                
                theta = xoroshiro128p_uniform_float32(rng_states, rng_idx) * 2.0 * 3.14159
                rnd_creep = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                d_actual = l_creep * (1.0 - math.sqrt(rnd_creep))
                
                dest_x_glob = src_x + (math.cos(theta) * d_actual) / dx
                dest_y_glob = src_y + (math.sin(theta) * d_actual) / dy
                # Creeping stays in layer (or follows terrain, simplified here to layer)
                
                di = int(math.floor(dest_x_glob))
                dj = int(math.floor(dest_y_glob))
                dk = k # Stay in same z-layer for creeping
                
                if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                    off_x = dest_x_glob - di
                    off_y = dest_y_glob - dj
                    off_z = 0.5
                    cuda.atomic.add(n_ep_received, (di, dj, dk), n_creeping)
                    cuda.atomic.add(incoming_x, (di, dj, dk), off_x * n_creeping)
                    cuda.atomic.add(incoming_y, (di, dj, dk), off_y * n_creeping)
                    cuda.atomic.add(incoming_z, (di, dj, dk), off_z * n_creeping)