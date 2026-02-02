"""
fire_gpu_qf_v2.py - Improved Fire Spread with Backing Fire & Slope Effects

Key improvements over previous version:
1. Slope-dependent spread rate (uphill harder, downhill easier)
2. Wind-direction dependent EP transport (backing fire much slower)
3. Reduced isotropic creeping spread
4. Proper flanking fire behavior

References:
- Rothermel (1972) slope factor
- Linn et al. (2020) QUIC-Fire paper
- Finney (1998) FARSITE slope/wind interactions
"""

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
    """
    Computes reaction rate, fuel consumption, and EP emission.
    Same as previous version - the core combustion physics are correct.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = fuel_density.shape
    
    if i < nx and j < ny and k < nz:
        
        # PHASE 1: SUB-GRID CENTROID UPDATE
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

        # PHASE 2: MOISTURE & COMBUSTION
        rho_f = fuel_density[i, j, k]
        moist = fuel_moisture[i, j, k]
        rr = 0.0
        
        MIN_FUEL = 1e-4
        MOIST_IGNITION_THRESHOLD = 0.30
        
        if rho_f > MIN_FUEL:
            has_energy = n_ep_received[i, j, k] > 0
            is_burning = time_since_ignition[i, j, k] > 0
            
            # Moisture evaporation
            if has_energy and moist > 0.001:
                energy_in = float(n_ep_received[i, j, k]) * eep * dt
                water_mass = moist * rho_f * vol
                energy_to_dry = water_mass * h_h2o
                
                if energy_in >= energy_to_dry:
                    fuel_moisture[i, j, k] = 0.0
                    remaining_energy = energy_in - energy_to_dry
                    if remaining_energy > eep * dt * 0.5 and not is_burning:
                        time_since_ignition[i, j, k] = 0.01
                else:
                    evaporated_mass = energy_in / h_h2o
                    new_water_mass = max(0.0, water_mass - evaporated_mass)
                    fuel_moisture[i, j, k] = new_water_mass / (rho_f * vol)
            
            # Ignition check
            current_moist = fuel_moisture[i, j, k]
            if has_energy and current_moist < MOIST_IGNITION_THRESHOLD and not is_burning:
                time_since_ignition[i, j, k] = 0.01
            
            # Combustion
            is_burning = time_since_ignition[i, j, k] > 0
            if is_burning:
                time_since_ignition[i, j, k] += dt
                if time_since_ignition[i, j, k] <= t_burnout:
                    rho_o2 = 0.25
                    psi = 1.0
                    sigma = 1.0
                    lambda_s = 1.0
                    rr = cm * rho_f * rho_o2 * psi * sigma * lambda_s
        
        reaction_rate[i, j, k] = rr
        
        # PHASE 3: FUEL CONSUMPTION & EP EMISSION
        if rr > 0:
            change = rr * dt
            fuel_density[i, j, k] = max(0.0, fuel_density[i, j, k] - change)
            
            sensible_heat_cost = cp_wood * (t_crit - t_ambient)
            effective_h = h_wood - sensible_heat_cost
            if effective_h < 0:
                effective_h = 0.0
            
            q_net = rr * effective_h * vol
            expected_n_ep = (q_net * (1.0 - c_rad_loss)) / eep
            
            n_ep_base = int(expected_n_ep)
            remainder = expected_n_ep - n_ep_base
            if (i + j + k) % 100 < int(remainder * 100):
                n_ep_base += 1
            
            ep_counts[i, j, k] = n_ep_base
        else:
            ep_counts[i, j, k] = 0


@cuda.jit
def transport_eps_kernel_v2(ep_counts,
                             n_ep_received, incoming_x, incoming_y, incoming_z,
                             centroid_x, centroid_y, centroid_z,
                             u, v, w, elevation, rng_states,
                             dx, dy, dz, dt, eep,
                             wind_dir_rad):
    """
    Improved EP transport with:
    1. Slope-dependent spread rate
    2. Wind-direction dependent transport (backing fire penalty)
    3. Reduced isotropic creeping
    
    Parameters:
    -----------
    wind_dir_rad : float
        Wind direction in radians (math convention, direction wind is GOING TO)
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = ep_counts.shape
    
    if i < nx and j < ny and k < nz:
        count = ep_counts[i, j, k]
        
        if count > 0:
            rng_idx = (k * (nx * ny)) + (j * nx) + i
            
            # Source position
            src_x = i + centroid_x[i, j, k]
            src_y = j + centroid_y[i, j, k]
            src_z = k + centroid_z[i, j, k]
            
            # Local wind
            uc = u[i, j, k]
            vc = v[i, j, k]
            wc = w[i, j, k]
            u_horiz = math.sqrt(uc*uc + vc*vc)
            
            # =================================================================
            # COMPUTE LOCAL SLOPE
            # =================================================================
            i_next = min(i + 1, nx - 1)
            i_prev = max(i - 1, 0)
            j_next = min(j + 1, ny - 1)
            j_prev = max(j - 1, 0)
            
            dz_dx = (elevation[i_next, j] - elevation[i_prev, j]) / (2.0 * dx)
            dz_dy = (elevation[i, j_next] - elevation[i, j_prev]) / (2.0 * dy)
            
            # Slope magnitude and direction
            slope_mag = math.sqrt(dz_dx**2 + dz_dy**2)
            if slope_mag > 0.001:
                # Upslope direction (direction of steepest ascent)
                upslope_dir_x = dz_dx / slope_mag
                upslope_dir_y = dz_dy / slope_mag
            else:
                upslope_dir_x = 0.0
                upslope_dir_y = 0.0
            
            # Slope angle in degrees
            slope_angle_deg = math.atan(slope_mag) * 180.0 / 3.14159
            
            # =================================================================
            # INTENSITY & FLAME PARAMETERS
            # =================================================================
            area_cell = dx * dy
            intensity = (count * eep) / area_cell
            
            w_star = 0.377 * math.pow(intensity * 0.001, 0.4)
            w_star = max(0.1, min(20.0, w_star))
            
            # Dominance ratio
            phi = w_star / (w_star + u_horiz + 1e-6)
            
            # Base length scale
            h_flame = dz + 0.0155 * math.pow(intensity, 0.4)
            stretch = 1.0
            if w_star > 1e-6:
                stretch = math.sqrt((uc*uc + vc*vc) / (w_star*w_star))
            l_scale = h_flame * max(1.0, min(stretch, 2.0))  # Cap stretch factor
            l_scale = min(l_scale, dx * 2.0)  # Never spread more than 2 cells
            
            # =================================================================
            # PARTITION EPs BY SPREAD TYPE
            # =================================================================
            # Key change: Much less creeping (creep% instead of 30%)
            # This prevents isotropic fill-in of interior
            creep_pct = 0.05# FIXME: tuning factor (probably correct though)
            n_creeping = int(count * creep_pct) 
            n_wind = count - n_creeping
            
            # =================================================================
            # WIND-DOMINATED TRANSPORT WITH DIRECTIONAL WEIGHTING
            # =================================================================
            if n_wind > 0:
                # Wind direction (unit vector)
                if u_horiz > 0.1:
                    wind_unit_x = uc / u_horiz
                    wind_unit_y = vc / u_horiz
                else:
                    wind_unit_x = math.cos(wind_dir_rad)
                    wind_unit_y = math.sin(wind_dir_rad)
                
                for ep_idx in range(n_wind):
                    # Random angle perturbation
                    # This determines if EP goes with wind, against, or flanking
                    angle_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * 0.8
                    
                    # Base direction is wind direction
                    ep_angle = math.atan2(wind_unit_y, wind_unit_x) + angle_pert
                    
                    # Direction this EP wants to go
                    ep_dir_x = math.cos(ep_angle)
                    ep_dir_y = math.sin(ep_angle)
                    
                    # ==========================================================
                    # COMPUTE SPREAD RATE MODIFIER
                    # ==========================================================
                    
                    # 1. Wind alignment factor
                    # Dot product: +1 = with wind, -1 = against wind, 0 = flanking
                    wind_alignment = ep_dir_x * wind_unit_x + ep_dir_y * wind_unit_y
                    
                    # Wind effect on spread rate
                    # With wind: up to 3x faster
                    # Against wind (backing): 0.1x to 0.3x slower
                    # Flanking: ~0.5x
                    if wind_alignment > 0:
                        # Head fire - spread WITH wind (reduced intensity)
                        wind_factor = 1.0 + 1.0 * wind_alignment * min(u_horiz / 5.0, 1.0)
                        wind_factor = min(wind_factor, 2.0)  # Cap at 2x (was 4x)
                    else:
                        # Backing fire - spread AGAINST wind
                        backing_penalty = 0.15 + 0.25 * (1.0 + wind_alignment)  # 0.15 to 0.4
                        wind_factor = backing_penalty

                    wind_factor = wind_factor * 1.0  # FIXME: tuning factor
                    
                    # 2. Slope alignment factor
                    # Dot product with UPSLOPE direction
                    # Positive = going uphill, negative = going downhill
                    slope_alignment = ep_dir_x * upslope_dir_x + ep_dir_y * upslope_dir_y
                    
                    # Rothermel slope factor: phi_s = 5.275 * beta^(-0.3) * tan(slope)^2
                    # Simplified version:
                    # Uphill: spread rate increases with slope^2
                    # Downhill: spread rate decreases slightly
                    tan_slope = slope_mag
                    
                    if slope_alignment > 0:
                        # Going UPHILL - fire spreads faster uphill
                        # But we're simulating this as "harder to ignite uphill"
                        # because the actual flame tilt helps
                        slope_factor = 1.0 + 0.5 * tan_slope * tan_slope * slope_alignment
                        slope_factor = min(slope_factor, 1.0)
                    else:
                        # Going DOWNHILL - fire spreads slower downhill
                        # Flames tilt away from fuel
                        slope_factor = 1.0 + 0.3 * slope_alignment * tan_slope
                        slope_factor = max(slope_factor, 0.3)
                    
                    slope_factor = slope_factor * 1.0  # FIXME: tuning factor

                    # Combined spread rate modifier
                    spread_modifier = wind_factor * slope_factor
                    
                    # ==========================================================
                    # COMPUTE TRANSPORT DISTANCE
                    # ==========================================================
                    
                    # Modified length scale
                    l_effective = l_scale * spread_modifier * 0.6 # FIXME tuning factor
                    
                    # Minimum spread distance (prevents total stall)
                    l_effective = max(l_effective, dx * 0.1)
                    
                    # Triangular distribution for distance
                    rnd_dist = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                    dist_travel = l_effective * (1.0 - math.sqrt(1.0 - rnd_dist))
                    
                    # Tower/trough bifurcation
                    rnd_bifurcation = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                    is_tower = rnd_bifurcation < phi
                    
                    if is_tower:
                        # Tower mode: strong vertical
                        total_w = w_star + wc
                        total_u = ep_dir_x * u_horiz * 0.3
                        total_v = ep_dir_y * u_horiz * 0.3
                    else:
                        # Trough mode: mostly horizontal
                        total_w = wc * 0.5
                        total_u = ep_dir_x * u_horiz + uc * 0.5
                        total_v = ep_dir_y * u_horiz + vc * 0.5
                    
                    # Normalize and apply distance
                    mag = math.sqrt(total_u**2 + total_v**2 + total_w**2)
                    if mag > 1e-6:
                        dx_travel = (total_u / mag) * dist_travel
                        dy_travel = (total_v / mag) * dist_travel
                        dz_travel = (total_w / mag) * dist_travel
                    else:
                        # No wind - spread radially
                        dx_travel = ep_dir_x * dist_travel
                        dy_travel = ep_dir_y * dist_travel
                        dz_travel = 0.0
                    
                    # Destination
                    dest_x_glob = src_x + dx_travel / dx
                    dest_y_glob = src_y + dy_travel / dy
                    dest_z_glob = src_z + dz_travel / dz
                    
                    di = int(math.floor(dest_x_glob))
                    dj = int(math.floor(dest_y_glob))
                    dk = int(math.floor(dest_z_glob))
                    
                    # Clamp to valid range
                    dk = max(0, min(dk, nz - 1))
                    
                    if 0 <= di < nx and 0 <= dj < ny:
                        off_x = dest_x_glob - di
                        off_y = dest_y_glob - dj
                        off_z = dest_z_glob - dk
                        
                        cuda.atomic.add(n_ep_received, (di, dj, dk), 1)
                        cuda.atomic.add(incoming_x, (di, dj, dk), off_x)
                        cuda.atomic.add(incoming_y, (di, dj, dk), off_y)
                        cuda.atomic.add(incoming_z, (di, dj, dk), off_z)
            
            # =================================================================
            # CREEPING TRANSPORT (Very limited)
            # =================================================================
            if n_creeping > 0:
                l_creep = dx * 0.8  # Shorter range than before (FIXME tuning (0.8 default))
                
                for _ in range(n_creeping):
                    theta = xoroshiro128p_uniform_float32(rng_states, rng_idx) * 2.0 * 3.14159
                    rnd_creep = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                    d_actual = l_creep * rnd_creep  # Uniform, not triangular
                    
                    # Creeping direction
                    creep_dir_x = math.cos(theta)
                    creep_dir_y = math.sin(theta)
                    
                    # Apply wind bias to creeping too (slight)
                    if u_horiz > 0.1:
                        creep_dir_x = creep_dir_x * 0.7 + wind_unit_x * 0.3
                        creep_dir_y = creep_dir_y * 0.7 + wind_unit_y * 0.3
                        # Renormalize
                        creep_mag = math.sqrt(creep_dir_x**2 + creep_dir_y**2)
                        if creep_mag > 0.01:
                            creep_dir_x /= creep_mag
                            creep_dir_y /= creep_mag
                    
                    dest_x_glob = src_x + (creep_dir_x * d_actual) / dx
                    dest_y_glob = src_y + (creep_dir_y * d_actual) / dy
                    
                    di = int(math.floor(dest_x_glob))
                    dj = int(math.floor(dest_y_glob))
                    dk = k
                    
                    if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                        off_x = dest_x_glob - di
                        off_y = dest_y_glob - dj
                        off_z = 0.5
                        
                        cuda.atomic.add(n_ep_received, (di, dj, dk), 1)
                        cuda.atomic.add(incoming_x, (di, dj, dk), off_x)
                        cuda.atomic.add(incoming_y, (di, dj, dk), off_y)
                        cuda.atomic.add(incoming_z, (di, dj, dk), off_z)


# Keep old transport kernel for compatibility
@cuda.jit
def transport_eps_kernel(ep_counts,
                          n_ep_received, incoming_x, incoming_y, incoming_z,
                          centroid_x, centroid_y, centroid_z,
                          u, v, w, elevation, rng_states,
                          dx, dy, dz, dt, eep):
    """Original transport kernel - kept for backward compatibility."""
    i, j, k = cuda.grid(3)
    nx, ny, nz = ep_counts.shape
    
    if i < nx and j < ny and k < nz:
        count = ep_counts[i, j, k]
        
        if count > 0:
            rng_idx = (k * (nx * ny)) + (j * nx) + i
            
            src_x = i + centroid_x[i, j, k]
            src_y = j + centroid_y[i, j, k]
            src_z = k + centroid_z[i, j, k]
            
            uc = u[i, j, k]
            vc = v[i, j, k]
            wc = w[i, j, k]
            u_horiz = math.sqrt(uc*uc + vc*vc)
            
            area_cell = dx * dy
            intensity = (count * eep) / area_cell
            
            w_star = 0.377 * math.pow(intensity * 0.001, 0.4)
            w_star = max(0.1, min(20.0, w_star))
            
            phi = w_star / (w_star + u_horiz + 1e-6)
            
            h_flame = dz + 0.0155 * math.pow(intensity, 0.4)
            stretch = 1.0
            if w_star > 1e-6:
                stretch = math.sqrt((uc*uc + vc*vc) / (w_star*w_star))
            l_scale = h_flame * max(1.0, stretch)
            
            if u_horiz > 0.5:
                l_scale = max(l_scale, dx * 1.5)
            
            n_wind = int(count * 0.7)
            n_creeping = count - n_wind
            
            if n_wind > 0:
                rnd_bifurcation = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                is_tower = rnd_bifurcation < phi
                
                rnd_dist = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                dist_travel = l_scale * (1.0 - math.sqrt(1.0 - rnd_dist))
                
                tke_est = 0.1 * (u_horiz + w_star)
                pert_mag = math.sqrt(tke_est) * 0.5
                u_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                v_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                w_pert = xoroshiro128p_normal_float32(rng_states, rng_idx) * pert_mag
                
                if is_tower:
                    total_w = w_star + wc + w_pert
                    total_u = uc * 0.2 + u_pert
                    total_v = vc * 0.2 + v_pert
                else:
                    total_w = wc + w_pert
                    total_u = uc + u_pert
                    total_v = vc + v_pert
                
                mag = math.sqrt(total_u**2 + total_v**2 + total_w**2)
                if mag > 1e-6:
                    dx_travel = (total_u / mag) * dist_travel
                    dy_travel = (total_v / mag) * dist_travel
                    dz_travel = (total_w / mag) * dist_travel
                else:
                    dx_travel = 0.0
                    dy_travel = 0.0
                    dz_travel = 0.0
                
                dest_x_glob = src_x + dx_travel / dx
                dest_y_glob = src_y + dy_travel / dy
                dest_z_glob = src_z + dz_travel / dz
                
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
            
            if n_creeping > 0:
                l_creep = max(2.0, dx * 1.2)
                
                theta = xoroshiro128p_uniform_float32(rng_states, rng_idx) * 2.0 * 3.14159
                rnd_creep = xoroshiro128p_uniform_float32(rng_states, rng_idx)
                d_actual = l_creep * (1.0 - math.sqrt(rnd_creep))
                
                dest_x_glob = src_x + (math.cos(theta) * d_actual) / dx
                dest_y_glob = src_y + (math.sin(theta) * d_actual) / dy
                
                di = int(math.floor(dest_x_glob))
                dj = int(math.floor(dest_y_glob))
                dk = k
                
                if 0 <= di < nx and 0 <= dj < ny and 0 <= dk < nz:
                    off_x = dest_x_glob - di
                    off_y = dest_y_glob - dj
                    off_z = 0.5
                    cuda.atomic.add(n_ep_received, (di, dj, dk), n_creeping)
                    cuda.atomic.add(incoming_x, (di, dj, dk), off_x * n_creeping)
                    cuda.atomic.add(incoming_y, (di, dj, dk), off_y * n_creeping)
                    cuda.atomic.add(incoming_z, (di, dj, dk), off_z * n_creeping)