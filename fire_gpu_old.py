import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

@cuda.jit
def compute_reaction_and_fuel_kernel(fuel_density, fuel_moisture, n_ep_received, 
                                     time_since_ignition, reaction_rate, ep_counts, 
                                     dt, cm, t_burnout, h_wood, vol, c_rad_loss, eep):
    i, j, k = cuda.grid(3)
    nx, ny, nz = fuel_density.shape
    
    if i < nx and j < ny and k < nz:
        rho_f = fuel_density[i, j, k]
        
        # --- Reaction Rate Logic ---
        rr = 0.0
        # Check if cell has fuel
        if rho_f > 1e-4: # Threshold to avoid processing empty cells
            has_energy = n_ep_received[i, j, k] > 0
            is_burning = time_since_ignition[i, j, k] > 0
            
            if has_energy or is_burning:
                # If just ignited, set time to small epsilon
                if time_since_ignition[i, j, k] == 0:
                    time_since_ignition[i, j, k] = 0.01
                
                time_since_ignition[i, j, k] += dt
                
                if time_since_ignition[i, j, k] <= t_burnout:
                    # Constants locally
                    rho_o2 = 0.25
                    sigma = 1.0
                    lambda_s = 1.0
                    psi = 1.0
                    
                    rr = cm * rho_f * rho_o2 * psi * sigma * lambda_s
        
        reaction_rate[i, j, k] = rr
        
        # --- Update Fuel ---
        if rr > 0:
            change = rr * dt
            # Use max to prevent negative fuel
            fuel_density[i, j, k] = max(0.0, fuel_density[i, j, k] - change)
            
            # --- Generate EP Counts ---
            q_net = rr * h_wood * vol * dt
            
            # Stochastic rounding for EP generation to prevent "flickering" at low intensities
            expected_n_ep = (q_net * (1.0 - c_rad_loss)) / eep
            n_ep_base = int(expected_n_ep)
            remainder = expected_n_ep - n_ep_base
            
            # Simple pseudo-random check for remainder (using position as seed for stability)
            # This is deterministic but varies spatially, preventing "pulsing"
            if (i + j + k) % 100 < (remainder * 100):
                n_ep_base += 1
                
            ep_counts[i, j, k] = n_ep_base
        else:
            ep_counts[i, j, k] = 0

@cuda.jit
def transport_eps_kernel(ep_counts, n_ep_received, u, v, w, rng_states, dx, dy, dz):
    """
    Moves packets from ep_counts (source) to n_ep_received (dest)
    using atomic adds to avoid race conditions.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = ep_counts.shape
    
    if i < nx and j < ny and k < nz:
        count = ep_counts[i, j, k]
        
        if count > 0:
            # Flatten index for RNG
            # Stride must match grid dimensions exactly
            rng_idx = (k * (nx * ny)) + (j * nx) + i
            
            # --- 1. Wind-Driven Transport (Convection) ---
            # 70% of energy moves with the wind
            n_wind = int(count * 0.7) 
            n_creeping = count - n_wind
            
            if n_wind > 0:
                uc = u[i, j, k]
                vc = v[i, j, k]
                wc = w[i, j, k]
                
                # We loop through packets or groups of packets to diffuse them
                # Optimization: Move them in a single averaged chunk for speed
                # but add substantial noise (turbulence)
                
                # Noise scale: Higher = more spread, less jumping
                scale = 2.0 
                
                # Generate unique noise for this cell's "wind packet"
                up = xoroshiro128p_normal_float32(rng_states, rng_idx)
                vp = xoroshiro128p_normal_float32(rng_states, rng_idx)
                wp = xoroshiro128p_normal_float32(rng_states, rng_idx)
                
                # Calculate travel distance (time step is implicitly 1s here)
                dx_travel = (uc + up * scale) 
                dy_travel = (vc + vp * scale) 
                # Bias upward (heat rises) + random vertical turbulence
                dz_travel = (wc + wp * scale) + 2.0 
                
                dest_i = int(i + dx_travel / dx)
                dest_j = int(j + dy_travel / dy)
                dest_k = int(k + dz_travel / dz)
                
                # Bounds check
                if 0 <= dest_i < nx and 0 <= dest_j < ny and 0 <= dest_k < nz:
                    cuda.atomic.add(n_ep_received, (dest_i, dest_j, dest_k), n_wind)

            # --- 2. Creeping Transport (Diffusion/Radiation) ---
            # Moves to immediate neighbors (radius 1.5)
            if n_creeping > 0:
                # Different RNG call to avoid correlation with wind direction
                theta = xoroshiro128p_uniform_float32(rng_states, rng_idx) * 2.0 * 3.14159
                dist = 1.5 # Meters
                
                dx_c = math.cos(theta) * dist
                dy_c = math.sin(theta) * dist
                
                # Creeping is mostly horizontal, slight vertical chance
                dz_c = (xoroshiro128p_uniform_float32(rng_states, rng_idx) - 0.2) 
                
                dest_i_c = int(i + dx_c / dx)
                dest_j_c = int(j + dy_c / dy)
                dest_k_c = int(k + dz_c / dz)
                
                if 0 <= dest_i_c < nx and 0 <= dest_j_c < ny and 0 <= dest_k_c < nz:
                    cuda.atomic.add(n_ep_received, (dest_i_c, dest_j_c, dest_k_c), n_creeping)