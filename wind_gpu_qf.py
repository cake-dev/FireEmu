"""
wind_gpu_qf.py - Wind Solver for QUIC-Fire

This implementation addresses the physics requirements for research publication,
based on:
- Briggs (1984) plume rise theory
- Cionco (1965) canopy wind profiles  
- Davidson (1989) trajectory/dilution model
- Linn et al. (2020) QUIC-Fire paper

Key improvements over wind_gpu_stable.py:
1. Dimensionally correct buoyancy velocity formulation
2. Height-dependent Cionco profile within canopy
3. Proper plume entrainment decay
4. Height-limited terrain influence
5. Plume bending under crosswind
"""

import math
from numba import cuda


@cuda.jit
def extract_wind_slices_kernel(u, v, w, out_buffer, z_indices):
    """
    Extracts U, V, W vectors at specific z-indices for output/visualization.
    Unchanged from stable version.
    """
    i, j = cuda.grid(2)
    nx, ny, nz = u.shape
    n_layers = out_buffer.shape[0]
    
    if i < nx and j < ny:
        for l in range(n_layers):
            k = z_indices[l]
            if k >= 0 and k < nz:
                out_buffer[l, 0, i, j] = u[i, j, k]
                out_buffer[l, 1, i, j] = v[i, j, k]
                out_buffer[l, 2, i, j] = w[i, j, k]


@cuda.jit
def project_wind_over_terrain_kernel(u, v, w, elevation, dx, dy, dz):
    """
    Applies terrain-following vertical velocity component.
    
    Physics: w_terrain = u * (∂z_s/∂x) + v * (∂z_s/∂y)
    
    Improvement: Height-limited influence - terrain effect decays with altitude
    to prevent unrealistic vertical velocities at high levels.
    
    Reference: Standard kinematic boundary condition for terrain-following coordinates
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        # Central difference for terrain gradient
        i_next = min(i + 1, nx - 1)
        i_prev = max(i - 1, 0)
        j_next = min(j + 1, ny - 1)
        j_prev = max(j - 1, 0)
        
        dz_dx = (elevation[i_next, j] - elevation[i_prev, j]) / (2.0 * dx)
        dz_dy = (elevation[i, j_next] - elevation[i, j_prev]) / (2.0 * dy)
        
        # Basic terrain-induced vertical velocity
        w_terrain = u[i, j, k] * dz_dx + v[i, j, k] * dz_dy
        
        # Height above local terrain (in grid units)
        local_terrain_height = elevation[i, j]
        z_above_terrain = (k * dz) - local_terrain_height
        
        # Terrain influence decay - effect diminishes with height
        # Using exponential decay with scale height of ~5 terrain heights
        terrain_scale = max(1.0, local_terrain_height) * 5.0
        if z_above_terrain > 0:
            terrain_influence = math.exp(-z_above_terrain / terrain_scale)
        else:
            terrain_influence = 1.0
        
        # Clamp influence to reasonable range
        terrain_influence = max(0.0, min(1.0, terrain_influence))
        
        w[i, j, k] += w_terrain * terrain_influence


@cuda.jit
def apply_drag_kernel_research(u, fuel_density, fuel_density_0, z_coords, 
                                canopy_height, u_ref, z_ref, k_vk, z0, dz):
    """
    Applies vegetation drag using proper Cionco (1965) exponential profile.
    
    Within canopy (z < H):
        u(z) = u_H * exp[α * (z/H - 1)]
    
    Above canopy (z >= H):
        u(z) = (u*/κ) * ln(z/z0)
    
    The attenuation coefficient α varies with fuel density:
    - Sparse fuel (low LAI): α ≈ 0.5-1.0
    - Dense fuel (high LAI): α ≈ 2.0-3.0
    
    As fuel burns, profile transitions from Cionco to log-law.
    
    References:
    - Cionco (1965) J. Appl. Meteorol. 4, 517-522
    - Paper Fig. 1 and Section 2.2
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        z = z_coords[k] + 0.5 * dz
        
        # Friction velocity from reference conditions
        if z_ref > z0:
            u_star = u_ref * k_vk / math.log(z_ref / z0)
        else:
            u_star = 0.1 * u_ref  # Fallback
        
        # Log-law profile (no vegetation case)
        if z <= z0:
            u_log = 0.0
        else:
            u_log = (u_star / k_vk) * math.log(z / z0)
        
        # Check if this cell has/had fuel
        rho_0 = fuel_density_0[i, j, k]
        rho_f = fuel_density[i, j, k]
        
        if rho_0 > 0.01:  # Cell originally had fuel
            # Determine local canopy properties
            # Attenuation coefficient scales with fuel density
            # Typical range: 0.5 (sparse) to 3.0 (dense)
            alpha_max = 2.5
            alpha = alpha_max * min(1.0, rho_0 / 2.0)  # Normalize to ~2 kg/m³
            
            # Current fuel fraction
            fuel_frac = rho_f / rho_0
            
            # Effective canopy height for this column
            # In practice should come from fuel data; using parameter for now
            h_canopy = canopy_height
            
            if z < h_canopy and h_canopy > 0:
                # WITHIN CANOPY: Cionco exponential profile
                # u(z) = u_H * exp[α * (z/H - 1)]
                
                # Wind at canopy top (matches log profile)
                if h_canopy > z0:
                    u_canopy_top = (u_star / k_vk) * math.log(h_canopy / z0)
                else:
                    u_canopy_top = u_log
                
                # Cionco profile
                u_cionco = u_canopy_top * math.exp(alpha * (z / h_canopy - 1.0))
                
                # Blend between Cionco (full fuel) and log (no fuel)
                u[i, j, k] = u_cionco * fuel_frac + u_log * (1.0 - fuel_frac)
            else:
                # ABOVE CANOPY: Log profile, possibly with roughness modification
                u[i, j, k] = u_log
        else:
            # No fuel - pure log profile
            u[i, j, k] = u_log


@cuda.jit 
def apply_drag_kernel(u, fuel_density, fuel_density_0, z_coords, u_ref, z_ref, k_vk, z0, dz):
    """
    Original simplified drag kernel - kept for backward compatibility.
    Use apply_drag_kernel_research for publication-quality simulations.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        z = z_coords[k] + 0.5 * dz
        
        if z <= z0:
            u_log = 0.0
        else:
            u_star = u_ref * k_vk / math.log(z_ref / z0)
            u_log = (u_star / k_vk) * math.log(z / z0)
        
        if fuel_density_0[i, j, k] > 0:
            attenuation = 0.3
            fuel_frac = fuel_density[i, j, k] / fuel_density_0[i, j, k]
            current_attenuation = 1.0 - fuel_frac * (1.0 - attenuation)
            u[i, j, k] = u_log * current_attenuation
        else:
            u[i, j, k] = u_log


@cuda.jit
def apply_buoyancy_column_kernel_research(w, reaction_rate, u, v,
                                          dx, dy, dz, g, rho_air, cp_air, 
                                          t_ambient, h_wood):
    """
    Research-quality buoyancy/plume rise implementation.
    
    Based on Briggs (1984) plume rise theory and Davidson (1989):
    
    Buoyancy flux: FB = (g/Ta) * (E / (ρa * cp_a))  [m⁴/s³]
    
    Near-source plume velocity (Briggs):
        w* = (FB / (π * β² * z))^(1/3)
    
    Where β ≈ 0.4-0.6 is the entrainment coefficient.
    
    Plume bending in crosswind:
        The plume trajectory bends downwind, so vertical velocity
        decreases more rapidly when horizontal wind is strong.
    
    Entrainment:
        As plume rises, it entrains ambient air and decelerates.
        Rate depends on both height and horizontal velocity.
    
    References:
    - Briggs (1984) Atmospheric Science and Power Production, Ch. 8
    - Davidson (1989) Atmos. Environ. 23, 341-349
    - Paper Eq. in Section 2.2
    """
    # 2D Grid - We loop over Z inside the kernel to carry momentum up
    i, j = cuda.grid(2)
    nx, ny, nz = w.shape
    
    if i < nx and j < ny:
        # Entrainment coefficient (Briggs recommends 0.4-0.6)
        beta = 0.5
        
        # State variable for the updraft velocity accumulating in this column
        current_updraft = 0.0
        
        # Track cumulative buoyancy for plume merging effects
        cumulative_buoyancy = 0.0
        
        # Loop from ground (k=0) to top (k=nz-1)
        for k in range(nz):
            z = (k + 0.5) * dz  # Height at cell center
            
            # Get horizontal wind at this level for plume bending
            u_horiz = math.sqrt(u[i, j, k]**2 + v[i, j, k]**2)
            
            # --- ENTRAINMENT DECAY ---
            # Plume velocity decays due to entrainment of ambient air
            # Rate depends on plume size (grows with height) and wind speed
            
            # Base decay rate from vertical entrainment
            base_decay = 0.92  # ~8% loss per layer
            
            # Enhanced decay in strong crosswind (plume dilutes faster)
            wind_factor = 1.0 + 0.02 * u_horiz  # More wind = more entrainment
            
            decay_rate = base_decay / wind_factor
            decay_rate = max(0.7, min(0.98, decay_rate))  # Clamp to reasonable range
            
            current_updraft *= decay_rate
            
            # --- ADD NEW BUOYANCY FROM THIS CELL ---
            rr = reaction_rate[i, j, k]
            
            if rr > 0:
                # Heat release rate [W] = R [kg/m³/s] * H [J/kg] * V [m³]
                E = rr * h_wood * (dx * dy * dz)
                
                # Buoyancy flux [m⁴/s³]
                # FB = (g/Ta) * (E / (ρa * cpa))
                FB = (g / t_ambient) * (E / (rho_air * cp_air))
                
                cumulative_buoyancy += FB
                
                # Briggs near-source plume velocity [m/s]
                # w* = (FB / (π * β² * z_eff))^(1/3)
                #
                # z_eff is effective height - cannot be zero
                # Use cell size as minimum (virtual origin concept)
                z_eff = max(dz, z)
                
                w_induced = (FB / (math.pi * beta * beta * z_eff)) ** (1.0/3.0)
                
                # Cap induced velocity to physical limits
                # Forest fires typically produce 5-20 m/s updrafts at source
                w_induced = min(w_induced, 25.0)
                
                # Add to column updraft
                current_updraft += w_induced
            
            # --- PLUME BENDING EFFECT ---
            # Strong horizontal wind bends plume, reducing effective vertical velocity
            # Based on plume trajectory angle: tan(θ) = w/U
            if current_updraft > 0.1 and u_horiz > 0.5:
                # Froude-like ratio
                plume_angle = math.atan2(current_updraft, u_horiz)
                
                # Reduce vertical velocity for bent plumes
                # This is a simplification of full trajectory calculation
                bending_factor = math.sin(plume_angle)
                bending_factor = max(0.3, bending_factor)  # Don't reduce too much
                
                current_updraft *= bending_factor
            
            # --- APPLY TO WIND FIELD ---
            # Add buoyancy-induced updraft to existing w (which may have terrain effects)
            w[i, j, k] += current_updraft


@cuda.jit
def apply_buoyancy_column_kernel(w, reaction_rate, dx, dy, dz, g, rho_air, cp_air, t_ambient, h_wood):
    """
    Original simplified buoyancy kernel - kept for backward compatibility.
    Use apply_buoyancy_column_kernel_research for publication-quality simulations.
    """
    i, j = cuda.grid(2)
    nx, ny, nz = w.shape
    
    if i < nx and j < ny:
        current_updraft = 0.0
        
        for k in range(nz):
            current_updraft *= 0.90
            
            rr = reaction_rate[i, j, k]
            if rr > 0:
                E = rr * h_wood * (dx * dy * dz)
                FB = g * E / (math.pi * rho_air * cp_air * t_ambient)
                w_induced = FB**(1.0/3.0)
                current_updraft += w_induced
            
            w[i, j, k] += current_updraft


@cuda.jit
def rotate_wind_kernel(u, v, wind_rad):
    """
    Rotates wind from u-only (aligned with x-axis) to arbitrary direction.
    
    Input: u contains wind magnitude, v is zero
    Output: u = magnitude * cos(θ), v = magnitude * sin(θ)
    
    Where θ is meteorological wind direction converted to math angle.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if i < nx and j < ny and k < nz:
        u_mag = u[i, j, k]
        u[i, j, k] = u_mag * math.cos(wind_rad)
        v[i, j, k] = u_mag * math.sin(wind_rad)


@cuda.jit
def reset_w_kernel(w):
    """
    Resets vertical velocity field to zero.
    Called before applying terrain and buoyancy effects each timestep.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = w.shape
    if i < nx and j < ny and k < nz:
        w[i, j, k] = 0.0


@cuda.jit
def compute_divergence_kernel(u, v, w, divergence, dx, dy, dz):
    """
    Computes velocity field divergence for mass conservation check.
    
    div(V) = ∂u/∂x + ∂v/∂y + ∂w/∂z
    
    For incompressible flow, div(V) should be ~0.
    This is diagnostic only - full mass-consistent solver would be needed
    for true QUIC-URB compatibility.
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if 1 <= i < nx-1 and 1 <= j < ny-1 and 1 <= k < nz-1:
        du_dx = (u[i+1, j, k] - u[i-1, j, k]) / (2.0 * dx)
        dv_dy = (v[i, j+1, k] - v[i, j-1, k]) / (2.0 * dy)
        dw_dz = (w[i, j, k+1] - w[i, j, k-1]) / (2.0 * dz)
        
        divergence[i, j, k] = du_dx + dv_dy + dw_dz


@cuda.jit
def apply_convergence_from_plumes_kernel(u, v, w, reaction_rate, 
                                          dx, dy, dz, g, rho_air, cp_air, 
                                          t_ambient, h_wood):
    """
    Applies horizontal convergence toward fire plumes.
    
    Paper Section 2.2:
    "The displacement of the heated air draws in adjacent air to fill 
    the vacated volume, producing a convergence zone."
    
    This creates the "draw" effect between fires that is crucial for
    fire-fire interaction in prescribed burning.
    
    Simple approximation: horizontal inflow proportional to local updraft
    """
    i, j, k = cuda.grid(3)
    nx, ny, nz = u.shape
    
    if 1 <= i < nx-1 and 1 <= j < ny-1 and k < nz:
        # Only apply convergence in lower atmosphere where fires burn
        z = (k + 0.5) * dz
        if z > 50.0:  # Above 50m, convergence effect is weak
            return
        
        # Check neighboring cells for fire activity
        rr_center = reaction_rate[i, j, k]
        
        # Compute gradient of reaction rate to find fire direction
        drr_dx = (reaction_rate[i+1, j, k] - reaction_rate[i-1, j, k]) / (2.0 * dx)
        drr_dy = (reaction_rate[i, j+1, k] - reaction_rate[i, j-1, k]) / (2.0 * dy)
        
        # Magnitude of fire gradient
        grad_mag = math.sqrt(drr_dx**2 + drr_dy**2)
        
        if grad_mag > 1e-6:
            # Inflow velocity toward fire (simplified)
            # Scale by height above ground (weaker at height)
            height_factor = max(0.1, 1.0 - z / 30.0)
            
            # Convergence velocity ~1-3 m/s near strong fires
            v_converge = 2.0 * min(grad_mag * 100.0, 1.0) * height_factor
            
            # Add convergence to u, v (toward higher reaction rate)
            u[i, j, k] += v_converge * (drr_dx / grad_mag)
            v[i, j, k] += v_converge * (drr_dy / grad_mag)