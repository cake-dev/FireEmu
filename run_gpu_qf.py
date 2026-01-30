"""
run_gpu_qf.py - QUIC-Fire GPU Simulation Runner (Research Quality)

This module orchestrates the fire simulation using research-quality physics:
- Briggs (1984) plume rise theory
- Cionco (1965) canopy wind profiles
- Linn et al. (2020) QUIC-Fire EP transport

References:
- Linn et al. (2020) Environ. Model. Softw. 125, 104616
"""

import numpy as np
import os
from numba import cuda
import config_qf as config
import wind_gpu_qf as wind_gpu
import fire_gpu_qf as fire_gpu
import gpu_utils
import scipy.ndimage
import concurrent

# Set to True to use HDF5 (single file, requires h5py).
# Set to False to use Parallel NPZ (multiple files, faster without deps).
ENABLE_HDF5 = False 

@cuda.jit
def inject_ignition_kernel(n_ep_received, ignition_mask, ignition_strength):
    """
    Adds ignition energy to the n_ep_received array based on a mask.
    This runs before reaction calculation to force ignition.
    
    Parameters:
    -----------
    n_ep_received : 3D array
        EP accumulator array
    ignition_mask : 3D array
        Binary mask where 1 = ignite this cell
    ignition_strength : int
        Number of EPs to inject (typically 10000 for strong ignition)
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = n_ep_received.shape
    
    if x < nx and y < ny and z < nz:
        if ignition_mask[x, y, z] > 0:
            cuda.atomic.add(n_ep_received, (x, y, z), ignition_strength)


@cuda.jit
def dry_ignition_cells_kernel(fuel_moisture, ignition_mask):
    """
    Sets moisture to zero in ignition cells to ensure immediate ignition.
    This bypasses the moisture evaporation phase for forced ignitions.
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = fuel_moisture.shape
    
    if x < nx and y < ny and z < nz:
        if ignition_mask[x, y, z] > 0:
            fuel_moisture[x, y, z] = 0.0


def interpolate_wind(current_time, schedule):
    """
    Interpolates wind speed and direction from schedule.
    
    Parameters:
    -----------
    current_time : float
        Current simulation time in seconds
    schedule : list of tuples
        [(time_sec, speed_m_s, direction_deg), ...]
        
    Returns:
    --------
    tuple: (speed, direction)
    """
    if not schedule:
        return 0.0, 0.0
        
    if current_time <= schedule[0][0]:
        return schedule[0][1], schedule[0][2]
    if current_time >= schedule[-1][0]:
        return schedule[-1][1], schedule[-1][2]
        
    for i in range(len(schedule) - 1):
        t1, s1, d1 = schedule[i]
        t2, s2, d2 = schedule[i+1]
        
        if t1 <= current_time <= t2:
            fraction = (current_time - t1) / (t2 - t1)
            s = s1 + (s2 - s1) * fraction
            
            # Handle direction wrap-around
            diff = d2 - d1
            if diff > 180: diff -= 360
            if diff < -180: diff += 360
            d = d1 + diff * fraction
            if d < 0: d += 360
            if d >= 360: d -= 360
            return s, d
            
    return schedule[0][1], schedule[0][2]


def run_simulation(params, run_id, output_dir):
    """
    Main simulation loop for QUIC-Fire.
    
    Parameters:
    -----------
    params : dict
        Simulation parameters including:
        - qf_config: Grid and timing configuration
        - wind_schedule: List of (time, speed, direction) tuples
        - moisture: Default moisture fraction
        - ignition: Ignition data (list or ATV dict)
        - custom_fuel: 3D fuel density array
        - custom_terrain: 2D terrain elevation array
        - custom_moisture: Optional 3D moisture array
    run_id : str or int
        Identifier for this run (used for RNG seeding)
    output_dir : str
        Directory for output files
    """
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    qf_config = params.get('qf_config', {})
    
    # Wind
    wind_schedule = params.get('wind_schedule', [])
    default_speed = params.get('wind_speed', 10.0)
    default_dir = params.get('wind_dir', 0.0)
    
    # Moisture
    moisture_val = params.get('moisture', 0.1)
    
    # Ignition
    ignition_data = params.get('ignition', [])
    
    # Grid dimensions
    nx = qf_config.get('nx', config.NX)
    ny = qf_config.get('ny', config.NY)
    nz = qf_config.get('nz', config.NZ)
    dx = qf_config.get('dx', config.DX)
    dy = qf_config.get('dy', config.DY)
    dz = qf_config.get('dz', config.DZ)
    dt = qf_config.get('dt', config.DT)
    total_time = qf_config.get('sim_time', config.TOTAL_TIME)
    
    # Canopy height for Cionco profile (can be overridden)
    canopy_height = qf_config.get('canopy_height', config.CANOPY_HEIGHT)
    
    # GPU kernel configuration
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nx + threads_per_block[0] - 1) // threads_per_block[0],
        (ny + threads_per_block[1] - 1) // threads_per_block[1],
        (nz + threads_per_block[2] - 1) // threads_per_block[2]
    )
    
    tpb_2d = (8, 8)
    bpg_2d = (
        (nx + tpb_2d[0] - 1) // tpb_2d[0], 
        (ny + tpb_2d[1] - 1) // tpb_2d[1]
    )

    # =========================================================================
    # INITIALIZE HOST DATA
    # =========================================================================
    
    # --- Fuel Density ---
    if 'custom_fuel' in params:
        fuel_host = params['custom_fuel'].astype(np.float32)
        if fuel_host.shape != (nx, ny, nz):
            print(f"Resizing input fuel {fuel_host.shape} to {(nx, ny, nz)}")
            temp = np.zeros((nx, ny, nz), dtype=np.float32)
            min_x = min(nx, fuel_host.shape[0])
            min_y = min(ny, fuel_host.shape[1])
            min_z = min(nz, fuel_host.shape[2])
            temp[:min_x, :min_y, :min_z] = fuel_host[:min_x, :min_y, :min_z]
            fuel_host = temp
    else:
        # Default: uniform fuel in lowest layers
        fuel_host = np.zeros((nx, ny, nz), dtype=np.float32)
        fuel_host[:, :, 0:3] = 0.5  # 0.5 kg/m³ in bottom 3 layers

    # --- Terrain ---
    if 'custom_terrain' in params:
        elevation_host = np.ascontiguousarray(params['custom_terrain'], dtype=np.float32)
    else:
        elevation_host = np.zeros((nx, ny), dtype=np.float32)

    # Normalize terrain so minimum is at z=0
    if np.min(elevation_host) <= 0.1 and np.median(elevation_host) > 20:
        # Detected zero-padding artifact (e.g., edges are 0 but terrain is high)
        valid_mask = elevation_host > 0.1
        if np.any(valid_mask):
            elev_min = np.min(elevation_host[valid_mask])
            print(f"Detected padded terrain. Shifting by valid min: {elev_min:.1f}m")
            elevation_host = np.where(valid_mask, elevation_host - elev_min, 0)
            elevation_host = np.maximum(0, elevation_host)
    else:
        # Standard normalization
        elev_min = np.min(elevation_host)
        if elev_min != 0:
            print(f"Normalizing terrain by min: {elev_min:.1f}m")
            elevation_host = elevation_host - elev_min
        
    # Convert to meters and smooth for numerical stability
    elevation_meters = elevation_host * dz
    elevation_physics = scipy.ndimage.gaussian_filter(elevation_meters, sigma=1.0).astype(np.float32)
    
    # Z coordinates for wind profile
    z_coords_host = (np.arange(nz) * dz).astype(np.float32)

    # --- Moisture ---
    if 'custom_moisture' in params:
        fuel_moisture_host = params['custom_moisture'].astype(np.float32)
        if fuel_moisture_host.shape != (nx, ny, nz):
            temp = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val
            min_x = min(nx, fuel_moisture_host.shape[0])
            min_y = min(ny, fuel_moisture_host.shape[1])
            min_z = min(nz, fuel_moisture_host.shape[2])
            temp[:min_x, :min_y, :min_z] = fuel_moisture_host[:min_x, :min_y, :min_z]
            fuel_moisture_host = temp
    else:
        fuel_moisture_host = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val

    # =========================================================================
    # ALLOCATE GPU ARRAYS
    # =========================================================================
    
    # Terrain and coordinates
    elevation_dev = cuda.to_device(elevation_physics)
    z_coords_dev = cuda.to_device(z_coords_host)
    
    # Fuel state
    fuel_0_dev = cuda.to_device(fuel_host)  # Initial fuel (for drag calculation)
    fuel_dev = cuda.to_device(fuel_host)     # Current fuel
    fuel_moisture_dev = cuda.to_device(fuel_moisture_host)
    
    # Wind field
    u_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    v_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    w_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    # Fire state
    reaction_rate_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    time_since_ignition_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    # Sub-grid centroid tracking (Paper Section 2.6)
    centroid_x_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    centroid_y_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    centroid_z_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    ep_history_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    
    # EP transport accumulators
    incoming_x_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    incoming_y_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    incoming_z_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    n_ep_received_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    ep_counts_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)

    # =========================================================================
    # INITIALIZE GPU ARRAYS
    # =========================================================================
    
    # Initialize centroids to cell centers
    gpu_utils.init_centroid_kernel[blocks_per_grid, threads_per_block](
        centroid_x_dev, centroid_y_dev, centroid_z_dev
    )
    
    # Zero all accumulator arrays
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](ep_history_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](reaction_rate_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](time_since_ignition_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_x_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_y_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_z_dev)

    # =========================================================================
    # PREPARE IGNITION
    # =========================================================================
    
    atv_ignition_lines = []
    legacy_ignition_points = []
    
    if isinstance(ignition_data, dict) and ignition_data.get('type') == 5:
        # ATV-style moving ignition
        atv_ignition_lines = ignition_data['lines']
        print(f"Initialized ATV Ignition with {len(atv_ignition_lines)} lines.")
    elif isinstance(ignition_data, list):
        # Point ignition list
        legacy_ignition_points = ignition_data
        print(f"Initialized point ignition with {len(legacy_ignition_points)} points.")
        
    # Apply immediate (t=0) ignitions
    if legacy_ignition_points:
        temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
        temp_moist_mask = np.zeros((nx, ny, nz), dtype=np.int32)
        
        for pt in legacy_ignition_points:
            if isinstance(pt, dict):
                ix, iy, iz = int(pt['x']), int(pt['y']), int(pt['z'])
            else:
                ix, iy, iz = int(pt[0]), int(pt[1]), int(pt[2])
                
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                temp_ep[ix, iy, iz] = 10000  # Strong ignition
                temp_moist_mask[ix, iy, iz] = 1
        
        n_ep_received_dev.copy_to_device(temp_ep)
        
        # Dry the ignition cells to ensure immediate ignition
        mask_dev = cuda.to_device(temp_moist_mask)
        dry_ignition_cells_kernel[blocks_per_grid, threads_per_block](
            fuel_moisture_dev, mask_dev
        )
        cuda.synchronize()

    # =========================================================================
    # INITIALIZE RNG
    # =========================================================================
    
    if isinstance(run_id, str):
        numeric_seed = abs(hash(run_id)) % (2**32)
    else:
        numeric_seed = int(run_id)
    
    rng_states = gpu_utils.init_rng(nx * ny * nz, seed=numeric_seed)

    # =========================================================================
    # OUTPUT SETUP
    # =========================================================================
    
    # Wind slice recording heights
    target_heights = [5.0, 10.0, 15.0]
    z_indices = [int(h / dz) for h in target_heights]
    z_indices = [min(max(z, 0), nz-1) for z in z_indices]
    z_indices_dev = cuda.to_device(np.array(z_indices, dtype=np.int32))
    wind_snapshot_dev = cuda.device_array((3, 3, nx, ny), dtype=np.float32)
    
    # Timing
    total_steps = int(total_time / dt)
    vol = dx * dy * dz
    
    # Output interval
    out_int_fire = qf_config.get('out_int_fire', 100)
    expected_frames = len(range(0, total_steps, out_int_fire))
    
    print(f"Grid: {nx}x{ny}x{nz} | Cell: {dx}x{dy}x{dz}m | dt={dt}s")
    print(f"Total time: {total_time}s | Steps: {total_steps} | Output frames: {expected_frames}")
    
    # Memory-mapped storage for large outputs
    os.makedirs(output_dir, exist_ok=True)
    fuel_mm_path = os.path.join(output_dir, "temp_fuel.dat")
    rr_mm_path = os.path.join(output_dir, "temp_rr.dat")
    wind_mm_path = os.path.join(output_dir, "temp_wind.dat")
    
    fuel_mm = np.memmap(fuel_mm_path, dtype='float16', mode='w+', 
                        shape=(expected_frames, nx, ny, nz))
    rr_mm = np.memmap(rr_mm_path, dtype='float16', mode='w+', 
                      shape=(expected_frames, nx, ny, nz))
    wind_mm = np.memmap(wind_mm_path, dtype='float16', mode='w+', 
                        shape=(expected_frames, 3, 3, nx, ny))
    
    frame_idx = 0
    current_wind_speed = default_speed
    current_wind_dir = default_dir

    # =========================================================================
    # MAIN SIMULATION LOOP
    # =========================================================================
    
    try:
        for t in range(total_steps):
            current_sim_time = t * dt
            
            # -----------------------------------------------------------------
            # DYNAMIC IGNITION (ATV moving ignition)
            # -----------------------------------------------------------------
            if atv_ignition_lines:
                active_ignition_mask = np.zeros((nx, ny, nz), dtype=np.int32)
                found_active = False
                
                for line in atv_ignition_lines:
                    t_start, t_end = line['t_start'], line['t_end']
                    
                    # Check temporal overlap
                    start_overlap = max(current_sim_time, t_start)
                    end_overlap = min(current_sim_time + dt, t_end)
                    
                    if start_overlap < end_overlap:
                        found_active = True
                        total_duration = max(t_end - t_start, 1e-6)
                        
                        # Parametric position along line
                        alpha_start = (start_overlap - t_start) / total_duration
                        alpha_end = (end_overlap - t_start) / total_duration
                        
                        p_start_x = line['x_start'] + (line['x_end'] - line['x_start']) * alpha_start
                        p_start_y = line['y_start'] + (line['y_end'] - line['y_start']) * alpha_start
                        p_end_x = line['x_start'] + (line['x_end'] - line['x_start']) * alpha_end
                        p_end_y = line['y_start'] + (line['y_end'] - line['y_start']) * alpha_end
                        
                        # Rasterize the line segment
                        dist = np.sqrt((p_end_x - p_start_x)**2 + (p_end_y - p_start_y)**2)
                        steps = int(max(dist / min(dx, dy) * 2, 2))
                        
                        for s in range(steps + 1):
                            lerp = s / steps
                            curr_x = p_start_x + (p_end_x - p_start_x) * lerp
                            curr_y = p_start_y + (p_end_y - p_start_y) * lerp
                            
                            ix = int(curr_x / dx)
                            iy = int(curr_y / dy)
                            
                            if 0 <= ix < nx and 0 <= iy < ny:
                                # Ignite at surface (z=0) and one layer above
                                active_ignition_mask[ix, iy, 0] = 1
                                if nz > 1:
                                    active_ignition_mask[ix, iy, 1] = 1
                
                if found_active:
                    mask_dev = cuda.to_device(active_ignition_mask)
                    inject_ignition_kernel[blocks_per_grid, threads_per_block](
                        n_ep_received_dev, mask_dev, 10000
                    )
                    dry_ignition_cells_kernel[blocks_per_grid, threads_per_block](
                        fuel_moisture_dev, mask_dev
                    )
                    cuda.synchronize()

            # -----------------------------------------------------------------
            # WIND UPDATE
            # -----------------------------------------------------------------
            if wind_schedule:
                current_wind_speed, current_wind_dir = interpolate_wind(
                    current_sim_time, wind_schedule
                )
            else:
                current_wind_speed, current_wind_dir = default_speed, default_dir
            
            # Convert meteorological direction to math angle
            # Met: 0=N, 90=E, wind FROM that direction
            # Math: 0=E, CCW positive, wind TO that direction
            wind_rad = np.radians(270 - current_wind_dir)
            
            # -----------------------------------------------------------------
            # PHYSICS PIPELINE
            # -----------------------------------------------------------------
            
            # 1. Apply vegetation drag (Cionco profile)
            wind_gpu.apply_drag_kernel_research[blocks_per_grid, threads_per_block](
                u_dev, fuel_dev, fuel_0_dev, z_coords_dev,
                canopy_height, current_wind_speed, 10.0,  # z_ref = 10m
                config.K_VON_KARMAN, config.Z0, dz
            )
            
            # 2. Rotate wind to correct direction
            wind_gpu.rotate_wind_kernel[blocks_per_grid, threads_per_block](
                u_dev, v_dev, wind_rad
            )
            
            # 3. Reset vertical velocity
            wind_gpu.reset_w_kernel[blocks_per_grid, threads_per_block](w_dev)
            
            # 4. Apply terrain-following vertical velocity
            wind_gpu.project_wind_over_terrain_kernel[blocks_per_grid, threads_per_block](
                u_dev, v_dev, w_dev, elevation_dev, dx, dy, dz
            )
            
            # 5. Apply fire-induced buoyancy (Briggs plume rise)
            wind_gpu.apply_buoyancy_column_kernel_research[bpg_2d, tpb_2d](
                w_dev, reaction_rate_dev, u_dev, v_dev,
                dx, dy, dz, config.G, config.RHO_AIR, config.CP_AIR,
                config.T_AMBIENT, config.H_WOOD
            )
            
            # 6. Compute reaction rate and fuel consumption
            fire_gpu.compute_reaction_and_fuel_kernel[blocks_per_grid, threads_per_block](
                fuel_dev, fuel_moisture_dev,
                n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
                centroid_x_dev, centroid_y_dev, centroid_z_dev, ep_history_dev,
                time_since_ignition_dev, reaction_rate_dev, ep_counts_dev,
                dt, config.CM, config.T_BURNOUT, config.H_WOOD, vol,
                config.C_RAD_LOSS, config.EEP,
                config.CP_WOOD, config.T_CRIT, config.T_AMBIENT, config.H_H2O_EFF
            )
            
            # 7. Zero EP accumulators for next transport step
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_x_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_y_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_z_dev)
            cuda.synchronize()
            
            # 8. Transport EPs to neighboring cells
            fire_gpu.transport_eps_kernel_v2[blocks_per_grid, threads_per_block](
                ep_counts_dev,
                n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
                centroid_x_dev, centroid_y_dev, centroid_z_dev,
                u_dev, v_dev, w_dev, elevation_dev,
                rng_states, dx, dy, dz, dt, config.EEP, wind_rad
            )

            # -----------------------------------------------------------------
            # PROGRESS OUTPUT
            # -----------------------------------------------------------------
            if t % 50 == 0:
                rr_host_temp = reaction_rate_dev.copy_to_host()
                ignited_mask = rr_host_temp > 0.001
                n_burning = np.sum(ignited_mask)
                
                msg = f"Step {t+1}/{total_steps} | t={current_sim_time:.1f}s | "
                msg += f"Wind: {current_wind_speed:.1f} m/s @ {current_wind_dir:.0f}° | "
                msg += f"Burning cells: {n_burning}"
                
                if n_burning > 0:
                    avg_rr = np.mean(rr_host_temp[ignited_mask])
                    max_rr = np.max(rr_host_temp)
                    msg += f" | RR: avg={avg_rr:.4f}, max={max_rr:.4f}"

                # print percent fuel consumed
                fuel_host_temp = fuel_dev.copy_to_host()
                total_fuel = np.sum(fuel_host_temp + rr_host_temp * dt)
                initial_fuel = np.sum(fuel_host)
                pct_consumed = 100.0 * (initial_fuel - total_fuel) / initial_fuel
                msg += f" | Fuel consumed: {pct_consumed:.2f}%"
                
                print(msg)

            # -----------------------------------------------------------------
            # DATA OUTPUT
            # -----------------------------------------------------------------
            if t % out_int_fire == 0 and frame_idx < expected_frames:
                rho_host = fuel_dev.copy_to_host()
                rr_host = reaction_rate_dev.copy_to_host()
                
                fuel_mm[frame_idx] = rho_host.astype(np.float16)
                rr_mm[frame_idx] = rr_host.astype(np.float16)
                
                wind_gpu.extract_wind_slices_kernel[bpg_2d, tpb_2d](
                    u_dev, v_dev, w_dev, wind_snapshot_dev, z_indices_dev
                )
                w_slice_host = wind_snapshot_dev.copy_to_host()
                wind_mm[frame_idx] = w_slice_host.astype(np.float16)
                
                frame_idx += 1

        # =====================================================================
        # SAVE OUTPUTS
        # =====================================================================
        print("Finalizing output...")
        
        fuel_mm.flush()
        rr_mm.flush()
        wind_mm.flush()

        if ENABLE_HDF5:
            # --- OPTION A: HDF5 (Single High-Performance File) ---
            print("Saving as HDF5...")
            import h5py
            h5_path = os.path.join(output_dir, "simulation_output.h5")
            
            with h5py.File(h5_path, 'w') as f:
                # Compression 'lzf' is extremely fast, 'gzip' is standard but slower
                f.create_dataset('fuel', data=fuel_mm[:frame_idx], compression='lzf')
                f.create_dataset('reaction_rate', data=rr_mm[:frame_idx], compression='lzf')
                f.create_dataset('wind_local', data=wind_mm[:frame_idx], compression='lzf')
                
                # Scalars and Metadata
                f.attrs['wind_speed'] = current_wind_speed
                f.attrs['wind_dir'] = current_wind_dir
                f.attrs['moisture'] = moisture_val
                f.attrs['dt'] = dt
                f.attrs['total_time'] = total_time
                
                f.create_dataset('terrain', data=elevation_host)
                f.create_dataset('custom_terrain', data=elevation_host)
                f.create_dataset('wind_heights', data=np.array(target_heights))
                f.create_dataset('grid_shape', data=np.array([nx, ny, nz]))
                f.create_dataset('cell_size', data=np.array([dx, dy, dz]))
                
            print(f"Saved HDF5: {h5_path}")

        else:
            # --- OPTION B: Parallel NPZ (Split Files) ---
            print("Saving as Parallel NPZ...")
            
            def save_chunk(filename, **kwargs):
                path = os.path.join(output_dir, filename)
                np.savez_compressed(path, **kwargs)
                return path

            tasks = [
                {
                    "filename": "outputs_fuel.npz",
                    "fuel": fuel_mm[:frame_idx]
                },
                {
                    "filename": "outputs_rr.npz",
                    "reaction_rate": rr_mm[:frame_idx]
                },
                {
                    "filename": "outputs_wind.npz",
                    "wind_local": wind_mm[:frame_idx]
                },
                {
                    "filename": "outputs_meta.npz",
                    "wind_speed": np.array([current_wind_speed]),
                    "wind_dir": np.array([current_wind_dir]),
                    "moisture": np.array([moisture_val]),
                    "terrain": elevation_host,
                    "custom_terrain": elevation_host,
                    "wind_heights": np.array(target_heights),
                    "grid_shape": np.array([nx, ny, nz]),
                    "cell_size": np.array([dx, dy, dz]),
                    "dt": np.array([dt]),
                    "total_time": np.array([total_time])
                }
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(save_chunk, **t) for t in tasks]
                for future in concurrent.futures.as_completed(futures):
                    print(f"Saved: {future.result()}")

    finally:
        # Cleanup GPU memory
        del u_dev, v_dev, w_dev, fuel_dev, fuel_0_dev
        del reaction_rate_dev, wind_snapshot_dev
        del fuel_moisture_dev, time_since_ignition_dev
        del centroid_x_dev, centroid_y_dev, centroid_z_dev
        del ep_history_dev, n_ep_received_dev, ep_counts_dev
        del incoming_x_dev, incoming_y_dev, incoming_z_dev
        
        # Close memmaps
        del fuel_mm, rr_mm, wind_mm
        
        # Remove temp files
        for f in [fuel_mm_path, rr_mm_path, wind_mm_path]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass