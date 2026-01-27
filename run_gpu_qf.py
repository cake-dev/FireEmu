import numpy as np
import os
from numba import cuda
import config_stable as config
import wind_gpu_stable as wind_gpu
import fire_gpu_stable as fire_gpu
import gpu_utils
import scipy.ndimage
# from quicfire_io import QuicFireIO, QuicFireCSVWriter

@cuda.jit
def inject_ignition_kernel(n_ep_received, ignition_mask, ignition_strength):
    """
    Adds ignition energy to the n_ep_received array based on a mask.
    This runs before reaction calculation to force ignition.
    """
    x, y, z = cuda.grid(3)
    nx, ny, nz = n_ep_received.shape
    
    if x < nx and y < ny and z < nz:
        if ignition_mask[x, y, z] > 0:
            # Add massive energy packets to force ignition
            n_ep_received[x, y, z] += ignition_strength

def interpolate_wind(current_time, schedule):
    """Interpolates wind speed and direction from schedule [(t, s, d), ...]."""
    if not schedule:
        return 0.0, 0.0 # Default fallback
        
    if current_time <= schedule[0][0]:
        return schedule[0][1], schedule[0][2]
    if current_time >= schedule[-1][0]:
        return schedule[-1][1], schedule[0][2]
        
    for i in range(len(schedule) - 1):
        t1, s1, d1 = schedule[i]
        t2, s2, d2 = schedule[i+1]
        
        if t1 <= current_time <= t2:
            fraction = (current_time - t1) / (t2 - t1)
            s = s1 + (s2 - s1) * fraction
            
            diff = d2 - d1
            if diff > 180: diff -= 360
            if diff < -180: diff += 360
            d = d1 + diff * fraction
            if d < 0: d += 360
            if d >= 360: d -= 360
            return s, d
            
    return schedule[0][1], schedule[0][2]

def run_simulation(params, run_id, output_dir):
    # --- PARAMS & CONFIG ---
    qf_config = params.get('qf_config', {})
    
    # Wind Schedule
    wind_schedule = params.get('wind_schedule', [])
    default_speed = params.get('wind_speed', 10.0)
    default_dir = params.get('wind_dir', 0.0)
    
    moisture_val = params.get('moisture', 0.1)
    
    # Ignition Data (Can be list or dict for ATV)
    ignition_data = params.get('ignition', [])
    
    # Dimensions from Config or Params
    nx = qf_config.get('nx', config.NX)
    ny = qf_config.get('ny', config.NY)
    nz = qf_config.get('nz', config.NZ)
    dx = qf_config.get('dx', config.DX)
    dy = qf_config.get('dy', config.DY)
    dz = qf_config.get('dz', config.DZ)
    dt = qf_config.get('dt', config.DT)
    total_time = qf_config.get('sim_time', config.TOTAL_TIME)
    
    # Kernel Config
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nx + threads_per_block[0] - 1) // threads_per_block[0],
        (ny + threads_per_block[1] - 1) // threads_per_block[1],
        (nz + threads_per_block[2] - 1) // threads_per_block[2]
    )

    # --- 1. SETUP HOST DATA ---
    # Fuel Density
    if 'custom_fuel' in params:
        fuel_host = params['custom_fuel'] 
        if fuel_host.shape != (nx, ny, nz):
            print(f"Resizing input fuel {fuel_host.shape} to {(nx, ny, nz)}")
            temp = np.zeros((nx, ny, nz), dtype=np.float32)
            min_x = min(nx, fuel_host.shape[0])
            min_y = min(ny, fuel_host.shape[1])
            min_z = min(nz, fuel_host.shape[2])
            temp[:min_x, :min_y, :min_z] = fuel_host[:min_x, :min_y, :min_z]
            fuel_host = temp
    else:
        fuel_host = np.zeros((nx, ny, nz), dtype=np.float32)

    # Terrain
    if 'custom_terrain' in params:
        elevation_host = np.ascontiguousarray(params['custom_terrain'])
    else:
        elevation_host = np.zeros((nx, ny), dtype=np.float32)
        
    elevation_meters = elevation_host * dz
    elevation_physics = scipy.ndimage.gaussian_filter(elevation_meters, sigma=1.0)
    z_coords_host = np.arange(nz) * dz

    # --- 2. ALLOCATE GPU ---
    elevation_dev = cuda.to_device(elevation_physics)
    z_coords_dev = cuda.to_device(z_coords_host)
    fuel_0_dev = cuda.to_device(fuel_host)
    fuel_dev = cuda.to_device(fuel_host)
    u_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    v_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    w_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    reaction_rate_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    time_since_ignition_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    centroid_x_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    centroid_y_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    centroid_z_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    ep_history_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    
    gpu_utils.init_centroid_kernel[blocks_per_grid, threads_per_block](
        centroid_x_dev, centroid_y_dev, centroid_z_dev
    )
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](ep_history_dev)

    incoming_x_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    incoming_y_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    incoming_z_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    n_ep_received_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    ep_counts_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    
    # Moisture Data
    if 'custom_moisture' in params:
        fuel_moisture_host = params['custom_moisture']
        if fuel_moisture_host.shape != (nx, ny, nz):
             temp = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val
             min_x = min(nx, fuel_moisture_host.shape[0])
             min_y = min(ny, fuel_moisture_host.shape[1])
             min_z = min(nz, fuel_moisture_host.shape[2])
             temp[:min_x, :min_y, :min_z] = fuel_moisture_host[:min_x, :min_y, :min_z]
             fuel_moisture_host = temp
    else:
        fuel_moisture_host = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val
        
    fuel_moisture_dev = cuda.to_device(fuel_moisture_host)

    # Initialize accumulation arrays
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](reaction_rate_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](time_since_ignition_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)

    # --- IGNITION PREP ---
    # Handle both Legacy list and ATV dict structure
    atv_ignition_lines = []
    legacy_ignition_points = []
    
    if isinstance(ignition_data, dict) and ignition_data.get('type') == 5:
        atv_ignition_lines = ignition_data['lines']
        print(f"Initialized ATV Ignition with {len(atv_ignition_lines)} lines.")
    elif isinstance(ignition_data, list):
        legacy_ignition_points = ignition_data
        
    # Apply Legacy Ignition (Immediate t=0)
    temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
    if legacy_ignition_points:
        for pt in legacy_ignition_points:
            if isinstance(pt, dict): ix, iy, iz = pt['x'], pt['y'], pt['z']
            else: ix, iy, iz = pt
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                print(iz)
                temp_ep[ix, iy, iz] = 10000 
    
    n_ep_received_dev.copy_to_device(temp_ep)

    if isinstance(run_id, str):
        numeric_seed = abs(hash(run_id)) % (2**32) 
    else:
        numeric_seed = int(run_id)

    rng_states = gpu_utils.init_rng(nx * ny * nz, seed=numeric_seed)

    # --- Setup Wind Recording (5m, 10m, 15m) ---
    target_heights = [5.0, 10.0, 15.0]
    z_indices = [int(h / dz) for h in target_heights]
    z_indices = [min(max(z, 0), nz-1) for z in z_indices]
    z_indices_dev = cuda.to_device(np.array(z_indices, dtype=np.int32))
    wind_snapshot_dev = cuda.device_array((3, 3, nx, ny), dtype=np.float32)
    
    tpb_2d = (8, 8)
    bpg_2d = ((nx + tpb_2d[0] - 1) // tpb_2d[0], (ny + tpb_2d[1] - 1) // tpb_2d[1])

    total_steps = int(total_time / dt)
    vol = dx * dy * dz
    
    # --- OUTPUT SETUP ---
    csv_writer = None
    # if qf_config:
    #     print(f"Initializing QUIC-Fire Output Mode in {output_dir}")
    #     csv_writer = QuicFireCSVWriter(nx, ny, dx, dy, qf_config.get('origin_x', 0), qf_config.get('origin_y', 0))
    # csv_writer = None
    
    # --- MEMORY MAPPED NPZ STORAGE (FLOAT16) ---
    out_int_fire = qf_config.get('out_int_fire', 100)
    expected_frames = len(range(0, total_steps, out_int_fire))
    
    print(f"Initializing memory-mapped storage for {expected_frames} frames (float16)...")
    print(f"Running for {total_steps} steps (Total Time: {total_time}s)")

    # Temporary paths
    fuel_mm_path = os.path.join(output_dir, "temp_fuel.dat")
    rr_mm_path = os.path.join(output_dir, "temp_rr.dat")
    wind_mm_path = os.path.join(output_dir, "temp_wind.dat")
    
    # Create memmaps
    fuel_mm = np.memmap(fuel_mm_path, dtype='float16', mode='w+', shape=(expected_frames, nx, ny, nz))
    rr_mm = np.memmap(rr_mm_path, dtype='float16', mode='w+', shape=(expected_frames, nx, ny, nz))
    wind_mm = np.memmap(wind_mm_path, dtype='float16', mode='w+', shape=(expected_frames, 3, 3, nx, ny))
    
    frame_idx = 0
    current_wind_speed = default_speed
    current_wind_dir = default_dir
    
    try:
        for t in range(total_steps):
            current_sim_time = t * dt
            
            # --- DYNAMIC IGNITION (ATV) ---
            # If we have ATV lines, check if they are active in this timestep
            if atv_ignition_lines:
                active_ignition_mask = np.zeros((nx, ny, nz), dtype=np.int32)
                found_active = False
                
                for line in atv_ignition_lines:
                    # Check temporal overlap
                    t_start, t_end = line['t_start'], line['t_end']
                    
                    # Overlap logic: simulation interval [current_sim_time, current_sim_time + dt]
                    # line interval [t_start, t_end]
                    start_overlap = max(current_sim_time, t_start)
                    end_overlap = min(current_sim_time + dt, t_end)
                    
                    if start_overlap < end_overlap:
                        found_active = True
                        # Interpolate positions
                        total_duration = t_end - t_start
                        if total_duration <= 0: total_duration = 1e-6
                        
                        # Parametric t (0.0 to 1.0)
                        alpha_start = (start_overlap - t_start) / total_duration
                        alpha_end = (end_overlap - t_start) / total_duration
                        
                        p_start_x = line['x_start'] + (line['x_end'] - line['x_start']) * alpha_start
                        p_start_y = line['y_start'] + (line['y_end'] - line['y_start']) * alpha_start
                        
                        p_end_x = line['x_start'] + (line['x_end'] - line['x_start']) * alpha_end
                        p_end_y = line['y_start'] + (line['y_end'] - line['y_start']) * alpha_end
                        
                        # Rasterize segment
                        # Simple DDA or stepping
                        dist = np.sqrt((p_end_x - p_start_x)**2 + (p_end_y - p_start_y)**2)
                        steps = int(max(dist / min(dx, dy) * 2, 2)) # Supersample to ensure we hit cells
                        
                        for s in range(steps + 1):
                            lerp = s / steps
                            curr_x = p_start_x + (p_end_x - p_start_x) * lerp
                            curr_y = p_start_y + (p_end_y - p_start_y) * lerp
                            
                            ix = int(curr_x / dx)
                            iy = int(curr_y / dy)
                            
                            if 0 <= ix < nx and 0 <= iy < ny:
                                # Determine Z based on terrain + 1 cell
                                terrain_z_idx = int(elevation_host[ix, iy]) # Elevation host is indices
                                
                                # Ignite column just to be safe (Absolute Z 1-25)
                                iz_list = range(1, 26)
                                
                                for iz in iz_list:
                                    if 0 <= iz < nz:
                                        # print(f'igniting {iz}')
                                        active_ignition_mask[ix, iy, iz] = 1

                if found_active:
                    # Upload mask and inject
                    mask_dev = cuda.to_device(active_ignition_mask)
                    inject_ignition_kernel[blocks_per_grid, threads_per_block](
                        n_ep_received_dev, mask_dev, 10000
                    )
                    cuda.synchronize() # Ensure injection finishes before reaction

            # --- DYNAMIC WIND UPDATE ---
            if wind_schedule:
                current_wind_speed, current_wind_dir = interpolate_wind(current_sim_time, wind_schedule)
            else:
                current_wind_speed, current_wind_dir = default_speed, default_dir
                
            wind_rad = np.radians(270 - current_wind_dir)
            
            # --- Physics Pipeline ---
            wind_gpu.apply_drag_kernel[blocks_per_grid, threads_per_block](
                u_dev, fuel_dev, fuel_0_dev, z_coords_dev, current_wind_speed, 10.0, config.K_VON_KARMAN, config.Z0, config.DZ
            )
            wind_gpu.rotate_wind_kernel[blocks_per_grid, threads_per_block](u_dev, v_dev, wind_rad)
            wind_gpu.reset_w_kernel[blocks_per_grid, threads_per_block](w_dev)
            wind_gpu.project_wind_over_terrain_kernel[blocks_per_grid, threads_per_block](
                u_dev, v_dev, w_dev, elevation_dev, dx, dy
            )
            wind_gpu.apply_buoyancy_column_kernel[bpg_2d, tpb_2d](
                w_dev, reaction_rate_dev, dx, dy, dz, config.G, config.RHO_AIR, config.CP_AIR, config.T_AMBIENT, config.H_WOOD
            )
            fire_gpu.compute_reaction_and_fuel_kernel[blocks_per_grid, threads_per_block](
                fuel_dev, fuel_moisture_dev, 
                n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
                centroid_x_dev, centroid_y_dev, centroid_z_dev, ep_history_dev,
                time_since_ignition_dev, reaction_rate_dev, ep_counts_dev,
                dt, config.CM, config.T_BURNOUT, config.H_WOOD, vol, config.C_RAD_LOSS, config.EEP,
                config.CP_WOOD, config.T_CRIT, config.T_AMBIENT
            )
            
            # Zero out EP accumulators for next step
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_x_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_y_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_z_dev)
            cuda.synchronize()
            
            fire_gpu.transport_eps_kernel[blocks_per_grid, threads_per_block](
                ep_counts_dev, 
                n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
                centroid_x_dev, centroid_y_dev, centroid_z_dev,
                u_dev, v_dev, w_dev, elevation_dev, 
                rng_states, dx, dy, dz, dt, config.EEP
            )

            # output progress every n steps
            n_output = 50
            if t % n_output == 0:
                print(f"Step {t+1}/{total_steps} - Time: {current_sim_time:.1f}s - Wind: {current_wind_speed:.1f} m/s @ {current_wind_dir:.1f}°")
                # print avg rr of ignited cells
                rr_host_temp = reaction_rate_dev.copy_to_host()
                ignited_mask = rr_host_temp > 0.001
                if np.any(ignited_mask):
                    avg_rr = np.mean(rr_host_temp[ignited_mask])
                    print(f"  Avg Reaction Rate (ignited cells): {avg_rr:.4f} 1/s")

            # --- OUTPUT ---
            if t % out_int_fire == 0:
                
                # 1. Fuel & RR (Float16 cast on host before write)
                rho_host = fuel_dev.copy_to_host()
                rr_host = reaction_rate_dev.copy_to_host()
                
                if frame_idx < expected_frames:
                    fuel_mm[frame_idx] = rho_host.astype(np.float16)
                    rr_mm[frame_idx] = rr_host.astype(np.float16)
                    
                    # 2. Wind Slices (New, required for advanced viz)
                    wind_gpu.extract_wind_slices_kernel[bpg_2d, tpb_2d](
                        u_dev, v_dev, w_dev, wind_snapshot_dev, z_indices_dev
                    )
                    w_slice_host = wind_snapshot_dev.copy_to_host()
                    wind_mm[frame_idx] = w_slice_host.astype(np.float16)
                    
                    frame_idx += 1
                
                # 3. CSV Output (Legacy/Verification)
                if csv_writer:
                    csv_writer.write_sparse_csv(
                        rho_host, 
                        os.path.join(output_dir, f"fuels_dens_t{int(current_sim_time)}_all_z.csv"),
                        "FuelDensity_kg_m3"
                    )
                    energy_grid = rr_host * config.H_WOOD * vol * (1.0 - config.C_RAD_LOSS) / 1000.0
                    csv_writer.write_sparse_csv(
                        energy_grid, 
                        os.path.join(output_dir, f"fire_energy_t{int(current_sim_time)}_all_z.csv"),
                        "Energy_kW_m2"
                    )

        # --- SAVE COMPLETE NPZ ---
        print("Finalizing NPZ output (Compressing from disk)...")
        npz_filename = os.path.join(output_dir, f"simulation_output.npz")
        
        # Ensure data is written
        fuel_mm.flush()
        rr_mm.flush()
        wind_mm.flush()
        
        # Save compressed (Matches 'Working Reference' schema)
        np.savez_compressed(
            npz_filename,
            fuel=fuel_mm[:frame_idx],
            reaction_rate=rr_mm[:frame_idx],
            wind_local=wind_mm[:frame_idx],
            wind_speed=np.array([current_wind_speed]),
            wind_dir=np.array([current_wind_dir]),
            moisture=np.array([moisture_val]),
            terrain=elevation_host,
            custom_terrain=elevation_host, # For compat
            wind_heights=np.array(target_heights)
        )
        print(f"NPZ Save Complete: {npz_filename}")

    finally:
        # Cleanup
        del u_dev, v_dev, w_dev, fuel_dev, reaction_rate_dev, wind_snapshot_dev
        
        # Close memmaps
        if 'fuel_mm' in locals(): del fuel_mm
        if 'rr_mm' in locals(): del rr_mm
        if 'wind_mm' in locals(): del wind_mm
        
        # Remove temp files
        for f in [fuel_mm_path, rr_mm_path, wind_mm_path]:
            if os.path.exists(f): os.remove(f)