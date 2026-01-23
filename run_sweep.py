import numpy as np
import os
from numba import cuda
import config
import wind_gpu
import fire_gpu
import gpu_utils
import scipy.ndimage

def run_simulation(params, run_id, output_dir):
    # --- 1. Unpack Parameters & Apply Physics Overrides ---
    # Environmental defaults
    wind_speed = params.get('wind_speed', 10.0)
    wind_dir_deg = params.get('wind_dir', 0.0)
    moisture = params.get('moisture', 1.0)
    ignition_points = params.get('ignition', [])
    run_name = params.get('run_name', 'unknown')

    # Physics Overrides (Critical for Parameter Sweeps)
    # We prioritize values passed in 'params', fallback to 'config.py'
    dt = params.get('dt', config.DT)
    dx = params.get('dx', config.DX)
    dy = params.get('dy', config.DY)
    dz = params.get('dz', config.DZ)
    slope_factor = params.get('slope_factor', config.SLOPE_FACTOR)
    jump_hack = params.get('jump_hack', config.JUMP_HACK)
    mod_dt = params.get('mod_dt', config.MOD_DT)
    eep = params.get('eep', config.EEP)
    
    # Grid dimensions (Assumed fixed for this pipeline, but physics scale changes with DX/DY)
    nx, ny, nz = config.NX, config.NY, config.NZ
    
    # Recalculate derived constants based on specific run physics
    vol = dx * dy * dz
    
    # Calculate steps to maintain constant physical duration
    # We assume config.TOTAL_TIME is the target duration in seconds
    total_steps = int(config.TOTAL_TIME / dt)
    
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nx + threads_per_block[0] - 1) // threads_per_block[0],
        (ny + threads_per_block[1] - 1) // threads_per_block[1],
        (nz + threads_per_block[2] - 1) // threads_per_block[2]
    )

    # --- 2. Setup Host Data ---
    if 'custom_fuel' in params:
        fuel_host = params['custom_fuel'] 
        fuel_host = np.ascontiguousarray(fuel_host.transpose(1, 2, 0))
    else:
        fuel_host = np.zeros((nx, ny, nz), dtype=np.float32)

    if 'custom_terrain' in params:
        elevation_host = np.ascontiguousarray(params['custom_terrain'])
    else:
        elevation_host = np.zeros((nx, ny), dtype=np.float32)

    # --- CRITICAL FIX FOR BLOCKY TERRAIN ---
    # We create a "Physics Elevation" map. 
    # If the world is blocky (Minecraft style), the gradient is zero on top of blocks 
    # and infinite at the edges. This kills the "uphill acceleration" effect.
    # We apply a Gaussian blur to create a smooth "ramp" representing the average slope.
    # sigma=1.5 smoothes out sharp 1-block steps into a continuous slope.
    elevation_physics = scipy.ndimage.gaussian_filter(elevation_host, sigma=1.5)

    # Use local DZ for coordinates
    z_coords_host = np.arange(nz) * dz
    wind_rad = np.radians(270 - wind_dir_deg)

    # --- 3. Allocate Device Memory ---
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
    fuel_moisture_dev = cuda.to_device(np.ones((nx, ny, nz), dtype=np.float32) * moisture)

    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](reaction_rate_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](time_since_ignition_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)

    # Ignition
    temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
    for pt in ignition_points:
        if isinstance(pt, dict): ix, iy, iz = pt['x'], pt['y'], pt['z']
        else: ix, iy, iz = pt
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            temp_ep[ix, iy, iz] = 5000
    n_ep_received_dev = cuda.to_device(temp_ep)

    rng_states = gpu_utils.init_rng(nx * ny * nz, seed=run_id)

    # --- Setup Wind Recording (5m, 10m, 15m) ---
    target_heights = [5.0, 10.0, 15.0]
    # Use local DZ to find indices
    z_indices = [int(h / dz) for h in target_heights]
    z_indices = [min(max(z, 0), nz-1) for z in z_indices]
    
    z_indices_dev = cuda.to_device(np.array(z_indices, dtype=np.int32))
    wind_snapshot_dev = cuda.device_array((3, 3, nx, ny), dtype=np.float32)
    
    # 2D Kernel Configuration
    tpb_2d = (8, 8)
    bpg_2d = (
        (nx + tpb_2d[0] - 1) // tpb_2d[0],
        (ny + tpb_2d[1] - 1) // tpb_2d[1]
    )

    # Save Interval Logic
    save_interval = getattr(config, 'SAVE_INTERVAL', 1)
    # Adjust save_interval if DT changes significantly? 
    # For now, we keep it based on steps, but users might want to adjust this in params too.
    
    num_frames = total_steps // save_interval
    
    history_fuel = np.zeros((num_frames, nx, ny, nz), dtype=np.float16) 
    history_rr = np.zeros((num_frames, nx, ny, nz), dtype=np.float16)
    history_wind = np.zeros((num_frames, 3, 3, nx, ny), dtype=np.float16)
    
    frame_idx = 0
    
    for t in range(total_steps):
        # --- Physics Pipeline ---
        
        # 1. Drag & Terrain (3D Kernels)
        # Using local dz
        wind_gpu.apply_drag_kernel[blocks_per_grid, threads_per_block](
            u_dev, fuel_dev, fuel_0_dev, z_coords_dev, wind_speed, 10.0, config.K_VON_KARMAN, config.Z0, dz
        )
        wind_gpu.rotate_wind_kernel[blocks_per_grid, threads_per_block](u_dev, v_dev, wind_rad)
        wind_gpu.reset_w_kernel[blocks_per_grid, threads_per_block](w_dev)
        
        # Using local dx, dy
        wind_gpu.project_wind_over_terrain_kernel[blocks_per_grid, threads_per_block](
            u_dev, v_dev, w_dev, elevation_dev, dx, dy
        )
        
        # 2. Buoyancy
        # Using local dx, dy, dz
        wind_gpu.apply_buoyancy_column_kernel[bpg_2d, tpb_2d](
            w_dev, reaction_rate_dev, dx, dy, dz, config.G, config.RHO_AIR, config.CP_AIR, config.T_AMBIENT, config.H_WOOD
        )
        
        # 3. Fire Logic
        # Using local dt, eep, vol
        fire_gpu.compute_reaction_and_fuel_kernel[blocks_per_grid, threads_per_block](
            fuel_dev, fuel_moisture_dev, 
            n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
            centroid_x_dev, centroid_y_dev, centroid_z_dev, ep_history_dev,
            time_since_ignition_dev, reaction_rate_dev, ep_counts_dev,
            dt, config.CM, config.T_BURNOUT, config.H_WOOD, vol, config.C_RAD_LOSS, eep,
            config.CP_WOOD, config.T_CRIT, config.T_AMBIENT
        )
        
        gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
        gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_x_dev)
        gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_y_dev)
        gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_z_dev)
        cuda.synchronize()
        
        # 4. Transport
        # Using local dx, dy, dz, dt
        fire_gpu.transport_eps_kernel[blocks_per_grid, threads_per_block](
            ep_counts_dev, 
            n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
            centroid_x_dev, centroid_y_dev, centroid_z_dev,
            u_dev, v_dev, w_dev, elevation_dev, 
            rng_states, dx, dy, dz, dt, slope_factor, jump_hack, mod_dt
        )

        if t % save_interval == 0 and frame_idx < num_frames:
            history_fuel[frame_idx] = fuel_dev.copy_to_host().astype(np.float16)
            history_rr[frame_idx] = reaction_rate_dev.copy_to_host().astype(np.float16)
            
            wind_gpu.extract_wind_slices_kernel[bpg_2d, tpb_2d](
                u_dev, v_dev, w_dev, wind_snapshot_dev, z_indices_dev
            )
            history_wind[frame_idx] = wind_snapshot_dev.copy_to_host().astype(np.float16)
            
            frame_idx += 1

    # --- 4. Save Output with Metadata ---
    filename = os.path.join(output_dir, f"run_{run_id}.npz")
    np.savez_compressed(
        filename,
        fuel=history_fuel,        
        reaction_rate=history_rr,
        wind_local=history_wind,
        wind_speed=np.array([wind_speed]),
        wind_dir=np.array([wind_dir_deg]),
        moisture=np.array([moisture]),
        terrain=elevation_host,
        wind_heights=np.array(target_heights),
        # SAVE METADATA FOR VIZ
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        run_name=run_name,
        slope_factor=slope_factor,
        jump_hack=params.get('jump_hack', config.JUMP_HACK),
        mod_dt=params.get('mod_dt', config.MOD_DT),
    )
    
    del u_dev, v_dev, w_dev, fuel_dev, reaction_rate_dev, wind_snapshot_dev