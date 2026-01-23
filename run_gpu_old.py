import numpy as np
import os
from numba import cuda
import config
import wind_gpu
import fire_gpu
import gpu_utils

def run_simulation(params, run_id, output_dir):
    # Unpack parameters
    wind_speed = params.get('wind_speed', 10.0)
    wind_dir_deg = params.get('wind_dir', 0.0)
    moisture = params.get('moisture', 1.0)
    ignition_points = params.get('ignition', [])
    
    # Grid Setup
    nx, ny, nz = config.NX, config.NY, config.NZ
    dx, dy, dz = config.DX, config.DY, config.DZ
    
    # 1. Setup Host Data
    # Fuel: Use params or default zero
    if 'custom_fuel' in params:
        fuel_host = params['custom_fuel'] 
        
        # --- FIX: Ensure contiguous memory after transpose ---
        # 1. Transpose changes shape from (Z, X, Y) to (X, Y, Z)
        # 2. ascontiguousarray forces the memory to match the new shape physically
        fuel_host = np.ascontiguousarray(fuel_host.transpose(1, 2, 0))
        
    else:
        fuel_host = np.zeros((nx, ny, nz), dtype=np.float32)

    # Terrain: Use params or default flat
    if 'custom_terrain' in params:
        # Ensure it is contiguous just in case generation made it weird
        elevation_host = np.ascontiguousarray(params['custom_terrain'])
    else:
        elevation_host = np.zeros((nx, ny), dtype=np.float32)

    z_coords_host = np.arange(nz) * dz
    wind_rad = np.radians(wind_dir_deg)

    # 2. Allocate Device Memory
    # Static
    elevation_dev = cuda.to_device(elevation_host)
    z_coords_dev = cuda.to_device(z_coords_host)
    fuel_0_dev = cuda.to_device(fuel_host)
    
    # Dynamic
    fuel_dev = cuda.to_device(fuel_host)
    u_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    v_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    w_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    reaction_rate_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    time_since_ignition_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    n_ep_received_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    ep_counts_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    
    # Moisture map
    fuel_moisture_dev = cuda.to_device(np.ones((nx, ny, nz), dtype=np.float32) * moisture)

    # Zero init
    gpu_utils.zero_array_3d[(8,8,8), (8,8,8)](reaction_rate_dev)
    gpu_utils.zero_array_3d[(8,8,8), (8,8,8)](time_since_ignition_dev)
    gpu_utils.zero_array_3d[(8,8,8), (8,8,8)](n_ep_received_dev)

    # 3. Ignition
    temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
    for pt in ignition_points:
        # Handle dict or tuple format
        if isinstance(pt, dict):
            ix, iy, iz = pt['x'], pt['y'], pt['z']
        else:
            ix, iy, iz = pt
            
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            temp_ep[ix, iy, iz] = 5000
            
    n_ep_received_dev = cuda.to_device(temp_ep)

    # 4. Kernel Config
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nx + threads_per_block[0] - 1) // threads_per_block[0],
        (ny + threads_per_block[1] - 1) // threads_per_block[1],
        (nz + threads_per_block[2] - 1) // threads_per_block[2]
    )
    
    rng_states = gpu_utils.init_rng(nx * ny * nz, seed=run_id)

    # 5. Storage Pre-allocation (CPU)
    # Strategy: Save generic "channels" for ML training (Fuel, ReactionRate)
    total_steps = int(config.TOTAL_TIME / config.DT)
    save_interval = getattr(config, 'SAVE_INTERVAL', 1) # Default to 1 if not in config
    num_frames = total_steps // save_interval
    
    # Buffers to hold history - using float16 to save space
    history_fuel = np.zeros((num_frames, nx, ny, nz), dtype=np.float16) 
    history_rr = np.zeros((num_frames, nx, ny, nz), dtype=np.float16)

    # 6. Loop
    vol = dx * dy * dz
    frame_idx = 0
    
    for t in range(total_steps):
        # Physics Pipeline
        wind_gpu.apply_drag_kernel[blocks_per_grid, threads_per_block](
            u_dev, fuel_dev, fuel_0_dev, z_coords_dev, wind_speed, 10.0, config.K_VON_KARMAN, config.Z0, config.DZ
        )
        wind_gpu.rotate_wind_kernel[blocks_per_grid, threads_per_block](u_dev, v_dev, wind_rad)
        wind_gpu.reset_w_kernel[blocks_per_grid, threads_per_block](w_dev)
        wind_gpu.project_wind_over_terrain_kernel[blocks_per_grid, threads_per_block](
            u_dev, v_dev, w_dev, elevation_dev, dx, dy
        )
        wind_gpu.apply_buoyancy_kernel[blocks_per_grid, threads_per_block](
            w_dev, reaction_rate_dev, dx, dy, dz, config.G, config.RHO_AIR, config.CP_AIR, config.T_AMBIENT, config.H_WOOD
        )
        fire_gpu.compute_reaction_and_fuel_kernel[blocks_per_grid, threads_per_block](
            fuel_dev, fuel_moisture_dev, n_ep_received_dev, time_since_ignition_dev, reaction_rate_dev, ep_counts_dev,
            config.DT, config.CM, config.T_BURNOUT, config.H_WOOD, vol, config.C_RAD_LOSS, config.EEP
        )
        gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
        cuda.synchronize()
        fire_gpu.transport_eps_kernel[blocks_per_grid, threads_per_block](
            ep_counts_dev, n_ep_received_dev, u_dev, v_dev, w_dev, rng_states, dx, dy, dz
        )

        # Recording
        if t % save_interval == 0 and frame_idx < num_frames:
            # Copy to host and cast to float16 to save space
            history_fuel[frame_idx] = fuel_dev.copy_to_host().astype(np.float16)
            history_rr[frame_idx] = reaction_rate_dev.copy_to_host().astype(np.float16)
            frame_idx += 1

    # 7. Save to Disk
    filename = os.path.join(output_dir, f"run_{run_id}.npz")
    
    np.savez_compressed(
        filename,
        fuel=history_fuel,        # Time-series (Time, X, Y, Z)
        reaction_rate=history_rr, # Time-series (Time, X, Y, Z)
        wind_speed=np.array([wind_speed]),
        wind_dir=np.array([wind_dir_deg]),
        moisture=np.array([moisture]),
        terrain=elevation_host    # Saved as 'terrain', 2D array (X, Y)
    )
    
    # Explicit cleanup
    del u_dev, v_dev, w_dev, fuel_dev, reaction_rate_dev