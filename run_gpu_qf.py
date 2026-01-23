import numpy as np
import os
from numba import cuda
import config_stable as config
import wind_gpu_stable as wind_gpu
import fire_gpu_stable as fire_gpu
import gpu_utils
import scipy.ndimage
from quicfire_io import QuicFireIO, QuicFireCSVWriter

def interpolate_wind(current_time, schedule):
    """Interpolates wind speed and direction from schedule [(t, s, d), ...]."""
    if not schedule:
        return 0.0, 0.0 # Default fallback
        
    # If before start, return first
    if current_time <= schedule[0][0]:
        return schedule[0][1], schedule[0][2]
    # If after end, return last
    if current_time >= schedule[-1][0]:
        return schedule[-1][1], schedule[-1][2]
        
    # Find interval
    for i in range(len(schedule) - 1):
        t1, s1, d1 = schedule[i]
        t2, s2, d2 = schedule[i+1]
        
        if t1 <= current_time <= t2:
            fraction = (current_time - t1) / (t2 - t1)
            s = s1 + (s2 - s1) * fraction
            
            # Direction interpolation (handle 359 -> 1 wrap)
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
    
    # Wind Schedule: [(t, speed, dir), ...]
    wind_schedule = params.get('wind_schedule', [])
    
    # Static fallbacks if no schedule
    default_speed = params.get('wind_speed', 10.0)
    default_dir = params.get('wind_dir', 0.0)
    
    moisture_val = params.get('moisture', 0.1)
    ignition_points = params.get('ignition', [])
    
    # Dimensions
    nx = qf_config.get('nx', config.NX)
    ny = qf_config.get('ny', config.NY)
    nz = qf_config.get('nz', config.NZ)
    dx = qf_config.get('dx', config.DX)
    dy = qf_config.get('dy', config.DY)
    dz = qf_config.get('dz', config.DZ)
    dt = qf_config.get('dt', config.DT)
    total_time = qf_config.get('sim_time', config.TOTAL_TIME)

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
        # Check shape compatibility
        if fuel_host.shape != (nx, ny, nz):
            print(f"Resizing input fuel {fuel_host.shape} to {(nx, ny, nz)}")
            # Simple resize/crop for safety
            temp = np.zeros((nx, ny, nz), dtype=np.float32)
            min_x, min_y, min_z = min(nx, fuel_host.shape[0]), min(ny, fuel_host.shape[1]), min(nz, fuel_host.shape[2])
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
    
    # Moisture Data (Can be custom grid or uniform)
    if 'custom_moisture' in params:
        fuel_moisture_host = params['custom_moisture']
        # Resize if needed
        if fuel_moisture_host.shape != (nx, ny, nz):
             temp = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val
             min_x, min_y, min_z = min(nx, fuel_moisture_host.shape[0]), min(ny, fuel_moisture_host.shape[1]), min(nz, fuel_moisture_host.shape[2])
             temp[:min_x, :min_y, :min_z] = fuel_moisture_host[:min_x, :min_y, :min_z]
             fuel_moisture_host = temp
    else:
        fuel_moisture_host = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val
        
    fuel_moisture_dev = cuda.to_device(fuel_moisture_host)

    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](reaction_rate_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](time_since_ignition_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)

    # Ignition
    temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
    for pt in ignition_points:
        if isinstance(pt, dict): ix, iy, iz = pt['x'], pt['y'], pt['z']
        else: ix, iy, iz = pt
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            temp_ep[ix, iy, iz] = 10000 
            
    n_ep_received_dev = cuda.to_device(temp_ep)

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
    
    # --- CSV WRITER ---
    csv_writer = None
    if qf_config:
        print(f"Initializing QUIC-Fire Output Mode in {output_dir}")
        csv_writer = QuicFireCSVWriter(nx, ny, dx, dy, qf_config.get('origin_x', 0), qf_config.get('origin_y', 0))
    
    print(f"Running for {total_steps} steps (Total Time: {total_time}s)")
    
    for t in range(total_steps):
        current_sim_time = t * dt
        
        # --- DYNAMIC WIND UPDATE ---
        if wind_schedule:
            wind_speed, wind_dir_deg = interpolate_wind(current_sim_time, wind_schedule)
        else:
            wind_speed, wind_dir_deg = default_speed, default_dir
            
        wind_rad = np.radians(270 - wind_dir_deg)
        
        # --- Physics Pipeline ---
        wind_gpu.apply_drag_kernel[blocks_per_grid, threads_per_block](
            u_dev, fuel_dev, fuel_0_dev, z_coords_dev, wind_speed, 10.0, config.K_VON_KARMAN, config.Z0, config.DZ
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

        # --- OUTPUT ---
        if csv_writer:
            # Output Check
            if t % qf_config.get('out_int_fire', 100) == 0:
                rho_host = fuel_dev.copy_to_host()
                rr_host = reaction_rate_dev.copy_to_host()
                
                csv_writer.write_sparse_csv(
                    rho_host, 
                    os.path.join(output_dir, f"fuels_dens_t{int(current_sim_time)}_all_z.csv"),
                    "FuelDensity_kg_m3"
                )
                
                # Energy = ReactionRate * Heat * Vol * (1-RadLoss) / 1000 (to kW)
                energy_grid = rr_host * config.H_WOOD * vol * (1.0 - config.C_RAD_LOSS) / 1000.0
                csv_writer.write_sparse_csv(
                    energy_grid, 
                    os.path.join(output_dir, f"fire_energy_t{int(current_sim_time)}_all_z.csv"),
                    "Energy_kW_m2"
                )

    del u_dev, v_dev, w_dev, fuel_dev, reaction_rate_dev, wind_snapshot_dev