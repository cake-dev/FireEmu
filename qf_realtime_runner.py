"""
qf_realtime_runner.py - Generator-based QUIC-Fire Runner for Streamlit

This module is adapted from run_gpu_qf.py to yield state frame-by-frame
instead of running to completion and saving files.
"""

import numpy as np
from numba import cuda
import config_qf as config
import wind_gpu_qf as wind_gpu
import fire_gpu_qf as fire_gpu
import gpu_utils
import scipy.ndimage

@cuda.jit
def inject_ignition_kernel(n_ep_received, ignition_mask, ignition_strength):
    x, y, z = cuda.grid(3)
    nx, ny, nz = n_ep_received.shape
    if x < nx and y < ny and z < nz:
        if ignition_mask[x, y, z] > 0:
            cuda.atomic.add(n_ep_received, (x, y, z), ignition_strength)

@cuda.jit
def dry_ignition_cells_kernel(fuel_moisture, ignition_mask):
    x, y, z = cuda.grid(3)
    nx, ny, nz = fuel_moisture.shape
    if x < nx and y < ny and z < nz:
        if ignition_mask[x, y, z] > 0:
            fuel_moisture[x, y, z] = 0.0

def interpolate_wind(current_time, schedule):
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
            diff = d2 - d1
            if diff > 180: diff -= 360
            if diff < -180: diff += 360
            d = d1 + diff * fraction
            if d < 0: d += 360
            if d >= 360: d -= 360
            return s, d
    return schedule[0][1], schedule[0][2]

def simulation_iterator(params, yield_interval=1):
    """
    Generator that yields simulation state.
    
    Parameters:
    -----------
    yield_interval : int
        Only yield state (and copy from GPU) every N steps.
    """
    # Unpack Configuration
    qf_config = params.get('qf_config', {})
    wind_schedule = params.get('wind_schedule', [])
    default_speed = params.get('wind_speed', 10.0)
    default_dir = params.get('wind_dir', 0.0)
    moisture_val = params.get('moisture', 0.1)
    ignition_data = params.get('ignition', [])
    
    # Dimensions
    nx = qf_config.get('nx', config.NX)
    ny = qf_config.get('ny', config.NY)
    nz = qf_config.get('nz', config.NZ)
    dx = qf_config.get('dx', config.DX)
    dy = qf_config.get('dy', config.DY)
    dz = qf_config.get('dz', config.DZ)
    dt = qf_config.get('dt', config.DT)
    total_time = qf_config.get('sim_time', config.TOTAL_TIME)
    canopy_height = qf_config.get('canopy_height', config.CANOPY_HEIGHT)

    # CUDA Config
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nx + 7) // 8,
        (ny + 7) // 8,
        (nz + 7) // 8
    )
    tpb_2d = (8, 8)
    bpg_2d = ((nx + 7) // 8, (ny + 7) // 8)

    # --- Initialize Host Data ---
    if 'custom_fuel' in params:
        fuel_host = params['custom_fuel'].astype(np.float32)
        # Handle resizing if needed
        if fuel_host.shape != (nx, ny, nz):
            temp = np.zeros((nx, ny, nz), dtype=np.float32)
            mx, my, mz = map(min, zip(fuel_host.shape, (nx,ny,nz)))
            temp[:mx, :my, :mz] = fuel_host[:mx, :my, :mz]
            fuel_host = temp
    else:
        fuel_host = np.zeros((nx, ny, nz), dtype=np.float32)
        fuel_host[:, :, 0:3] = 0.5

    if 'custom_terrain' in params:
        elevation_host = np.ascontiguousarray(params['custom_terrain'], dtype=np.float32)
    else:
        elevation_host = np.zeros((nx, ny), dtype=np.float32)

    # Normalize Terrain
    elev_min = np.min(elevation_host)
    if elev_min != 0:
        elevation_host -= elev_min
    elevation_meters = elevation_host * dz
    elevation_physics = scipy.ndimage.gaussian_filter(elevation_meters, sigma=1.0).astype(np.float32)
    z_coords_host = (np.arange(nz) * dz).astype(np.float32)

    if 'custom_moisture' in params:
        fuel_moisture_host = params['custom_moisture'].astype(np.float32)
    else:
        fuel_moisture_host = np.ones((nx, ny, nz), dtype=np.float32) * moisture_val

    # --- Allocate GPU ---
    elevation_dev = cuda.to_device(elevation_physics)
    z_coords_dev = cuda.to_device(z_coords_host)
    fuel_0_dev = cuda.to_device(fuel_host)
    fuel_dev = cuda.to_device(fuel_host)
    fuel_moisture_dev = cuda.to_device(fuel_moisture_host)
    
    u_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    v_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    w_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    reaction_rate_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    time_since_ignition_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    
    centroid_x_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    centroid_y_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    centroid_z_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    ep_history_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    
    incoming_x_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    incoming_y_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    incoming_z_dev = cuda.device_array((nx, ny, nz), dtype=np.float32)
    n_ep_received_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)
    ep_counts_dev = cuda.device_array((nx, ny, nz), dtype=np.int32)

    # --- Initialize GPU ---
    gpu_utils.init_centroid_kernel[blocks_per_grid, threads_per_block](centroid_x_dev, centroid_y_dev, centroid_z_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](ep_history_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](reaction_rate_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](time_since_ignition_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_x_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_y_dev)
    gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_z_dev)

    # --- Handle Ignition ---
    atv_ignition_lines = []
    legacy_ignition_points = []
    
    if isinstance(ignition_data, dict) and ignition_data.get('type') == 5:
        atv_ignition_lines = ignition_data['lines']
    elif isinstance(ignition_data, list):
        legacy_ignition_points = ignition_data
        
    if legacy_ignition_points:
        temp_ep = np.zeros((nx, ny, nz), dtype=np.int32)
        temp_moist_mask = np.zeros((nx, ny, nz), dtype=np.int32)
        for pt in legacy_ignition_points:
            if isinstance(pt, dict):
                ix, iy, iz = int(pt['x']), int(pt['y']), int(pt['z'])
            else:
                ix, iy, iz = int(pt[0]), int(pt[1]), int(pt[2])
            
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                temp_ep[ix, iy, iz] = 10000
                temp_moist_mask[ix, iy, iz] = 1
        
        n_ep_received_dev.copy_to_device(temp_ep)
        mask_dev = cuda.to_device(temp_moist_mask)
        dry_ignition_cells_kernel[blocks_per_grid, threads_per_block](fuel_moisture_dev, mask_dev)
        cuda.synchronize()

    rng_states = gpu_utils.init_rng(nx * ny * nz, seed=12345)
    total_steps = int(total_time / dt)

    # --- Main Loop (Generator) ---
    try:
        for t in range(total_steps):
            current_sim_time = t * dt
            
            # 1. Dynamic Ignition (ATV)
            if atv_ignition_lines:
                active_mask = np.zeros((nx, ny, nz), dtype=np.int32)
                found = False
                for line in atv_ignition_lines:
                    t_start, t_end = line['t_start'], line['t_end']
                    if current_sim_time >= t_start and current_sim_time < t_end:
                        found = True
                        frac = (current_sim_time - t_start) / max(t_end - t_start, 1e-6)
                        cx = line['x_start'] + (line['x_end'] - line['x_start']) * frac
                        cy = line['y_start'] + (line['y_end'] - line['y_start']) * frac
                        ix, iy = int(cx/dx), int(cy/dy)
                        if 0 <= ix < nx and 0 <= iy < ny:
                             active_mask[ix, iy, 0] = 1
                             if nz > 1: active_mask[ix, iy, 1] = 1
                
                if found:
                    mask_dev = cuda.to_device(active_mask)
                    inject_ignition_kernel[blocks_per_grid, threads_per_block](n_ep_received_dev, mask_dev, 10000)
                    dry_ignition_cells_kernel[blocks_per_grid, threads_per_block](fuel_moisture_dev, mask_dev)

            # 2. Wind Update
            if wind_schedule:
                ws, wd = interpolate_wind(current_sim_time, wind_schedule)
            else:
                ws, wd = default_speed, default_dir
            wind_rad = np.radians(270 - wd)

            # 3. Physics Pipeline
            wind_gpu.apply_drag_kernel_research[blocks_per_grid, threads_per_block](
                u_dev, fuel_dev, fuel_0_dev, z_coords_dev, canopy_height, ws, 10.0, config.K_VON_KARMAN, config.Z0, dz
            )
            wind_gpu.rotate_wind_kernel[blocks_per_grid, threads_per_block](u_dev, v_dev, wind_rad)
            wind_gpu.reset_w_kernel[blocks_per_grid, threads_per_block](w_dev)
            wind_gpu.project_wind_over_terrain_kernel[blocks_per_grid, threads_per_block](u_dev, v_dev, w_dev, elevation_dev, dx, dy, dz)
            wind_gpu.apply_buoyancy_column_kernel_research[bpg_2d, tpb_2d](w_dev, reaction_rate_dev, u_dev, v_dev, dx, dy, dz, config.G, config.RHO_AIR, config.CP_AIR, config.T_AMBIENT, config.H_WOOD)
            
            fire_gpu.compute_reaction_and_fuel_kernel[blocks_per_grid, threads_per_block](
                fuel_dev, fuel_moisture_dev, n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
                centroid_x_dev, centroid_y_dev, centroid_z_dev, ep_history_dev, time_since_ignition_dev, reaction_rate_dev,
                ep_counts_dev, dt, config.CM, config.T_BURNOUT, config.H_WOOD, dx*dy*dz, config.C_RAD_LOSS, config.EEP,
                config.CP_WOOD, config.T_CRIT, config.T_AMBIENT, config.H_H2O_EFF
            )
            
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](n_ep_received_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_x_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_y_dev)
            gpu_utils.zero_array_3d[blocks_per_grid, threads_per_block](incoming_z_dev)
            cuda.synchronize()
            
            fire_gpu.transport_eps_kernel_v2[blocks_per_grid, threads_per_block](
                ep_counts_dev, n_ep_received_dev, incoming_x_dev, incoming_y_dev, incoming_z_dev,
                centroid_x_dev, centroid_y_dev, centroid_z_dev, u_dev, v_dev, w_dev, elevation_dev,
                rng_states, dx, dy, dz, dt, config.EEP, wind_rad
            )

            # Yield state for visualization
            # Optimization: Only copy to host if this frame is being rendered
            if t % yield_interval == 0:
                yield {
                    "step": t,
                    "time": current_sim_time,
                    "wind_speed": ws,
                    "wind_dir": wd,
                    "fuel": fuel_dev.copy_to_host(),
                    "reaction_rate": reaction_rate_dev.copy_to_host()
                }

    finally:
        del u_dev, v_dev, w_dev, fuel_dev, fuel_0_dev, reaction_rate_dev