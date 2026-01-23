import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
import os
import argparse
from tqdm import tqdm
import scipy.ndimage
import multiprocessing as mp
import config

# --- CONFIGURATION ---
DATA_DIR = "./training_data_sweep_7"
OUTPUT_DIR = "./visualizations_sweep_7"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global shared dictionary for workers
shared_data = {}

def init_worker(fuel, rr, terrain, ros_map, arrival_indices, sim_meta):
    """Initialize global data for worker processes."""
    shared_data['fuel'] = fuel
    shared_data['initial_fuel'] = fuel[0]
    shared_data['rr'] = rr
    shared_data['terrain'] = terrain
    shared_data['ros_map'] = ros_map
    shared_data['arrival_indices'] = arrival_indices
    shared_data['meta'] = sim_meta # Store dt/dx here

def load_data(run_id):
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with np.load(filepath) as data:
        fuel = data['fuel']
        rr = data['reaction_rate']
        
        # Handle Terrain
        if 'custom_terrain' in data:
            terrain = data['custom_terrain']
        elif 'terrain' in data:
            terrain = data['terrain']
        else:
            terrain = np.zeros((fuel.shape[1], fuel.shape[2]))
            
        # Handle Wind
        if 'wind_local' in data:
            wind_local = data['wind_local'].astype(np.float32)
        else:
            wind_local = None

        w_spd = float(data['wind_speed'][0]) if 'wind_speed' in data else 0.0
        w_dir = float(data['wind_dir'][0]) if 'wind_dir' in data else 0.0
        wind_heights = data['wind_heights'] if 'wind_heights' in data else np.array([5.0])

        # --- EXTRACT PHYSICS METADATA ---
        # Crucial for rendering the correct time steps and ROS
        sim_meta = {
            'dt': float(data['dt']) if 'dt' in data else config.DT,
            'dx': float(data['dx']) if 'dx' in data else config.DX,
            'dy': float(data['dy']) if 'dy' in data else config.DY,
            'run_name': str(data['run_name']) if 'run_name' in data else "unknown",
            'slope_factor': float(data['slope_factor']) if 'slope_factor' in data else config.SLOPE_FACTOR,
            'jump_hack': bool(data['jump_hack']) if 'jump_hack' in data else config.JUMP_HACK,
            'mod_dt': bool(data['mod_dt']) if 'mod_dt' in data else config.MOD_DT,
        }

    return fuel, rr, terrain, wind_local, (w_spd, w_dir), wind_heights, sim_meta

def calculate_ros_map(rr_vol, sim_meta):
    """Calculates ROS using the specific DT/DX of this run."""
    dt = sim_meta['dt']
    dx = sim_meta['dx']
    
    if rr_vol.ndim == 4:
        fire_2d_history = np.max(rr_vol, axis=3)
    else:
        fire_2d_history = rr_vol

    is_burnt = fire_2d_history > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    
    # Use specific DT
    arrival_time = arrival_indices * dt
    
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    
    # Use specific DX
    grads = np.gradient(smoothed_time, dx)
    dt_dy, dt_dx = grads
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    with np.errstate(divide='ignore'):
        ros_map = 1.0 / slowness
    
    ros_map[ros_map > 30.0] = 30.0 
    ros_map = scipy.ndimage.median_filter(ros_map, size=3)
    ros_map = scipy.ndimage.gaussian_filter(ros_map, sigma=1.0)
    ros_map[never_burnt_mask] = 0
    
    return arrival_indices, ros_map

def render_worker(frame_data):
    t = frame_data['t']
    wind_info = frame_data['wind_info']
    
    # Pull data
    fuel_vol = shared_data['fuel']
    rr_vol = shared_data['rr']
    terrain = shared_data['terrain']
    ros_map = shared_data['ros_map']
    arrival_map = shared_data['arrival_indices']
    sim_meta = shared_data['meta'] # Access metadata

    # Setup Frame
    f_t = fuel_vol[t]
    r_t = rr_vol[t]
    w_spd, w_dir = wind_info

    fig = plt.figure(figsize=(12, 6), dpi=80)
    gs = gridspec.GridSpec(1, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate real time for title
    real_time = t * sim_meta['dt']
    fig.suptitle(f"Run: {sim_meta['run_name']} | T={real_time:.1f}s | DT={sim_meta['dt']} DX={sim_meta['dx']} | Slope={sim_meta['slope_factor']} | jump_hack={sim_meta['jump_hack']} | mod_dt={sim_meta['mod_dt']}", fontsize=14)

    # 1. Fire view
    ax1.set_title("Fuel & Fire")
    ax1.imshow(np.max(f_t, axis=2).T, cmap='Greens', vmin=0, vmax=2.0, origin='lower')
    ax1.imshow(np.max(r_t, axis=2).T, cmap='hot', alpha=0.6, origin='lower')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    

    # 2. ROS view
    ax2.set_title("Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    im = ax2.imshow(masked_ros.T, cmap='viridis', vmin=0, vmax=5.0, origin='lower')
    ax2.imshow(terrain.T, cmap='Greens', alpha=0.3, origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image_rgba[:, :, :3]

def process_single_run(run_id, args):
    data = load_data(run_id)
    if data is None: return
    
    fuel, rr, terrain, wind_local, wind_info, wind_heights, sim_meta = data
    
    print(f"Processing {run_id} [{sim_meta['run_name']}] (DT={sim_meta['dt']}, DX={sim_meta['dx']})")

    arrival_indices, ros_map = calculate_ros_map(rr, sim_meta)

    # Determine render frames based on DT (avoid rendering empty frames if DT is small)
    # E.g. render every 1.0 simulation second
    render_step = int(1.0 / sim_meta['dt']) 
    render_step = max(1, render_step)
    
    render_tasks = []
    for t in range(0, fuel.shape[0], render_step):
        render_tasks.append({
            't': t,
            'wind_info': wind_info,
        })

    frames = []
    with mp.Pool(processes=args.workers, 
                 initializer=init_worker, 
                 initargs=(fuel, rr, terrain, ros_map, arrival_indices, sim_meta)) as pool:
        
        for frame in tqdm(pool.imap(render_worker, render_tasks), total=len(render_tasks)):
            frames.append(frame)

    output_path = os.path.join(OUTPUT_DIR, f"run_{run_id}_{sim_meta['run_name']}.mp4")
    iio.imwrite(output_path, np.stack(frames), fps=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Handle list of IDs "1,2,3" or range "1-5"
    if '-' in args.run_id:
        s, e = map(int, args.run_id.split('-'))
        rids = list(range(s, e+1))
    else:
        rids = [int(x) for x in args.run_id.split(',')]

    for rid in rids:
        process_single_run(rid, args)

if __name__ == "__main__":
    main()