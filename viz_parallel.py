import numpy as np
import matplotlib
# Set backend to Agg before importing pyplot to ensure headless, thread-safe rendering
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
import os
import argparse
from tqdm import tqdm
import scipy.ndimage
import multiprocessing as mp
from functools import partial
import config_new as config

# --- CONFIGURATION ---
DATA_DIR = "./training_data_new_moisture"
OUTPUT_DIR = "./visualizations_new_moisture"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX = config.DX
DY = config.DY
DT = config.DT
DZ = config.DZ

# Physics Constants for Flame Length Calc
H_WOOD = config.H_WOOD
CP_WOOD = getattr(config, 'CP_WOOD', 1700.0)
T_CRIT = getattr(config, 'T_CRIT', 500.0)
T_AMBIENT = getattr(config, 'T_AMBIENT', 300.0)
EFFECTIVE_H = H_WOOD - CP_WOOD * (T_CRIT - T_AMBIENT)

# --- GLOBAL SHARED DATA (For Worker Processes) ---
shared_data = {}

def init_worker(fuel, rr, moisture, terrain, ros_map, arrival_indices):
    """Initialize global data for worker processes."""
    shared_data['fuel'] = fuel
    shared_data['initial_fuel'] = fuel[0] # Store initial state for burn scar calc
    shared_data['rr'] = rr
    shared_data['moisture'] = moisture # Shared Moisture Data
    shared_data['terrain'] = terrain
    shared_data['ros_map'] = ros_map
    shared_data['arrival_indices'] = arrival_indices

def load_data(run_id):
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    with np.load(filepath) as data:
        fuel = data['fuel']
        rr = data['reaction_rate']
        
        if 'wind_local' in data:
            wind_local = data['wind_local'].astype(np.float32)
        else:
            wind_local = None

        if 'custom_terrain' in data:
            terrain = data['custom_terrain']
        elif 'terrain' in data:
            terrain = data['terrain']
        else:
            terrain = np.zeros((fuel.shape[1], fuel.shape[2]))
            
        w_spd = float(data['wind_speed'][0]) if 'wind_speed' in data else 0.0
        w_dir = float(data['wind_dir'][0]) if 'wind_dir' in data else 0.0
        
        # --- EXTRACT MOISTURE ---
        moisture_scalar = 0.0
        moisture_vol = None
        
        if 'moisture' in data:
            m_data = data['moisture']
            if m_data.ndim == 4: # New format
                moisture_vol = m_data
                moisture_scalar = np.mean(m_data[0]) 
            elif m_data.ndim == 1: # Old format
                moisture_scalar = float(m_data[0])
                moisture_vol = np.ones_like(fuel) * moisture_scalar
            else:
                moisture_scalar = float(m_data)
                moisture_vol = np.ones_like(fuel) * moisture_scalar
        else:
             moisture_vol = np.zeros_like(fuel)

        wind_heights = data['wind_heights'] if 'wind_heights' in data else np.array([5.0])
        
    return fuel, rr, moisture_vol, terrain, wind_local, (w_spd, w_dir, moisture_scalar), wind_heights

def calculate_ros_map(rr_vol):
    if rr_vol.ndim == 4:
        fire_2d_history = np.max(rr_vol, axis=3)
    else:
        fire_2d_history = rr_vol

    is_burnt = fire_2d_history > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    arrival_time = arrival_indices * DT
    
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    grads = np.gradient(smoothed_time, DX)
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
    """
    Worker function to render a single frame.
    """
    t = frame_data['t']
    v_data = frame_data['v_data']
    wind_info = frame_data['wind_info']
    v_title = frame_data['v_title']
    
    # Access shared heavy data
    fuel_vol = shared_data['fuel']
    fuel_0 = shared_data['initial_fuel']
    rr_vol = shared_data['rr']
    moisture_vol = shared_data['moisture'] 
    terrain = shared_data['terrain']
    ros_map = shared_data['ros_map']
    arrival_map = shared_data['arrival_indices']

    # --- RENDER LOGIC ---
    f_t = fuel_vol[t]
    r_t = rr_vol[t].astype(np.float32)
    m_t = moisture_vol[t] 
    
    top_fuel = np.max(f_t, axis=2) 
    top_fire = np.max(r_t, axis=2)
    
    # MOISTURE VIZ FIX: 
    # Use min() across Z to see the drying scar regardless of where it happens in the column.
    # Fire sets moisture to 0.0, so the path will appear distinctly white.
    top_moisture = np.min(m_t, axis=2)
    
    side_fuel = np.max(f_t, axis=1)
    side_fire = np.max(r_t, axis=1)
    
    fuel_loss = np.sum(fuel_0, axis=2) - np.sum(f_t, axis=2)
    burn_scar_mask = np.ma.masked_where(fuel_loss < 0.1, fuel_loss)
    
    # Flame Length
    column_rr_sum = np.sum(r_t, axis=2)
    intensity_map = column_rr_sum * EFFECTIVE_H * DZ
    flame_length_map = np.zeros_like(intensity_map)
    active_fire_mask = intensity_map > 1.0 
    if np.any(active_fire_mask):
        flame_length_map[active_fire_mask] = DZ + 0.0155 * np.power(intensity_map[active_fire_mask], 0.4)
    
    w_spd, w_dir, moisture_scalar = wind_info

    fig = plt.figure(figsize=(20, 10), dpi=80)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])      # Top Left: State
    ax_mois = fig.add_subplot(gs[0, 1])  # Top Mid: Moisture
    ax2 = fig.add_subplot(gs[0, 2])      # Top Right: Side View
    ax3 = fig.add_subplot(gs[1, 0])      # Bot Left: ROS
    ax4 = fig.add_subplot(gs[1, 1])      # Bot Mid: Flame Len
    ax5 = fig.add_subplot(gs[1, 2])      # Bot Right: Wind
    
    fig.suptitle(f"Time: {t}s | Wind: {w_spd:.1f} m/s @ {w_dir:.0f}° | Init Moisture: {moisture_scalar*100:.1f}%", fontsize=16)

    # 1. Top-Down State
    ax1.set_title("Fuel & Fire Intensity")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower', interpolation='nearest')
    ax1.set_ylabel("Y Distance")
    ax1.set_xlabel("X Distance")
    
    # 2. Fuel Moisture
    ax_mois.set_title("Min Column Moisture (Drying Track)")
    # Blues: Dark = Wet, White = Dry (0.0)
    vmax_m = max(0.1, moisture_scalar * 1.5)
    im_m = ax_mois.imshow(top_moisture.T, cmap='Blues', vmin=0.0, vmax=vmax_m, origin='lower', interpolation='nearest')
    ax_mois.contour(top_fire.T, levels=[0.1], colors='red', linewidths=1.0, alpha=0.6)
    plt.colorbar(im_m, ax=ax_mois, fraction=0.046, pad=0.04).set_label('Moisture Fraction')
    ax_mois.set_xlabel("X Distance")

    # 3. Side View
    ax2.set_title("Side View (XZ Profile)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto', interpolation='nearest')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto', interpolation='nearest')
    ax2.set_ylabel("Z Height")
    ax2.set_xlabel("X Distance")

    # 4. ROS
    ax3.set_title("Instantaneous Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='Greens', alpha=0.3, origin='lower', interpolation='nearest')
    im3 = ax3.imshow(masked_ros.T, cmap='viridis', vmin=0, vmax=5.0, origin='lower', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label('ROS (m/s)')
    ax3.set_ylabel("Y Distance")
    ax3.set_xlabel("X Distance")

    # 5. Flame Length
    ax4.set_title("Flame Length (m)")
    ax4.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    ax4.imshow(burn_scar_mask.T, cmap='gray', vmin=0, vmax=5.0, alpha=0.4, origin='lower', interpolation='nearest')
    ax4.set_facecolor('black') 
    masked_fl = np.ma.masked_where(flame_length_map < 0.1, flame_length_map)
    im4 = ax4.imshow(masked_fl.T, cmap='inferno', origin='lower', vmin=0, vmax=10.0, interpolation='nearest')
    wind_rad = np.radians(270 - w_dir)
    u_glob = np.cos(wind_rad)
    v_glob = np.sin(wind_rad)
    ax4.quiver(0.92, 0.92, u_glob, v_glob, transform=ax4.transAxes, 
               pivot='middle', scale=10, width=0.02, color='white', zorder=10)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).set_label('Flame Length (m)')
    ax4.set_xlabel("X Distance")
    
    # 6. Vertical Wind Field
    ax5.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    ax5.set_title(v_title)
    masked_w = np.ma.masked_where(np.abs(v_data) < 0.1, v_data)
    ax5.set_facecolor('darkgray')
    im5 = ax5.imshow(masked_w.T, cmap='coolwarm', origin='lower', vmin=-1.0, vmax=5.0, interpolation='nearest')
    ax5.contour(top_fire.T, levels=[0.1], colors='black', linewidths=0.8, alpha=0.5)
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).set_label('W (m/s)')
    ax5.set_xlabel("X Distance")

    plt.tight_layout()
    
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return image_rgba[:, :, :3]

def process_single_run(run_id, args):
    print(f"\n--- Processing Run {run_id} ---")
    print(f"Loading data...")
    data = load_data(run_id)
    if data is None: 
        print(f"Skipping Run {run_id} (Data not found)")
        return
        
    fuel, rr, moisture_vol, terrain, wind_local, wind_info, wind_heights = data
    
    print("Calculating Rate of Spread Map...")
    arrival_indices, ros_map = calculate_ros_map(rr)
    
    w_spd, w_dir, moisture_scalar = wind_info

    # --- DETERMINE Z-MODE ---
    selected_indices = []
    target_h_str = ""
    file_tag = ""
    
    if args.average:
        try:
            req_heights = [float(h.strip()) for h in args.average.split(',')]
            valid_heights = []
            for h in req_heights:
                diffs = np.abs(wind_heights - h)
                if np.min(diffs) < 1.0: 
                    idx = np.argmin(diffs)
                    selected_indices.append(idx)
                    valid_heights.append(wind_heights[idx])
            
            selected_indices = sorted(list(set(selected_indices))) 
            
            if not selected_indices:
                print(f"Error: None of the requested heights {req_heights} matched available data {wind_heights}.")
                return
                
            print(f"Mode: Averaging wind over layers: {valid_heights} meters (Indices: {selected_indices})")
            target_h_str = f"Avg({','.join(map(str, valid_heights))}m)"
            file_tag = "avg_" + "_".join([str(int(h)) for h in valid_heights])
            
        except ValueError:
            print("Error: --average format must be comma-separated numbers")
            return
            
    elif wind_local is not None:
        diffs = np.abs(wind_heights - args.height)
        if np.min(diffs) < 1.0: 
            height_idx = np.argmin(diffs)
            selected_indices = [height_idx]
            print(f"Mode: Visualizing Wind @ {wind_heights[height_idx]}m (Layer Index {height_idx})")
            target_h_str = f"{wind_heights[height_idx]}m"
            file_tag = f"z{int(wind_heights[height_idx])}"
        else:
            print(f"WARNING: Requested height {args.height}m not found. Defaulting to {wind_heights[0]}m.")
            selected_indices = [0]
            target_h_str = f"{wind_heights[0]}m"
            file_tag = f"z{int(wind_heights[0])}"
    else:
        target_h_str = "N/A"
        file_tag = "no_wind"

    # --- PHASE 1: PRE-CALCULATE PHYSICS & HISTORY ---
    print("Pre-calculating vertical wind history...")
    
    max_w_history = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
    
    render_tasks = []
    
    is_history = (args.history == 'on')
    v_title = f"Max Vert Velocity History @ {target_h_str}" if is_history else f"Instantaneous Vertical Wind @ {target_h_str}"

    for t in tqdm(range(0, fuel.shape[0], 2), desc="Physics Calc"):
        w_grid = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)

        if wind_local is not None:
            current_subset = wind_local[t][selected_indices]
            avg_frame = np.mean(current_subset, axis=0)
            w_grid = avg_frame[2] # Z component

            if args.wind_smooth > 0:
                w_grid = scipy.ndimage.uniform_filter(w_grid, size=args.wind_smooth)
            
            update_mask_w = np.abs(w_grid) > np.abs(max_w_history)
            max_w_history[update_mask_w] = w_grid[update_mask_w]

        if is_history:
            v_data_frame = max_w_history.copy()
        else:
            v_data_frame = w_grid.copy()

        task_data = {
            't': t,
            'v_data': v_data_frame,
            'wind_info': wind_info,
            'v_title': v_title,
        }
        render_tasks.append(task_data)

    print(f"Rendering {len(render_tasks)} frames on {args.workers} CPUs...")
    
    frames = []
    # Pass moisture volume to init_worker
    with mp.Pool(processes=args.workers, 
                 initializer=init_worker, 
                 initargs=(fuel, rr, moisture_vol, terrain, ros_map, arrival_indices)) as pool:
        
        for frame in tqdm(pool.imap(render_worker, render_tasks), total=len(render_tasks), desc="Rendering"):
            frames.append(frame)

    suffix_str = args.suffix if args.suffix else ""
    output_filename = f"run_{run_id}_viz_FL_{file_tag}_{suffix_str}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
        
    print(f"Saving video to {output_path}...")
    iio.imwrite(output_path, np.stack(frames), fps=10)
    print(f"Run {run_id} Complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help="ID of the run(s). Can be single '999', list '1,2,3', or range '1-5'")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the output file")
    parser.add_argument("--wind_smooth", type=int, default=0, help="Kernel size for spatial smoothing (box average) of wind")
    parser.add_argument("--history", type=str, choices=['on', 'off'], default='on', help="Enable persistent history for wind plots")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of render processes")
    parser.add_argument("--height", type=float, default=5.0, help="Height level to visualize (typically 5, 10, 15)")
    parser.add_argument("--average", type=str, default=None, help="Comma-separated list of heights to average")

    args = parser.parse_args()

    run_ids = []
    try:
        if '-' in args.run_id:
            start, end = map(int, args.run_id.split('-'))
            run_ids = list(range(start, end + 1))
        elif ',' in args.run_id:
            run_ids = [int(x) for x in args.run_id.split(',')]
        else:
            run_ids = [int(args.run_id)]
    except ValueError:
        print("Error: Invalid run ID format. Use '1', '1,2,3', or '1-5'.")
        return

    print(f"Batch processing {len(run_ids)} runs: {run_ids}")

    for rid in run_ids:
        try:
            process_single_run(rid, args)
        except Exception as e:
            print(f"CRITICAL ERROR processing Run {rid}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()