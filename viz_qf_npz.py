import numpy as np
import matplotlib
# Set backend to Agg for headless rendering
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import imageio.v3 as iio
import os
import argparse
import scipy.ndimage
import multiprocessing as mp
from tqdm import tqdm

# --- CONFIGURATION ---
DX_DEFAULT = 2.0
DY_DEFAULT = 2.0
DZ_DEFAULT = 1.0
DT_DEFAULT = 1.0 

# Physics for Flame Length
H_WOOD = 18.62e6
CP_WOOD = 1700.0
T_CRIT = 500.0
T_AMBIENT = 300.0
EFFECTIVE_H = H_WOOD - CP_WOOD * (T_CRIT - T_AMBIENT)

# --- SHARED DATA ---
shared_data = {}

def init_worker(fuel, rr, terrain, ros_map, arrival_indices):
    """Initialize global data for worker processes."""
    shared_data['fuel'] = fuel
    shared_data['initial_fuel'] = fuel[0] 
    shared_data['rr'] = rr
    shared_data['terrain'] = terrain
    shared_data['ros_map'] = ros_map
    shared_data['arrival_indices'] = arrival_indices

def init_3d_worker(terrain):
    """Initialize data just for the 3D terrain worker."""
    shared_data['terrain'] = terrain

def load_data(path):
    """
    Loads QF-generated data. Supports:
    1. Single .npz file (Legacy)
    2. Single .h5 file (HDF5)
    3. Directory containing split .npz files (Parallel Output)
    """
    fuel, rr, terrain, wind_local = None, None, None, None
    w_spd, w_dir, moisture = 0.0, 0.0, 0.1
    wind_heights = np.array([5.0])
    
    # --- CASE 1: HDF5 File ---
    if path.endswith('.h5'):
        print(f"Loading HDF5: {path}")
        import h5py
        with h5py.File(path, 'r') as f:
            fuel = f['fuel'][:]
            rr = f['reaction_rate'][:]
            if 'wind_local' in f:
                wind_local = f['wind_local'][:]
            
            if 'custom_terrain' in f:
                terrain = f['custom_terrain'][:]
            elif 'terrain' in f:
                terrain = f['terrain'][:]
                
            w_spd = f.attrs.get('wind_speed', 0.0)
            w_dir = f.attrs.get('wind_dir', 0.0)
            moisture = f.attrs.get('moisture', 0.1)
            
            if 'wind_heights' in f:
                wind_heights = f['wind_heights'][:]

        return fuel, rr, terrain, wind_local, (w_spd, w_dir, moisture), wind_heights

    # --- CASE 2: Split NPZ (Directory or File in Directory) ---
    search_dir = path if os.path.isdir(path) else os.path.dirname(path)
    meta_path = os.path.join(search_dir, "outputs_meta.npz")
    
    if os.path.exists(meta_path):
        print(f"Loading Split NPZ from: {search_dir}")
        
        # Load heavy arrays
        fuel = np.load(os.path.join(search_dir, "outputs_fuel.npz"))['fuel']
        rr = np.load(os.path.join(search_dir, "outputs_rr.npz"))['reaction_rate']
        
        wind_path = os.path.join(search_dir, "outputs_wind.npz")
        if os.path.exists(wind_path):
            wind_local = np.load(wind_path)['wind_local']
            
        # Load Metadata
        with np.load(meta_path) as meta:
            w_spd = float(meta['wind_speed'][0])
            w_dir = float(meta['wind_dir'][0])
            moisture = float(meta['moisture'][0])
            
            if 'custom_terrain' in meta:
                terrain = meta['custom_terrain']
            elif 'terrain' in meta:
                terrain = meta['terrain']
                
            if 'wind_heights' in meta:
                wind_heights = meta['wind_heights']
                
        return fuel, rr, terrain, wind_local, (w_spd, w_dir, moisture), wind_heights

    # --- CASE 3: Single NPZ File (Legacy) ---
    if path.endswith('.npz') and os.path.exists(path):
        print(f"Loading Single NPZ: {path}")
        with np.load(path) as data:
            fuel = data['fuel']
            rr = data['reaction_rate']
            
            if 'wind_local' in data:
                wind_local = data['wind_local']
                
            if 'custom_terrain' in data:
                terrain = data['custom_terrain']
            elif 'terrain' in data:
                terrain = data['terrain']
                
            w_spd = float(data['wind_speed'][0]) if 'wind_speed' in data else 0.0
            w_dir = float(data['wind_dir'][0]) if 'wind_dir' in data else 0.0
            moisture = float(data['moisture'][0]) if 'moisture' in data else 0.1
            
            wind_heights = data['wind_heights'] if 'wind_heights' in data else np.array([5.0])
            
        return fuel, rr, terrain, wind_local, (w_spd, w_dir, moisture), wind_heights

    print(f"Error: Could not determine format or find files for {path}")
    return None

def calculate_ros_map(rr_vol):
    """Calculates ROS from Reaction Rate volume history."""
    if rr_vol.ndim == 4:
        # Max over Z axis to get 2D fire history
        fire_2d_history = np.max(rr_vol, axis=3)
    else:
        fire_2d_history = rr_vol

    is_burnt = fire_2d_history > 0.001
    
    # First index where fire appears
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)
    
    never_burnt_mask = (arrival_indices == 0) & (~is_burnt[0])
    arrival_time = arrival_indices * DT_DEFAULT 
    
    # Smooth arrival time for cleaner gradients
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    
    grads = np.gradient(smoothed_time, DX_DEFAULT)
    dt_dy, dt_dx = grads
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    with np.errstate(divide='ignore'):
        ros_map = 1.0 / slowness
    
    ros_map[ros_map > 30.0] = 30.0 
    ros_map = scipy.ndimage.median_filter(ros_map, size=3)
    ros_map[never_burnt_mask] = 0
    
    return arrival_indices, ros_map

def render_worker(task_data):
    """
    Worker function to render a single SIMULATION frame.
    """
    t = task_data['t']
    v_data = task_data['v_data']
    wind_info = task_data['wind_info']
    v_title = task_data['v_title']

    # Access shared data
    fuel_vol = shared_data['fuel']
    fuel_0 = shared_data['initial_fuel']
    rr_vol = shared_data['rr']
    terrain = shared_data['terrain']
    ros_map = shared_data['ros_map']
    arrival_map = shared_data['arrival_indices']

    # Bounds check
    if t >= fuel_vol.shape[0]: return None

    # Cast to float32 for plotting (incoming might be float16)
    f_t = fuel_vol[t].astype(np.float32)
    r_t = rr_vol[t].astype(np.float32)
    
    # Projections
    top_fuel = np.max(f_t, axis=2) 
    top_fire = np.sum(r_t, axis=2) 
    side_fuel = np.max(f_t, axis=1)
    side_fire = np.max(r_t, axis=1)
    
    # Burn Scar
    fuel_loss = np.sum(fuel_0, axis=2) - np.sum(f_t, axis=2)
    burn_scar_mask = np.ma.masked_where(fuel_loss < 0.1, fuel_loss)
    
    # Flame Length
    column_rr_sum = np.sum(r_t, axis=2) 
    intensity_map = column_rr_sum * EFFECTIVE_H * DZ_DEFAULT 
    
    flame_length_map = np.zeros_like(intensity_map)
    active_fire_mask = intensity_map > 1.0 
    
    if np.any(active_fire_mask):
        flame_length_map[active_fire_mask] = DZ_DEFAULT + 0.0155 * np.power(intensity_map[active_fire_mask], 0.4)
    
    w_spd, w_dir, moisture = wind_info

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 10), dpi=80)
    # Layout: Top row (2 cols), Bottom row (3 cols)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])      # Top Left
    ax2 = fig.add_subplot(gs[0, 1:])     # Top Right (Spans 2 columns - Restored)
    ax3 = fig.add_subplot(gs[1, 0])      # Bottom Left
    ax4 = fig.add_subplot(gs[1, 1])      # Bottom Middle
    ax5 = fig.add_subplot(gs[1, 2])      # Bottom Right
    
    fig.suptitle(f"Sim Frame: {t} | Wind: {w_spd:.1f} m/s @ {w_dir:.0f}° | Moisture: {moisture*100:.1f}%", fontsize=16)

    # 1. Top-Down Fuel & Fire
    ax1.set_title("Top-Down Fuel & Fire")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=0.2, alpha=0.7, origin='lower', interpolation='nearest') 
    ax1.set_xlabel("X Distance")
    ax1.set_ylabel("Y Distance")
    
    # 2. Side View
    ax2.set_title("Side View (XZ Profile)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto', interpolation='nearest')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=0.2, alpha=0.9, origin='lower', aspect='auto', interpolation='nearest')
    ax2.set_xlabel("X Distance")
    ax2.set_ylabel("Z Height")
    
    # 3. ROS
    ax3.set_title("Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='Greens', alpha=0.3, origin='lower', interpolation='nearest')
    im3 = ax3.imshow(masked_ros.T, cmap='viridis', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    plt.colorbar(im3, ax=ax3).set_label('ROS (m/s)')
    ax3.set_xlabel("X Distance")

    # 4. Flame Length & Wind Vector
    ax4.set_title("Flame Length (m)")
    ax4.imshow(top_fuel.T, cmap='Greens', alpha=0.3, origin='lower', interpolation='nearest')
    ax4.imshow(burn_scar_mask.T, cmap='gray', alpha=0.4, origin='lower', interpolation='nearest')
    ax4.set_facecolor('black')
    
    masked_fl = np.ma.masked_where(flame_length_map < 0.1, flame_length_map)
    im4 = ax4.imshow(masked_fl.T, cmap='inferno', vmin=0, vmax=10.0, origin='lower', interpolation='nearest')
    plt.colorbar(im4, ax=ax4).set_label('Flame Length (m)')
    
    # Add Wind Arrow
    wind_rad = np.radians(270 - w_dir)
    u_glob = np.cos(wind_rad)
    v_glob = np.sin(wind_rad)
    ax4.quiver(0.92, 0.92, u_glob, v_glob, transform=ax4.transAxes, 
               pivot='middle', scale=10, width=0.02, color='white', zorder=10)
    ax4.set_xlabel("X Distance")

    # 5. Vertical Wind Field
    ax5.set_title(v_title)
    ax5.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    ax5.set_facecolor('darkgray')
    
    masked_w = np.ma.masked_where(np.abs(v_data) < 0.1, v_data)
    im5 = ax5.imshow(masked_w.T, cmap='coolwarm', origin='lower', vmin=-2.0, vmax=5.0, interpolation='nearest')
    
    ax5.contour(top_fire.T, levels=[0.05], colors='black', linewidths=0.8, alpha=0.5)
    plt.colorbar(im5, ax=ax5).set_label('W (m/s)')
    ax5.set_xlabel("X Distance")

    plt.tight_layout()
    
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return image_rgba[:, :, :3]

def render_3d_frame(angle):
    """
    Worker function to render a single 3D TERRAIN frame.
    """
    terrain = shared_data['terrain']
    
    fig = plt.figure(figsize=(12, 12), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    
    nx, ny = terrain.shape
    x = np.arange(nx) * DX_DEFAULT
    y = np.arange(ny) * DY_DEFAULT
    X, Y = np.meshgrid(x, y) # Meshgrid for plotting, note shape needs to match transpose usually
    
    # Note: X, Y shape (NY, NX) via meshgrid default, Terrain is (NX, NY)
    # Transpose terrain to match meshgrid (X corresponds to cols, Y to rows)
    surf = ax.plot_surface(X, Y, terrain.T, cmap='terrain', 
                           linewidth=0, antialiased=False, shade=True)
    
    # Set camera angle
    ax.view_init(elev=35, azim=angle)
    
    ax.set_title("3D Terrain Orbital View")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Elevation (m)")
    
    # Ensure aspect ratio is reasonable
    # Matplotlib 3D aspect ratio is tricky, usually set_box_aspect helps
    try:
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(terrain)*5)) # Exaggerate Z slightly
    except:
        pass # Older matplotlib might not support this

    fig.tight_layout()
    
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return image_rgba[:, :, :3]

def main():
    parser = argparse.ArgumentParser(description="Visualize QUIC-Fire Outputs")
    parser.add_argument("input_path", type=str, help="Path to data")
    parser.add_argument("--output", type=str, default="viz_output.mp4", help="Output sim video path")
    parser.add_argument("--output_3d", type=str, default="terrain_orbit.mp4", help="Output 3D terrain video path")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--orbit", action='store_true', help="Generate 3D terrain orbital video in addition to sim viz")
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--history", type=str, choices=['on', 'off'], default='on')
    
    args = parser.parse_args()
    
    # 1. Load Data
    data = load_data(args.input_path)
    if data is None: return
    fuel, rr, terrain, wind_local, wind_info, wind_heights = data
    
    print(f"Data Loaded. Shape: {fuel.shape}")
    
    # --- PHASE 1: STANDARD SIM VIZ ---
    print("--- Phase 1: Simulation Visualization ---")
    
    # Pre-calc metrics
    arrival_indices, ros_map = calculate_ros_map(rr)
    
    # Pre-calc wind
    diffs = np.abs(wind_heights - args.height)
    h_idx = np.argmin(diffs)
    actual_h = wind_heights[h_idx]
    
    max_w_history = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
    is_history = (args.history == 'on')
    v_title = f"Max Vert Velocity {'History' if is_history else ''} @ {actual_h}m"
    
    render_tasks = []
    num_frames = fuel.shape[0]
    
    print("Preparing physics frames...")
    for t in range(num_frames):
        w_grid = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
        if wind_local is not None:
            w_grid = wind_local[t, h_idx, 2, :, :].astype(np.float32)
            update_mask = np.abs(w_grid) > np.abs(max_w_history)
            max_w_history[update_mask] = w_grid[update_mask]
            
        v_data_frame = max_w_history.copy() if is_history else w_grid.copy()
        
        task = {
            't': t,
            'v_data': v_data_frame,
            'wind_info': wind_info,
            'v_title': v_title
        }
        render_tasks.append(task)
    
    frames = []
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(fuel, rr, terrain, ros_map, arrival_indices)) as pool:
        for frame in tqdm(pool.imap(render_worker, render_tasks), total=num_frames, desc="Rendering Sim"):
            if frame is not None:
                frames.append(frame)
            
    if frames:
        print(f"Saving Sim Video to {args.output}...")
        iio.imwrite(args.output, np.stack(frames), fps=10)
    
    # --- PHASE 2: 3D ORBIT VIZ (Optional) ---
    if args.orbit:
        print("\n--- Phase 2: 3D Terrain Orbit ---")
        angles = range(0, 360, 2) # 2 degree steps = 180 frames
        
        orbit_frames = []
        # Re-use pool or create new one. Create new one to be clean with initializer
        with mp.Pool(processes=args.workers, initializer=init_3d_worker, initargs=(terrain,)) as pool:
            for frame in tqdm(pool.imap(render_3d_frame, angles), total=len(angles), desc="Rendering Orbit"):
                if frame is not None:
                    orbit_frames.append(frame)
        
        if orbit_frames:
            print(f"Saving 3D Orbit Video to {args.output_3d}...")
            iio.imwrite(args.output_3d, np.stack(orbit_frames), fps=30)
            print("Orbit Done.")

if __name__ == "__main__":
    main()