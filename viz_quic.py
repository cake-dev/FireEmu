import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
import os
import argparse
import pandas as pd
from tqdm import tqdm
import glob
import re
import scipy.ndimage
import multiprocessing as mp

# --- DEFAULTS ---
DX_DEFAULT = 2.0
DY_DEFAULT = 2.0
DZ_DEFAULT = 1.0
DT_DEFAULT = 1.0 # Assumed time step between frames if not inferable

# Physics for Flame Length (Matches viz_stable.py)
H_WOOD = 18.62e6
CP_WOOD = 1700.0
T_CRIT = 500.0
T_AMBIENT = 300.0
EFFECTIVE_H = H_WOOD - CP_WOOD * (T_CRIT - T_AMBIENT)

# --- SHARED DATA FOR WORKERS ---
shared_data = {}

def init_worker(dm_data, ros_map, arrival_indices):
    """Initialize global data for worker processes."""
    shared_data['dm'] = dm_data
    shared_data['ros_map'] = ros_map
    shared_data['arrival_indices'] = arrival_indices

class DataManager:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.grid_info = None
        self.initial_fuel = None
        
        # Discover available files
        self.fuel_files = self._map_files("fuels_dens_t*_all_z.csv")
        self.energy_files = self._map_files("fire_energy_t*_all_z.csv")
        
        if not self.fuel_files:
            raise FileNotFoundError(f"No 'fuels_dens' CSV files found in {input_dir}")

        # Auto-detect grid dimensions
        self._detect_grid()
        
        # Load initial fuel for burn scar calculation
        self._load_initial_fuel()

    def _map_files(self, pattern):
        files = glob.glob(os.path.join(self.input_dir, pattern))
        file_map = {}
        regex = re.compile(r'_t(\d+)_')
        for f in files:
            match = regex.search(f)
            if match:
                ts = int(match.group(1))
                file_map[ts] = f
        return file_map

    def _detect_grid(self):
        first_ts = min(self.fuel_files.keys())
        df = pd.read_csv(self.fuel_files[first_ts])
        
        if 'IndexX' not in df.columns:
            # Fallback for old CSVs (not recommended given memory error, but safe for small grids)
            # Assuming small grid if IndexX missing, otherwise it would crash earlier
            print("Warning: IndexX column missing. Attempting legacy reconstruction (may fail on large grids).")
            # Logic omitted for brevity, assuming new format
            raise ValueError("CSV missing IndexX. Please re-run simulation.")

        nx = df['IndexX'].max() + 1
        ny = df['IndexY'].max() + 1
        nz = df['ZLevel'].max() + 1
        
        self.grid_info = {'nx': int(nx), 'ny': int(ny), 'nz': int(nz)}
        print(f"Detected Grid: {nx}x{ny}x{nz}")

    def reconstruct_grid(self, csv_path):
        nx, ny, nz = self.grid_info['nx'], self.grid_info['ny'], self.grid_info['nz']
        dense_grid = np.zeros((nx, ny, nz), dtype=np.float32)
        
        if not os.path.exists(csv_path):
            return dense_grid
            
        df = pd.read_csv(csv_path)
        ix = df['IndexX'].values.astype(int)
        iy = df['IndexY'].values.astype(int)
        iz = df['ZLevel'].values.astype(int)
        values = df.iloc[:, -1].values 
        
        valid = (ix < nx) & (iy < ny) & (iz < nz)
        dense_grid[ix[valid], iy[valid], iz[valid]] = values[valid]
        return dense_grid

    def _load_initial_fuel(self):
        ts = min(self.fuel_files.keys())
        print(f"Loading initial fuel state from t={ts}...")
        self.initial_fuel = self.reconstruct_grid(self.fuel_files[ts])

def calculate_ros_map(dm):
    """
    Pre-calculates Rate of Spread (ROS) map from the full fire history.
    This requires iterating through all energy files once.
    """
    print("Calculating Rate of Spread Map (scanning all energy files)...")
    nx, ny = dm.grid_info['nx'], dm.grid_info['ny']
    
    # Track arrival time (first time energy > threshold)
    arrival_time = np.full((nx, ny), -1.0, dtype=np.float32)
    
    sorted_times = sorted(dm.energy_files.keys())
    
    for t in tqdm(sorted_times, desc="Building Arrival Map"):
        # Load just 2D max for speed? No, CSV structure requires full read usually.
        # Optimization: Just read CSV and update arrival where valid
        path = dm.energy_files[t]
        if not os.path.exists(path): continue
        
        df = pd.read_csv(path)
        # Filter for active fire
        active = df[df.iloc[:, -1] > 0.1]
        
        if active.empty: continue
        
        ix = active['IndexX'].values.astype(int)
        iy = active['IndexY'].values.astype(int)
        
        # Flatten indices for current active cells
        flat_idx = ix * ny + iy
        
        # Update arrival time if it's -1 (not yet arrived)
        # We need to map back to 2D
        # This is a bit slow in python loop, let's use numpy masking
        
        # Current mask of active cells
        current_mask = np.zeros((nx, ny), dtype=bool)
        current_mask[ix, iy] = True
        
        # Update: where arrival_time is -1 AND currently active
        update_mask = (arrival_time == -1) & current_mask
        arrival_time[update_mask] = float(t)

    # Now calculate gradients
    # Mask unburnt areas
    is_burnt = arrival_time >= 0
    arrival_time[~is_burnt] = np.nan # Use NaN for gradients
    
    # Smooth arrival time
    smoothed_time = scipy.ndimage.gaussian_filter(arrival_time, sigma=1.5)
    
    # Gradients
    grads = np.gradient(smoothed_time, DX_DEFAULT)
    dt_dx, dt_dy = grads
    slowness = np.sqrt(dt_dx**2 + dt_dy**2)
    
    # Invert for ROS
    ros_map = np.zeros_like(slowness)
    with np.errstate(divide='ignore', invalid='ignore'):
        ros_map = 1.0 / slowness
    
    ros_map[~is_burnt] = 0
    ros_map[np.isnan(ros_map)] = 0
    ros_map[ros_map > 30.0] = 30.0 # Cap
    
    # Smooth ROS
    ros_map = scipy.ndimage.median_filter(ros_map, size=3)
    
    return arrival_time, ros_map

def render_worker(timestep):
    """
    Worker function to render a single frame.
    Uses shared_data to access the large grid info without copying.
    """
    dm = shared_data['dm']
    ros_map = shared_data['ros_map']
    arrival_indices = shared_data['arrival_indices']
    
    # Reconstruct grids for this timestep
    # Note: DataManager inside worker needs to be picklable or re-initialized. 
    # Simpler: We pass the filename and grid info. 
    # But for now, let's rely on copy-on-write fork (Linux) or standard pickling.
    
    fuel_grid = dm.reconstruct_grid(dm.fuel_files.get(timestep, ""))
    energy_grid = dm.reconstruct_grid(dm.energy_files.get(timestep, ""))
    initial_fuel = dm.initial_fuel
    
    nx, ny, nz = fuel_grid.shape
    
    # --- PROJECTIONS ---
    # Top Down
    top_fuel = np.max(fuel_grid, axis=2) # Max usually looks better for canopy
    top_fire = np.sum(energy_grid, axis=2) # Sum energy column
    
    # Side View (Sum over Y to see profile)
    side_fuel = np.max(fuel_grid, axis=1)
    side_fire = np.sum(energy_grid, axis=1)
    
    # Burn Scar
    fuel_loss = np.sum(initial_fuel, axis=2) - np.sum(fuel_grid, axis=2)
    burn_scar_mask = np.ma.masked_where(fuel_loss < 0.1, fuel_loss)
    
    # --- FLAME LENGTH ---
    flame_length_map = np.zeros_like(top_fire)
    active_fire_mask = top_fire > 1.0
    if np.any(active_fire_mask):
        # Convert flux to W/m2 approximately (assuming output is kW)
        intensity = top_fire * 1000.0 
        flame_length_map[active_fire_mask] = DZ_DEFAULT + 0.0155 * np.power(intensity[active_fire_mask], 0.4)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(20, 10), dpi=80)
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    fig.suptitle(f"QUIC-Fire Simulation | Time: {timestep}s", fontsize=16)
    
    # 1. Top Down
    ax1.set_title("Top-Down State")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=50.0, alpha=0.7, origin='lower', interpolation='nearest')
    ax1.set_ylabel("Y Index")
    
    # 2. Side View
    ax2.set_title("Side View (XZ Profile)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto', interpolation='nearest')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=50.0, alpha=0.9, origin='lower', aspect='auto', interpolation='nearest')
    ax2.set_ylabel("Z Height")
    
    # 3. ROS
    ax3.set_title("Rate of Spread (m/s)")
    # Mask ROS where fire hasn't arrived yet
    current_ros = ros_map.copy()
    not_yet_burnt = (arrival_indices > timestep) | (arrival_indices < 0)
    current_ros[not_yet_burnt] = 0
    masked_ros = np.ma.masked_where(current_ros == 0, current_ros)
    
    ax3.imshow(top_fuel.T, cmap='Greens', alpha=0.3, origin='lower')
    im3 = ax3.imshow(masked_ros.T, cmap='viridis', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    plt.colorbar(im3, ax=ax3).set_label("ROS (m/s)")
    
    # 4. Flame Length
    ax4.set_title("Flame Length (m)")
    ax4.imshow(top_fuel.T, cmap='Greens', alpha=0.3, origin='lower')
    # Overlay Scar
    ax4.imshow(burn_scar_mask.T, cmap='gray', alpha=0.4, origin='lower')
    
    masked_fl = np.ma.masked_where(flame_length_map < 0.1, flame_length_map)
    im4 = ax4.imshow(masked_fl.T, cmap='inferno', vmin=0, vmax=10.0, origin='lower', interpolation='nearest')
    plt.colorbar(im4, ax=ax4).set_label("Flame Length (m)")
    
    plt.tight_layout()
    
    # Render to buffer
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return image_rgba[:, :, :3] # RGB

def main():
    parser = argparse.ArgumentParser(description="Visualize QUIC-Fire CSV Outputs (Advanced)")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing .csv outputs")
    parser.add_argument("--output-dir", type=str, default="./viz_output", help="Folder to save video")
    parser.add_argument("--timesteps", type=str, default="all", help="e.g. '0-1000' or 'all'")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Scanning {args.input_dir}...")
    try:
        dm = DataManager(args.input_dir)
    except Exception as e:
        print(f"Error initializing data: {e}")
        return

    # Parse Timesteps
    available_ts = sorted(list(dm.fuel_files.keys()))
    requested_ts = []
    if args.timesteps == "all":
        requested_ts = available_ts
    elif '-' in args.timesteps:
        s, e = map(int, args.timesteps.split('-'))
        requested_ts = [t for t in available_ts if s <= t <= e]
    else:
        # Single or list
        try:
            reqs = [int(x) for x in args.timesteps.split(',')]
            requested_ts = [t for t in reqs if t in available_ts]
        except:
            print("Invalid timesteps format")
            return

    if not requested_ts:
        print("No valid timesteps found.")
        return

    # Calculate ROS map once
    arrival_indices, ros_map = calculate_ros_map(dm)

    print(f"Rendering {len(requested_ts)} frames using {args.workers} workers...")
    
    frames = []
    # Use pool to render
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(dm, ros_map, arrival_indices)) as pool:
        # Use imap to maintain order
        for frame in tqdm(pool.imap(render_worker, requested_ts), total=len(requested_ts)):
            frames.append(frame)
            
    # Save Video
    vid_path = os.path.join(args.output_dir, "simulation_advanced.mp4")
    print(f"Saving video to {vid_path}...")
    iio.imwrite(vid_path, np.stack(frames), fps=10)
    print("Done.")

if __name__ == "__main__":
    main()