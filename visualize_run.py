import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v3 as iio
import os
import argparse
from tqdm import tqdm
import scipy.ndimage

# --- CONFIGURATION ---
DATA_DIR = "./training_data_new_wind_2"
OUTPUT_DIR = "./visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DX = 2.0
DY = 2.0
DT = 1.0

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
        
    return fuel, rr, terrain, wind_local, (w_spd, w_dir)

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
    
    ros_map[never_burnt_mask] = 0
    ros_map[ros_map > 30.0] = 0 
    
    return arrival_indices, ros_map

def render_frame(t, fuel_vol, rr_vol, terrain, ros_map, arrival_map, wind_tuple, wind_info, 
                 h_wind_data, v_wind_data, h_title, v_title):
    f_t = fuel_vol[t]
    r_t = rr_vol[t]
    
    top_fuel = np.max(f_t, axis=2) 
    top_fire = np.max(r_t, axis=2)
    side_fuel = np.max(f_t, axis=1)
    side_fire = np.max(r_t, axis=1)
    
    # Unpack smoothed instantaneous winds (not strictly used for history plots, but available)
    u_grid, v_grid, w_grid = wind_tuple

    w_spd, w_dir = wind_info

    # --- LAYOUT SETUP ---
    # 2 Rows, 3 Columns
    # Top Row: [Top-Down (1 col)] [Side View (2 cols)]
    # Bottom Row: [ROS (1 col)] [Horiz Wind (1 col)] [Vert Wind (1 col)]
    
    fig = plt.figure(figsize=(20, 10), dpi=80)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])      # Top Left
    ax2 = fig.add_subplot(gs[0, 1:])     # Top Right (Spans 2 cols)
    ax3 = fig.add_subplot(gs[1, 0])      # Bottom Left
    ax4 = fig.add_subplot(gs[1, 1])      # Bottom Middle
    ax5 = fig.add_subplot(gs[1, 2])      # Bottom Right
    
    fig.suptitle(f"Time: {t}s | Global Wind: {w_spd:.1f} m/s @ {w_dir:.0f}°", fontsize=16)

    # 1. Top-Down State
    ax1.set_title("Top-Down State")
    ax1.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', interpolation='nearest')
    ax1.contour(terrain.T, levels=8, colors='white', alpha=0.3, linewidths=0.5)
    ax1.imshow(top_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.7, origin='lower', interpolation='nearest')
    ax1.set_ylabel("Y Distance")
    ax1.set_xlabel("X Distance")

    # 2. Side View (Spans 2 columns now)
    ax2.set_title("Side View (XZ Profile)")
    ax2.imshow(side_fuel.T, cmap='Greens', vmin=0, vmax=2.0, origin='lower', aspect='auto', interpolation='nearest')
    ax2.imshow(side_fire.T, cmap='hot', vmin=0.0, vmax=1.0, alpha=0.9, origin='lower', aspect='auto', interpolation='nearest')
    ax2.set_ylabel("Z Height")
    ax2.set_xlabel("X Distance")

    # 3. ROS (Bottom Left)
    ax3.set_title("Instantaneous Rate of Spread (m/s)")
    mask = arrival_map > t 
    masked_ros = np.ma.masked_where(mask | (ros_map == 0), ros_map)
    ax3.imshow(terrain.T, cmap='gray', alpha=0.3, origin='lower', interpolation='nearest')
    im3 = ax3.imshow(masked_ros.T, cmap='jet', vmin=0, vmax=5.0, origin='lower', interpolation='nearest')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label('ROS (m/s)')
    ax3.set_ylabel("Y Distance")
    ax3.set_xlabel("X Distance")

    # 4. Horizontal Wind Field (Bottom Middle)
    ax4.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    ax4.set_title(h_title)
    
    # Mask values close to global wind (ambient)
    # Using a slightly smaller epsilon (0.05) to capture subtle gusts
    masked_wind = np.ma.masked_where(h_wind_data < (w_spd + 0.05), h_wind_data)
    ax4.set_facecolor('darkgray') 
    
    # Plot Magnitude Heatmap
    im4 = ax4.imshow(masked_wind.T, cmap='viridis', origin='lower', vmin=0, vmax=15.0, interpolation='nearest')
    
    # Global Wind Vector Overlay
    u_glob = np.cos(np.radians(w_dir + 180))
    v_glob = np.sin(np.radians(w_dir + 180))
    
    ax4.quiver(0.92, 0.92, u_glob, v_glob, transform=ax4.transAxes, 
               pivot='middle', scale=10, width=0.02, color='black', zorder=9)
    ax4.quiver(0.92, 0.92, u_glob, v_glob, transform=ax4.transAxes, 
               pivot='middle', scale=10, width=0.012, color='white', zorder=10)
    
    ax4.text(0.92, 0.82, "Global", transform=ax4.transAxes, 
             ha='center', va='top', fontsize=8, color='black', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Overlay Fire Contour
    ax4.contour(top_fire.T, levels=[0.1], colors='black', linewidths=0.8, alpha=0.5)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).set_label('Speed (m/s)')
    ax4.set_xlabel("X Distance")
    
    # 5. Vertical Wind Field (Bottom Right)
    ax5.imshow(top_fuel.T, cmap='Greens', vmin=0, vmax=2.0, alpha=0.3, origin='lower', interpolation='nearest')
    ax5.set_title(v_title)
    
    # Mask values near 0
    masked_w = np.ma.masked_where(np.abs(v_wind_data) < 0.1, v_wind_data)
    ax5.set_facecolor('darkgray')
    
    # Plot Vertical W
    im5 = ax5.imshow(masked_w.T, cmap='coolwarm', origin='lower', vmin=-1.0, vmax=5.0, interpolation='nearest')
    
    # Overlay Fire Contour
    ax5.contour(top_fire.T, levels=[0.1], colors='black', linewidths=0.8, alpha=0.5)

    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).set_label('W (m/s)')
    ax5.set_xlabel("X Distance")

    plt.tight_layout()
    
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image_rgba[:, :, :3]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int, help="ID of the run to visualize (e.g. 999)")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix for the output file")
    parser.add_argument("--wind_smooth", type=int, default=0, help="Kernel size for spatial smoothing (box average) of wind")
    parser.add_argument("--history", type=str, choices=['on', 'off'], default='on', 
                        help="Enable persistent history for wind plots (on/off). Default: on")
    args = parser.parse_args()

    print(f"Loading run_{args.run_id}...")
    data = load_data(args.run_id)
    if data is None: return
    fuel, rr, terrain, wind_local, wind_info = data
    
    print("Calculating Rate of Spread Map...")
    _, ros_map = calculate_ros_map(rr)
    
    if rr.ndim == 4:
        fire_2d = np.max(rr, axis=3)
    else:
        fire_2d = rr
    is_burnt = fire_2d > 0.01
    arrival_indices = np.argmax(is_burnt, axis=0).astype(float)

    mode_str = "History" if args.history == 'on' else "Instant"
    print(f"Rendering frames (Mode: {mode_str}, Smooth: {args.wind_smooth})...")
    
    frames = []
    
    w_spd = wind_info[0]

    # Initialize History buffers
    # Initialize horizontal history with the global wind speed to prevent startup masking artifacts
    max_h_mag_history = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
    max_w_history = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)

    # FPS = 10 for smaller file size and smoother playback
    for t in tqdm(range(0, fuel.shape[0], 2)):
        
        # Default zero-grids if wind data missing
        u_grid = np.zeros((fuel.shape[1], fuel.shape[2]), dtype=np.float32)
        v_grid = np.zeros_like(u_grid)
        w_grid = np.zeros_like(u_grid)
        
        current_mag = np.zeros_like(u_grid)

        # Update Wind Data
        if wind_local is not None:
            # Extract instantaneous raw wind
            u_grid = wind_local[t, 0, 0]
            v_grid = wind_local[t, 0, 1]
            w_grid = wind_local[t, 0, 2]

            # --- SPATIAL SMOOTHING ---
            if args.wind_smooth > 0:
                u_grid = scipy.ndimage.uniform_filter(u_grid, size=args.wind_smooth)
                v_grid = scipy.ndimage.uniform_filter(v_grid, size=args.wind_smooth)
                w_grid = scipy.ndimage.uniform_filter(w_grid, size=args.wind_smooth)

            # --- CALCULATE MAGNITUDE ---
            current_mag = np.sqrt(u_grid**2 + v_grid**2)
            
            # --- UPDATE HISTORY ---
            # Horizontal: Accumulate maximum magnitude strictly
            # We initialize mask where new data exceeds history to ensure clean updates
            np.maximum(max_h_mag_history, current_mag, out=max_h_mag_history)

            # Vertical: Track Max Deviation (Up/Down)
            update_mask_w = np.abs(w_grid) > np.abs(max_w_history)
            max_w_history[update_mask_w] = w_grid[update_mask_w]
        
        # --- SELECT DATA FOR RENDERING ---
        if args.history == 'on':
            h_data = max_h_mag_history
            v_data = max_w_history
            h_title = "Max Wind Speed History (Most Extreme) @ 5m"
            v_title = "Max Vertical Velocity History @ 5m"
        else:
            h_data = current_mag
            v_data = w_grid
            h_title = "Instantaneous Wind Speed @ 5m"
            v_title = "Instantaneous Vertical Wind @ 5m"

        frames.append(render_frame(t, fuel, rr, terrain, ros_map, arrival_indices, 
                                   (u_grid, v_grid, w_grid), wind_info, 
                                   h_data, v_data, h_title, v_title))

    if args.suffix:
        output_path = os.path.join(OUTPUT_DIR, f"run_{args.run_id}_viz_{args.history}_{args.suffix}.mp4")
    else:
        output_path = os.path.join(OUTPUT_DIR, f"run_{args.run_id}_viz_{args.history}.mp4")
        
    print(f"Saving video to {output_path}...")
    iio.imwrite(output_path, np.stack(frames), fps=10)
    print("Done!")

if __name__ == "__main__":
    main()