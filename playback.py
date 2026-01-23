import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def load_data(run_dir, run_id):
    """Load the simulation data from the .npz file."""
    filename = f"run_{run_id}.npz"
    filepath = os.path.join(run_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        exit(1)
        
    print(f"Loading data from {filepath}...")
    data = np.load(filepath)
    return data

def create_animation(run_dir, run_id, save_video=False, fps=10):
    # 1. Load Data
    data = load_data(run_dir, run_id)
    
    # Extract arrays
    # shape: (num_frames, nx, ny, nz)
    fuel_history = data['fuel']
    rr_history = data['reaction_rate']
    terrain = data['terrain']
    
    num_frames, nx, ny, nz = fuel_history.shape
    
    # 2. Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Title info
    w_speed = data['wind_speed'][0]
    w_dir = data['wind_dir'][0]
    fig.suptitle(f"Run {run_id}: Wind {w_speed}m/s @ {w_dir}°", fontsize=14)
    
    # 3. Pre-calculate Terrain Surface
    # Create meshgrid for terrain (X, Y)
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij') # Important: indexing='ij' matches array layout
    
    # 4. Helper to get active fire coordinates
    def get_fire_scatter(frame_idx):
        """Returns X, Y, Z coordinates and Colors for active fire cells."""
        rr = rr_history[frame_idx]
        
        # Threshold for "visible fire"
        active_mask = rr > 0.001
        
        if not np.any(active_mask):
            return None, None, None
            
        # Get indices of active cells
        ix, iy, iz = np.where(active_mask)
        
        # Physical Z = Terrain Height + Z_index (approx)
        # Note: In the sim, z-index is relative to ground, but let's just plot valid indices
        # If we want exact visual height: z_plot = terrain[ix, iy] + iz
        z_plot = terrain[ix, iy] + iz
        
        # Color based on intensity (reaction rate)
        intensities = rr[active_mask]
        # Normalize for colormap (clip high values for better visual dynamic range)
        norm_intensities = np.clip(intensities / 0.1, 0, 1) 
        
        return X[ix, iy], Y[ix, iy], z_plot, norm_intensities

    # 5. Helper to get surface color (Burnt vs Unburnt)
    # We color the terrain surface based on the fuel remaining in the bottom layer (z=0)
    initial_surface_fuel = fuel_history[0, :, :, 0]
    # Avoid divide by zero
    initial_surface_fuel[initial_surface_fuel == 0] = 1.0 

    # Initial Plot Setup
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_zlim(0, np.max(terrain) + nz + 5)
    
    # View angle
    ax.view_init(elev=30, azim=-45)

    # Plot the terrain surface initially
    # We use plot_surface for the ground. We will update its facecolors.
    surf = ax.plot_surface(X, Y, terrain, cmap='gist_earth', shade=True, alpha=0.6)
    
    # Scatter plot for flames (initially empty)
    flame_scatter = ax.scatter([], [], [], c=[], cmap='autumn', s=10, alpha=0.8, vmin=0, vmax=1)

    # Text for frame counter
    time_text = ax.text2D(0.05, 0.95, "T=0", transform=ax.transAxes)

    print("Generating animation frames...")

    def update(frame):
        # A. Update Flames
        fx, fy, fz, f_int = get_fire_scatter(frame)
        
        if fx is not None:
            flame_scatter._offsets3d = (fx, fy, fz)
            flame_scatter.set_array(f_int)
        else:
            # Hide scatter if no fire
            flame_scatter._offsets3d = ([], [], [])

        # B. Update Terrain Colors (Burn Scar)
        # Calculate fraction burnt: 1.0 = Unburnt, 0.0 = Burnt
        current_fuel = fuel_history[frame, :, :, 0]
        fraction = current_fuel / initial_surface_fuel
        
        # Map to colors: Green (1.0) -> Black (0.0)
        # We use a colormap: 'YlGn_r' (Yellow-Green reversed) or custom
        # Let's map 0->Black, 1->Green.
        # Simple hack: Use scalar mappable
        # This part is computationally expensive for matplotlib surface, 
        # so we might skip surface color update if speed is an issue.
        # For a "Final Project" visual, we want it to look good, so let's try it.
        
        # Note: modifying plot_surface facecolors is tricky in animation. 
        # Often easier to just visualize the fire dots.
        # Let's stick to just animating the FIRE for performance/stability.
        
        time_text.set_text(f"Frame: {frame}/{num_frames}")
        return flame_scatter, time_text

    # Create Animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=100, blit=False
    )

    if save_video:
        output_file = f"playback_run_{run_id}.mp4"
        print(f"Saving video to {output_file} (this may take a minute)...")
        try:
            # Try writing with ffmpeg (best quality)
            ani.save(output_file, writer='ffmpeg', fps=fps, dpi=150)
        except Exception:
            # Fallback to Pillow (GIF) if ffmpeg is missing
            print("FFmpeg not found/failed. Saving as GIF instead.")
            output_file = f"playback_run_{run_id}.gif"
            ani.save(output_file, writer='pillow', fps=fps)
        
        print(f"Saved: {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Playback of Wildfire Simulation")
    parser.add_argument("--folder", type=str, default="./training_data_v2", help="Folder containing .npz files")
    parser.add_argument("--id", type=str, required=True, help="Run ID to visualize")
    parser.add_argument("--save", action="store_true", help="Save as video file (MP4/GIF) instead of showing")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for video")
    
    args = parser.parse_args()
    
    create_animation(args.folder, args.id, args.save, args.fps)