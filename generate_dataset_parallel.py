import multiprocessing as mp
import os
import time
import numpy as np
import matplotlib
# Set backend to Agg to avoid "display not found" errors on HPC/Headless servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
# import config_new as config
import config_stable as config
# import run_gpu_new as run_gpu
import run_gpu_stable as run_gpu
import world_gen

# Output directory
DATA_DIR = "./training_data_stable"
STATS_DIR = "./training_data_v2_stats"
NUM_SAMPLES = 8
NUM_WORKERS = 8  # A100 has 40GB VRAM, can handle multiple sims

def generate_single_sample(run_id):
    """
    Worker function to run one full simulation cycle.
    Returns: (success_bool, params_dict)
    """
    try:
        # 1. Generate Params (CPU Heavy)
        nx, ny, nz = config.NX, config.NY, config.NZ
        world_path = "world_data.npz"
        load_world = False
        if load_world and os.path.exists(world_path):
            data = np.load(world_path)
            fuel_grid = data['fuel']
            terrain_grid = data['terrain_z']
        else:
            fuel_grid, terrain_grid = world_gen.generate_world(nx, ny, nz)
        
        speed = np.random.uniform(10.0, 20.0)
        direction = np.random.uniform(0.0, 360.0)
        # direction = np.random.choice([0.0, 90.0, 180.0, 270.0])
        moisture = np.random.uniform(0.05, 0.4)
        # moisture = np.random.choice([0.1, 0.5, 1.0, 100])

        # defaults
        # speed = 20.0
        # direction = 180.0
        # moisture = 0.5
        
        # Smart Ignition Logic
        # ig_x, ig_y, ig_z = 0, 0, 0
        # for _ in range(20):
        #     rx = np.random.randint(nx * 0.1, nx * 0.9)
        #     ry = np.random.randint(ny * 0.1, ny * 0.9)
        #     ground_z = int(terrain_grid[rx, ry])
        #     if ground_z < nz and fuel_grid[ground_z, rx, ry] > 0:
        #         ig_x, ig_y, ig_z = rx, ry, ground_z
        #         break
        # else:
        #     ig_x, ig_y = nx // 2, ny // 2
        #     ig_z = int(terrain_grid[ig_x, ig_y])
        ig_x, ig_y = nx // 2, ny // 2
        ig_z = int(terrain_grid[ig_x, ig_y])
        # randomly place within central 50%
        # ig_x = np.random.randint(nx * 0.25, nx * 0.75)
        # ig_y = np.random.randint(ny * 0.25, ny * 0.75)
        # ig_z = int(terrain_grid[ig_x, ig_y])
        params = {
            'wind_speed': speed,
            'wind_dir': direction,
            'moisture': moisture,
            'ignition': [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}],
            'custom_fuel': fuel_grid,
            'custom_terrain': terrain_grid
        }

        # 2. Run Simulation (GPU Heavy)
        run_gpu.run_simulation(params, run_id=run_id, output_dir=DATA_DIR)
        
        # Return params for statistics (exclude heavy arrays)
        stats = {
            'speed': speed,
            'direction': direction,
            'moisture': moisture
        }
        return True, stats
        
    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        return False, None

def plot_distributions(stats_list, output_dir):
    """
    Generates plots for the dataset distributions.
    """
    if not stats_list:
        print("No stats to plot.")
        return

    speeds = [s['speed'] for s in stats_list]
    directions = [s['direction'] for s in stats_list]
    moistures = [s['moisture'] for s in stats_list]

    fig = plt.figure(figsize=(18, 5))

    # 1. Wind Speed Histogram
    ax1 = fig.add_subplot(131)
    ax1.hist(speeds, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Wind Speed Distribution (m/s)')
    ax1.set_xlabel('Speed (m/s)')
    ax1.set_ylabel('Count')

    # 2. Wind Direction (Polar Rose Plot)
    ax2 = fig.add_subplot(132, projection='polar')
    # Convert degrees to radians for polar plot
    rads = np.radians(directions)
    ax2.hist(rads, bins=36, color='salmon', edgecolor='black', alpha=0.7)
    ax2.set_title('Wind Direction Distribution')
    ax2.set_theta_zero_location('N') # 0 degrees at top
    ax2.set_theta_direction(-1)      # Clockwise

    # 3. Moisture Histogram
    ax3 = fig.add_subplot(133)
    ax3.hist(moistures, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_title('Fuel Moisture Distribution')
    ax3.set_xlabel('Moisture Factor')
    ax3.set_ylabel('Count')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dataset_stats.png")
    plt.savefig(plot_path)
    print(f"Distribution plots saved to {plot_path}")
    plt.close()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    # CRITICAL: Numba/CUDA requires 'spawn' start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"Generating {NUM_SAMPLES} simulations with {NUM_WORKERS} parallel workers...")
    start_time = time.time()
    
    successful_stats = []

    # Create a pool of workers
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # Use imap_unordered for better parallel efficiency
        results = list(tqdm(pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))
        
    # Process results
    success_count = 0
    for success, stats in results:
        if success:
            success_count += 1
            successful_stats.append(stats)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Completed {success_count}/{NUM_SAMPLES} runs.")
    print(f"Total time: {duration:.2f}s")
    print(f"Average time per sample: {duration/NUM_SAMPLES:.2f}s")
    
    # # Generate Plots
    print("Generating distribution statistics...")
    plot_distributions(successful_stats, STATS_DIR)

if __name__ == "__main__":
    main()