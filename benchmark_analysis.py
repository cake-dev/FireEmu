import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from tqdm import tqdm

# Import your modules
import config
import run_gpu
import world_gen

# --- SETUP ---
OUTPUT_DIR = "./benchmark_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure multiprocessing uses 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def run_single_param_set(params, run_id):
    """Wrapper to run a single simulation and return duration."""
    start = time.time()
    try:
        run_gpu.run_simulation(params, run_id, OUTPUT_DIR)
        return time.time() - start
    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        return None

def analyze_scaling_nx():
    print("\n--- Benchmarking World Size Scaling (NX) ---")
    
    # Test different grid sizes (assuming square grid NX=NY)
    # Keep sizes powers of 2 for GPU efficiency
    sizes = [64, 128, 256, 512]
    times = []
    
    # Base params
    params = {
        'wind_speed': 10.0,
        'wind_dir': 45.0,
        'moisture': 0.5,
        'ignition': [{'x': 32, 'y': 32, 'z': 1}] # Default ignition
    }

    for size in sizes:
        print(f"Running grid size: {size}x{size}x{config.NZ}...")
        
        # MONKEY PATCH CONFIG
        # We modify the global config object before the simulation runs
        config.NX = size
        config.NY = size
        
        # Need to regenerate world for new size
        fuel, terrain = world_gen.generate_world(size, size, config.NZ)
        params['custom_fuel'] = fuel
        params['custom_terrain'] = terrain
        # Update ignition to be safe in new bounds
        params['ignition'] = [{'x': size//2, 'y': size//2, 'z': int(terrain[size//2, size//2])}]

        # Run 1 sample to measure pure simulation time
        duration = run_single_param_set(params, run_id=f"bench_size_{size}")
        if duration:
            times.append(duration)
        else:
            times.append(0)

    # Reset Config
    config.NX = 128
    config.NY = 128
    
    return sizes, times

def analyze_scaling_workers():
    print("\n--- Benchmarking Parallel Worker Scaling ---")
    
    # Fixed workload: Generate n simulations
    NUM_SAMPLES = 32  # Total samples to run
    worker_counts = [1, 2, 4, 8, 16, 32]
    total_times = []
    
    # Generate dummy params list
    params_list = []
    for i in range(NUM_SAMPLES):
        fuel, terrain = world_gen.generate_world(config.NX, config.NY, config.NZ)
        p = {
            'wind_speed': 10.0, 
            'wind_dir': 0.0, 
            'moisture': 0.5,
            'ignition': [{'x': 64, 'y': 64, 'z': 1}],
            'custom_fuel': fuel,
            'custom_terrain': terrain
        }
        params_list.append((p, i))

    # Worker Wrapper for Pool
    def _worker_task(args):
        p, rid = args
        return run_gpu.run_simulation(p, f"bench_work_{rid}", OUTPUT_DIR)

    for n_workers in worker_counts:
        print(f"Testing with {n_workers} workers...")
        
        start_t = time.time()
        
        # Use Pool to run tasks
        with mp.Pool(processes=n_workers) as pool:
            # We map a wrapper function
            pool.starmap(run_gpu.run_simulation, 
                         [(p, f"bench_w{n_workers}_{i}", OUTPUT_DIR) for i, (p, _) in enumerate(params_list)])
            
        duration = time.time() - start_t
        total_times.append(duration)
        print(f"  -> Took {duration:.2f}s")

    return worker_counts, total_times

def analyze_physics_wind_vs_burned():
    print("\n--- Analyzing Physics: Wind Speed vs Burned Area ---")
    
    wind_speeds = [2.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    burned_amounts = []
    
    # Generate one world to keep terrain/fuel constant for fair comparison
    fuel_grid, terrain_grid = world_gen.generate_world(config.NX, config.NY, config.NZ)
    
    for speed in tqdm(wind_speeds):
        run_id = f"physics_wind_{int(speed)}"
        
        params = {
            'wind_speed': float(speed),
            'wind_dir': 90.0, # Constant direction
            'moisture': 0.5,
            'ignition': [{'x': 32, 'y': 64, 'z': int(terrain_grid[32, 64])}], # Start on one side
            'custom_fuel': fuel_grid.copy(), # Important: Send copy so previous run doesn't corrupt
            'custom_terrain': terrain_grid
        }
        
        # Run sim
        run_gpu.run_simulation(params, run_id, OUTPUT_DIR)
        
        # LOAD RESULTS to calculate burned area
        # run_gpu saves to: os.path.join(output_dir, f"run_{run_id}.npz")
        data_path = os.path.join(OUTPUT_DIR, f"run_{run_id}.npz")
        data = np.load(data_path)
        
        # History fuel shape: (num_frames, nx, ny, nz)
        fuel_history = data['fuel']
        
        # Calculate total mass lost
        initial_mass = np.sum(fuel_history[0])
        final_mass = np.sum(fuel_history[-1])
        mass_consumed = initial_mass - final_mass
        
        burned_amounts.append(mass_consumed)
        
    return wind_speeds, burned_amounts

if __name__ == "__main__":
    # 1. Run Experiments
    nx_sizes, nx_times = analyze_scaling_nx()
    worker_counts, worker_times = analyze_scaling_workers()
    wind_vals, burn_vals = analyze_physics_wind_vs_burned()

    # 2. Generate Plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: NX Scaling
    axs[0].plot(nx_sizes, nx_times, 'o-', color='tab:blue', linewidth=2)
    axs[0].set_title("Runtime vs World Size (Single Sim)")
    axs[0].set_xlabel("Grid Dimension N (NxNx32)")
    axs[0].set_ylabel("Time (seconds)")
    axs[0].grid(True)
    # Optional: Log scale if scaling is massive
    # axs[0].set_xscale('log', base=2)
    # axs[0].set_yscale('log')

    # Plot 2: Worker Scaling
    # Plot Speedup ideally, but raw time works too
    axs[1].plot(worker_counts, worker_times, 's--', color='tab:orange', linewidth=2)
    axs[1].set_title(f"Batch Processing Time (16 Samples)")
    axs[1].set_xlabel("Number of Parallel Workers")
    axs[1].set_ylabel("Total Time (seconds)")
    axs[1].grid(True)
    
    # Plot ideal scaling line for comparison
    ideal_times = [worker_times[0] / n for n in worker_counts]
    axs[1].plot(worker_counts, ideal_times, 'k:', label='Ideal Linear Scaling')
    axs[1].legend()

    # Plot 3: Physics Analysis
    axs[2].bar(wind_vals, burn_vals, width=2.0, color='tab:red', alpha=0.7, edgecolor='black')
    axs[2].plot(wind_vals, burn_vals, 'r-')
    axs[2].set_title("Fire Severity vs Wind Speed")
    axs[2].set_xlabel("Wind Speed (m/s)")
    axs[2].set_ylabel("Total Fuel Mass Consumed (kg equivalent)")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hpc_project_analysis.png")
    print("\nAnalysis Complete. Plots saved to 'hpc_project_analysis.png'")