import os
import numpy as np
import time
from tqdm import tqdm
import config
import run_gpu 
import world_gen

# Output directory
DATA_DIR = "./training_data_v1"
os.makedirs(DATA_DIR, exist_ok=True)

NUM_SAMPLES = 1000  # How many simulations to run

def generate_random_params():
    """Samples the parameter space."""
    nx, ny, nz = config.NX, config.NY, config.NZ
    
    # 1. Generate World FIRST so we know where the ground is
    fuel_grid, terrain_grid = world_gen.generate_world(config.NX, config.NY, config.NZ)
    
    # 2. Random Wind & Moisture
    speed = np.random.uniform(0.1, 25.0)
    direction = np.random.uniform(0.0, 360.0)
    moisture = np.random.uniform(0.1, 1.5) 
    
    # 3. Smart Ignition
    # Find a valid point on the surface that actually has fuel
    # Try up to 20 times to find a good spot
    ig_x, ig_y, ig_z = 0, 0, 0
    
    for _ in range(20):
        # Pick random X, Y (staying away from extreme edges)
        rx = np.random.randint(nx * 0.1, nx * 0.9)
        ry = np.random.randint(ny * 0.1, ny * 0.9)
        
        # Look up the ground level at this X, Y
        # terrain_grid values are floats, need int index
        ground_z = int(terrain_grid[rx, ry])
        
        # Check if point is valid and has fuel
        # We check ground_z (surface) and ground_z + 1 (just above surface)
        if ground_z < nz and fuel_grid[ground_z, rx, ry] > 0:
            ig_x, ig_y, ig_z = rx, ry, ground_z
            break
    else:
        # Fallback if we fail to find fuel (e.g. extremely sparse map)
        ig_x = nx // 2
        ig_y = ny // 2
        ig_z = int(terrain_grid[ig_x, ig_y])
    
    return {
        'wind_speed': speed,
        'wind_dir': direction,
        'moisture': moisture,
        'ignition': [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}],
        'custom_fuel': fuel_grid,
        'custom_terrain': terrain_grid
    }

def main():
    print(f"Generating {NUM_SAMPLES} simulations on Jetstream2 GPU...")
    
    start_time = time.time()
    
    for i in tqdm(range(NUM_SAMPLES)):
        params = generate_random_params()
        
        try:
            run_gpu.run_simulation(params, run_id=i, output_dir=DATA_DIR)
        except Exception as e:
            print(f"Run {i} failed: {e}")
            
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()