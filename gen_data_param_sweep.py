import multiprocessing as mp
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
import config
import run_sweep
import world_gen

# --- CONFIGURATION ---
DATA_DIR = "./training_data_sweep_7"
STATS_DIR = "./training_data_sweep_7_stats"
NUM_WORKERS = 8

# Define your parameter sweep here.
# Each dictionary overrides the defaults in config.py for that specific run.
PHYSICS_VARIANTS = [
    {'name': 'dt_1_0_dx_2_0_slope0',      'dt': 1.0, 'dx': 2.0, 'mod_dt': False, 'jump_hack': False, 'eep': 50000.0, 'slope_factor': 0.0},
    {'name': 'dt_1_0_dx_2_0_jump_hack_slope0',      'dt': 1.0, 'dx': 2.0, 'mod_dt': False, 'jump_hack': True, 'eep': 50000.0, 'slope_factor': 0.0},
    {'name': 'dt_0_1_dx_2_0_slope0',      'dt': 0.1, 'dx': 2.0, 'mod_dt': False, 'jump_hack': False, 'eep': 50000.0, 'slope_factor': 0.0},
    {'name': 'dt_0_1_dx_2_0_mod_dt_slope0',      'dt': 0.1, 'dx': 2.0, 'mod_dt': True, 'jump_hack': False, 'eep': 50000.0, 'slope_factor': 0.0},
    {'name': 'dt_1_0_dx_2_0_slope4',      'dt': 1.0, 'dx': 2.0, 'mod_dt': False, 'jump_hack': False, 'eep': 50000.0, 'slope_factor': 4.0},
    {'name': 'dt_1_0_dx_2_0_jump_hack_slope4',      'dt': 1.0, 'dx': 2.0, 'mod_dt': False, 'jump_hack': True, 'eep': 50000.0, 'slope_factor': 4.0},
    {'name': 'dt_0_1_dx_2_0_slope4',      'dt': 0.1, 'dx': 2.0, 'mod_dt': True, 'jump_hack': True, 'eep': 50000.0, 'slope_factor': 4.0},
    {'name': 'dt_0_1_dx_2_0_mod_dt_slope4',      'dt': 0.1, 'dx': 2.0, 'mod_dt': True, 'jump_hack': True, 'eep': 50000.0, 'slope_factor': 4.0},
]

def generate_single_sample(args):
    """
    Worker function.
    Args: (run_id, variant_config, world_path)
    """
    run_id, variant, world_path = args
    
    try:
        # 1. Load the SHARED World (Same landscape for everyone)
        if not os.path.exists(world_path):
            raise FileNotFoundError("World file not generated yet.")
            
        data = np.load(world_path)
        fuel_grid = data['fuel']
        terrain_grid = data['terrain_z']
        nx, ny, nz = fuel_grid.shape[1], fuel_grid.shape[2], fuel_grid.shape[0]

        # 2. Set Random Environmental Conditions (Wind/Moisture)
        # Note: If you want these fixed per run too, move them to PHYSICS_VARIANTS
        # speed = np.random.uniform(10.0, 20.0)
        # direction = np.random.uniform(0.0, 360.0)
        # moisture = np.random.uniform(0.1, 1.0)

        speed = 10.0
        direction = 180.0
        moisture = 0.5
        
        # 3. Ignition (Fixed to center or random? Let's use logic)
        # ig_x, ig_y = nx // 2, ny // 2
        # ig_z = int(terrain_grid[ig_x, ig_y])
        # place ignition center near the bottom
        ig_x = nx // 2
        ig_y = ny // 4
        ig_z = int(terrain_grid[ig_x, ig_y])

        # 4. Pack Parameters
        # We merge the variant config into the params dictionary
        params = {
            'run_name': variant['name'],
            'wind_speed': speed,
            'wind_dir': direction,
            'moisture': moisture,
            'ignition': [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}],
            'custom_fuel': fuel_grid,
            'custom_terrain': terrain_grid,
            
            # INJECT PHYSICS OVERRIDES HERE
            'dt': variant.get('dt', config.DT),
            'dx': variant.get('dx', config.DX),
            'dy': variant.get('dx', config.DY), # Assuming square cells usually
            'mod_dt': variant.get('mod_dt', config.MOD_DT),
            'jump_hack': variant.get('jump_hack', config.JUMP_HACK),
            'slope_factor': variant.get('slope_factor', config.SLOPE_FACTOR),
            'eep': variant.get('eep', config.EEP)
        }

        # 5. Run Simulation
        # Ensure your run_sweep.py saves these 'dt'/'dx' keys into the .npz output!
        run_sweep.run_simulation(params, run_id=run_id, output_dir=DATA_DIR)
        
        return True, {'id': run_id, 'variant': variant['name']}
        
    except Exception as e:
        print(f"Run {run_id} ({variant['name']}) failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # os.makedirs(STATS_DIR, exist_ok=True)
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. Generate ONE World for all experiments
    print("Generating shared world landscape...")
    nx, ny, nz = config.NX, config.NY, config.NZ
    fuel, terrain = world_gen.generate_world(nx, ny, nz)
    world_path = os.path.join(DATA_DIR, "shared_world.npz")
    np.savez(world_path, fuel=fuel, terrain_z=terrain)
    print(f"World saved to {world_path}")

    # 2. Prepare Tasks
    # We will create a task for every variant
    tasks = []
    run_counter = 0
    
    # Example: Run each variant n times
    SAMPLES_PER_VARIANT = 1
    
    for _ in range(SAMPLES_PER_VARIANT):
        for variant in PHYSICS_VARIANTS:
            # Tuple: (run_id, specific_config, path_to_world)
            tasks.append((run_counter, variant, world_path))
            run_counter += 1

    print(f"Queueing {len(tasks)} simulations on {NUM_WORKERS} workers...")
    
    start_time = time.time()
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(generate_single_sample, tasks), total=len(tasks)))
        
    success_count = sum(1 for r in results if r[0])
    
    end_time = time.time()
    print(f"Completed {success_count}/{len(tasks)} runs in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()