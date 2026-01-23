import argparse
import os
import shutil
import numpy as np
import world_gen
from quicfire_io import QuicFireIO
import run_gpu_qf as run_gpu

def main():
    parser = argparse.ArgumentParser(description="Run Fire Sim in QUIC-Fire Mode (Real Inputs)")
    parser.add_argument("--input-dir", type=str, default="./qf_inp", help="Directory containing .inp and .dat files")
    parser.add_argument("--output-dir", type=str, default="./qf_out", help="Directory for outputs")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Parse Configs
    print("Reading QUIC-Fire configurations...")
    sim_params = QuicFireIO.read_simparams(os.path.join(args.input_dir, "QU_simparams.inp"))
    fire_params = QuicFireIO.read_quic_fire_inp(os.path.join(args.input_dir, "QUIC_fire.inp"))
    origin_x, origin_y = QuicFireIO.read_raster_origin(args.input_dir)
    
    qf_config = {**sim_params, **fire_params}
    qf_config['origin_x'] = origin_x
    qf_config['origin_y'] = origin_y
    
    nx, ny, nz = qf_config['nx'], qf_config['ny'], qf_config['nz_fire']
    
    print(f"Grid: {nx}x{ny}x{nz} | Origin: {origin_x}, {origin_y}")
    
    # 2. Read Wind Schedule
    print("Reading Wind Schedule...")
    wind_schedule = QuicFireIO.read_sensor1(os.path.join(args.input_dir, "sensor1.inp"))
    if wind_schedule:
        print(f"Loaded {len(wind_schedule)} wind setpoints.")
    else:
        print("No wind schedule found. Using defaults.")

    # 3. Read Fuels
    print("Reading Fuel Data...")
    fuel_density = QuicFireIO.read_fuel_dat(
        os.path.join(args.input_dir, "treesrhof.dat"), 
        nx, ny, nz, 
        file_format=qf_config.get('fuel_file_format', 1)
    )
    
    # Optional: Read Moisture if available, else default
    fuel_moisture = QuicFireIO.read_fuel_dat(
        os.path.join(args.input_dir, "treesmoist.dat"), 
        nx, ny, nz, 
        file_format=qf_config.get('fuel_file_format', 1)
    )
    # Check if moisture file was empty/missing (returns zeros)
    if np.max(fuel_moisture) == 0:
        print("Moisture file missing or empty. Using default 0.05.")
        fuel_moisture = None # Signal to use param default
        
    # 4. Read Terrain (Optional)
    # If using topo, QF often uses QU_TopoInputs.inp logic or binary files.
    # For now, we generate flat or look for a specific file if needed. 
    # Let's default to flat unless we implement full TopoInputs parser.
    # User can supply custom_terrain array if they have a loader.
    print("Generating default terrain (Flat)...")
    terrain = np.zeros((nx, ny), dtype=np.float32)
    
    # 5. Default Ignition (Center) if not using ignite.dat parser yet
    ig_x, ig_y = nx // 2, ny // 2
    # Find Z surface
    ig_z = 0 # Flat
    
    # Payload
    sim_params_payload = {
        'wind_schedule': wind_schedule,
        'moisture': 0.05,
        'ignition': [{'x': ig_x, 'y': ig_y, 'z': ig_z}],
        'custom_fuel': fuel_density,
        'custom_terrain': terrain,
        'qf_config': qf_config
    }
    
    if fuel_moisture is not None:
        sim_params_payload['custom_moisture'] = fuel_moisture
    
    # 6. Run
    print("Starting Simulation...")
    run_gpu.run_simulation(sim_params_payload, run_id="qf_real_mode", output_dir=args.output_dir)
    print("Simulation Complete.")

if __name__ == "__main__":
    main()