import argparse
import os
import shutil
import numpy as np
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
    dz = qf_config.get('dz', 1.0) # Vertical cell size
    
    print(f"Grid: {nx}x{ny}x{nz} | Origin: {origin_x}, {origin_y} | dz: {dz}")
    
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
        
    # 4. Read Terrain
    print("Reading Terrain Data...")
    # Parse QU_TopoInputs.inp to find out if we need to load a custom file
    topo_config = QuicFireIO.read_topo_inputs(os.path.join(args.input_dir, "QU_TopoInputs.inp"))
    topo_flag = topo_config.get('topo_flag', 0)
    
    terrain_meters = np.zeros((nx, ny), dtype=np.float32)
    
    if topo_flag == 5:
        # Custom terrain file (e.g. usgs_dem.dat)
        fname = topo_config.get('filename', 'usgs_dem.dat')
        topo_path = os.path.join(args.input_dir, fname)
        print(f"Loading custom terrain from: {fname}")
        terrain_meters = QuicFireIO.read_topo_dat(topo_path, nx, ny)
    elif topo_flag == 0:
        print("Terrain flag is 0 (Flat). Using flat terrain.")
    else:
        print(f"Terrain flag {topo_flag} not fully supported in PyFire direct-load. Defaulting to flat.")

    # PREPARE TERRAIN INDICES
    # The solver calculates: elevation_meters = terrain_indices * dz
    # So we must pass indices = meters / dz
    terrain_indices = terrain_meters / dz

    # 5. Handle Ignition
    print("Reading Ignition Data...")
    ignite_path = os.path.join(args.input_dir, "ignite.dat")
    ignition_data = QuicFireIO.read_ignite_dat(ignite_path)

    if ignition_data and ignition_data['type'] == 5:
        print(f"Loaded ATV Ignition (Type 5) with {len(ignition_data['lines'])} lines.")
        ig_payload = ignition_data
        print(f"IG payload: {ig_payload}")
        # add z offset to payload

    else:
        # Fallback to default center point
        print("No valid ignite.dat or unknown type. Using default center ignition.")
        ig_x, ig_y = int(nx // 2), int(ny // 2)
        # Ignite at index 1 (just above surface)
        ig_z = 1 
        ig_list = [{'x': ig_x, 'y': ig_y, 'z': ig_z}]
        # Expand simple list
        for z_off in range(1, 10):
            ig_list.append({'x': ig_x, 'y': ig_y, 'z': ig_z+z_off})
        
        ig_payload = ig_list

    # Payload
    sim_params_payload = {
        'wind_schedule': wind_schedule,
        'moisture': 0.05,
        'ignition': ig_payload,
        'custom_fuel': fuel_density,
        'custom_terrain': terrain_indices, # Passing indices is correct for run_gpu_qf logic
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