"""
quicfire_interface.py - QUIC-Fire Input Parser and Simulation Interface

This module reads standard QUIC-Fire input files and launches the GPU simulation.

Supported input files:
- QU_simparams.inp: Grid dimensions
- QUIC_fire.inp: Fire simulation parameters
- sensor1.inp: Wind schedule
- treesrhof.dat: Fuel density
- treesmoist.dat: Fuel moisture (optional)
- QU_TopoInputs.inp + terrain file: Topography
- ignite.dat: Ignition configuration

References:
- Linn et al. (2020) QUIC-Fire paper
- QUIC-Fire User's Guide
"""

import argparse
import os
import numpy as np
from quicfire_io import QuicFireIO
import run_gpu_qf as run_gpu


def validate_inputs(qf_config, fuel_density, terrain, ignition_data):
    """
    Validates input data and reports any issues.
    
    Returns:
    --------
    bool: True if inputs are valid, False otherwise
    list: Warning messages
    """
    warnings = []
    is_valid = True
    
    nx, ny, nz = qf_config['nx'], qf_config['ny'], qf_config.get('nz_fire', qf_config.get('nz', 32))
    
    # Check fuel
    if fuel_density is None or np.max(fuel_density) == 0:
        warnings.append("WARNING: No fuel data loaded. Fire will not spread.")
        is_valid = False
    else:
        fuel_cells = np.sum(fuel_density > 0)
        fuel_frac = fuel_cells / (nx * ny * nz) * 100
        warnings.append(f"Fuel: {fuel_cells} cells with fuel ({fuel_frac:.1f}% of domain)")
        
        if fuel_frac < 1:
            warnings.append("WARNING: Very sparse fuel (<1%). Check fuel file loading.")
    
    # Check terrain
    if terrain is not None:
        elev_range = np.max(terrain) - np.min(terrain)
        warnings.append(f"Terrain: {np.min(terrain):.1f}m to {np.max(terrain):.1f}m (range: {elev_range:.1f}m)")
        
        if elev_range > nz * qf_config.get('dz', 1.0):
            warnings.append("WARNING: Terrain range exceeds grid height. May cause issues.")
    
    # Check ignition
    if ignition_data is None:
        warnings.append("WARNING: No ignition data. Using center point ignition.")
    elif isinstance(ignition_data, dict) and ignition_data.get('type') == 5:
        n_lines = len(ignition_data.get('lines', []))
        warnings.append(f"Ignition: ATV type with {n_lines} lines")
    elif isinstance(ignition_data, list):
        warnings.append(f"Ignition: Point ignition with {len(ignition_data)} points")
    
    return is_valid, warnings


def setup_default_ignition(nx, ny, nz):
    """
    Creates default center-point ignition when no ignite.dat is available.
    """
    ig_x, ig_y = int(nx // 2), int(ny // 2)
    ig_z = 0  # Surface layer
    
    # Create a small cluster of ignition points
    ig_list = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            ix = ig_x + dx
            iy = ig_y + dy
            if 0 <= ix < nx and 0 <= iy < ny:
                ig_list.append({'x': ix, 'y': iy, 'z': ig_z})
                if nz > 1:
                    ig_list.append({'x': ix, 'y': iy, 'z': 1})
    
    return ig_list


def main():
    parser = argparse.ArgumentParser(
        description="Run GPU Fire Simulation with QUIC-Fire Inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quicfire_interface.py --input-dir ./my_project/qf_inp
  python quicfire_interface.py --input-dir ./qf_inp --output-dir ./results
  
Required input files in input-dir:
  - QU_simparams.inp (grid dimensions)
  - QUIC_fire.inp (simulation parameters)
  - treesrhof.dat (fuel density)
  
Optional input files:
  - sensor1.inp (wind schedule)
  - treesmoist.dat (fuel moisture)
  - QU_TopoInputs.inp + terrain file
  - ignite.dat (ignition pattern)
        """
    )
    parser.add_argument("--input-dir", type=str, default="./qf_inp",
                        help="Directory containing QUIC-Fire input files")
    parser.add_argument("--output-dir", type=str, default="./qf_out",
                        help="Directory for simulation outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs without running simulation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()

    # =========================================================================
    # VALIDATE INPUT DIRECTORY
    # =========================================================================
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("QUIC-Fire GPU Simulation Interface")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # =========================================================================
    # 1. PARSE GRID CONFIGURATION
    # =========================================================================
    print("[1/6] Reading grid configuration...")
    
    simparams_path = os.path.join(args.input_dir, "QU_simparams.inp")
    if not os.path.exists(simparams_path):
        print(f"ERROR: Required file not found: {simparams_path}")
        return 1
    
    sim_params = QuicFireIO.read_simparams(simparams_path)
    
    fire_params_path = os.path.join(args.input_dir, "QUIC_fire.inp")
    if os.path.exists(fire_params_path):
        fire_params = QuicFireIO.read_quic_fire_inp(fire_params_path)
    else:
        print("WARNING: QUIC_fire.inp not found. Using defaults.")
        fire_params = {'sim_time': 300, 'dt': 1, 'out_int_fire': 10, 'nz_fire': 32}
    
    # Get georeferencing
    origin_x, origin_y = QuicFireIO.read_raster_origin(args.input_dir)
    
    # Combine configuration
    qf_config = {**sim_params, **fire_params}
    qf_config['origin_x'] = origin_x
    qf_config['origin_y'] = origin_y
    
    # Use nz_fire if available, otherwise nz
    if 'nz_fire' in qf_config:
        qf_config['nz'] = qf_config['nz_fire']
    
    nx, ny, nz = qf_config['nx'], qf_config['ny'], qf_config['nz']
    dx = qf_config.get('dx', 2.0)
    dy = qf_config.get('dy', 2.0)
    dz = qf_config.get('dz', 1.0)
    
    print(f"  Grid: {nx} x {ny} x {nz} cells")
    print(f"  Cell size: {dx} x {dy} x {dz} m")
    print(f"  Domain: {nx*dx} x {ny*dy} x {nz*dz} m")
    print(f"  Origin: ({origin_x}, {origin_y})")
    print(f"  Simulation time: {qf_config.get('sim_time', 300)} s")
    print(f"  Time step: {qf_config.get('dt', 1)} s")

    # =========================================================================
    # 2. READ WIND SCHEDULE
    # =========================================================================
    print("\n[2/6] Reading wind schedule...")
    
    sensor_path = os.path.join(args.input_dir, "sensor1.inp")
    wind_schedule = QuicFireIO.read_sensor1(sensor_path)
    
    if wind_schedule:
        print(f"  Loaded {len(wind_schedule)} wind setpoints")
        if args.verbose:
            for t, s, d in wind_schedule[:5]:
                print(f"    t={t}s: {s:.1f} m/s @ {d:.0f}°")
            if len(wind_schedule) > 5:
                print(f"    ... and {len(wind_schedule)-5} more")
    else:
        print("  No wind schedule found. Using default: 5 m/s from 225°")
        wind_schedule = [(0, 5.0, 225.0)]

    # =========================================================================
    # 3. READ FUEL DATA
    # =========================================================================
    print("\n[3/6] Reading fuel data...")
    
    fuel_path = os.path.join(args.input_dir, "treesrhof.dat")
    file_format = qf_config.get('fuel_file_format', 1)
    
    if os.path.exists(fuel_path):
        fuel_density = QuicFireIO.read_fuel_dat(fuel_path, nx, ny, nz, file_format)
        fuel_cells = np.sum(fuel_density > 0)
        print(f"  Loaded fuel density: {fuel_cells} cells with fuel")
        print(f"  Fuel range: {np.min(fuel_density):.3f} - {np.max(fuel_density):.3f} kg/m³")
    else:
        print(f"  WARNING: Fuel file not found: {fuel_path}")
        print("  Creating default uniform fuel layer")
        fuel_density = np.zeros((nx, ny, nz), dtype=np.float32)
        fuel_density[:, :, 0:3] = 0.5  # 0.5 kg/m³ in bottom 3 layers
    
    # Moisture
    moist_path = os.path.join(args.input_dir, "treesmoist.dat")
    fuel_moisture = None
    default_moisture = 0.05
    
    if os.path.exists(moist_path):
        fuel_moisture = QuicFireIO.read_fuel_dat(moist_path, nx, ny, nz, file_format)
        if np.max(fuel_moisture) > 0:
            print(f"  Loaded fuel moisture: {np.mean(fuel_moisture[fuel_moisture > 0]):.2%} average")
        else:
            print(f"  Moisture file empty. Using default: {default_moisture:.0%}")
            fuel_moisture = None
    else:
        print(f"  No moisture file. Using default: {default_moisture:.0%}")

    # =========================================================================
    # 4. READ TERRAIN
    # =========================================================================
    print("\n[4/6] Reading terrain data...")
    
    topo_config = QuicFireIO.read_topo_inputs(
        os.path.join(args.input_dir, "QU_TopoInputs.inp")
    )
    topo_flag = topo_config.get('topo_flag', 0)
    
    terrain_meters = np.zeros((nx, ny), dtype=np.float32)
    
    if topo_flag == 5:
        # Custom terrain file
        fname = topo_config.get('filename', 'usgs_dem.dat')
        topo_path = os.path.join(args.input_dir, fname)
        print(f"  Loading terrain from: {fname}")
        
        if os.path.exists(topo_path):
            terrain_meters = QuicFireIO.read_topo_dat(topo_path, nx, ny)
            print(f"  Terrain range: {np.min(terrain_meters):.1f} - {np.max(terrain_meters):.1f} m")
        else:
            print(f"  WARNING: Terrain file not found: {topo_path}")
            print("  Using flat terrain")
    elif topo_flag == 0:
        print("  Terrain flag = 0: Flat terrain")
    else:
        print(f"  Terrain flag {topo_flag} not supported. Using flat terrain")
    
    # Convert to grid indices
    terrain_indices = terrain_meters / dz

    # =========================================================================
    # 5. READ IGNITION DATA
    # =========================================================================
    print("\n[5/6] Reading ignition data...")
    
    ignite_path = os.path.join(args.input_dir, "ignite.dat")
    ignition_data = QuicFireIO.read_ignite_dat(ignite_path)
    
    if ignition_data and ignition_data.get('type') == 5:
        n_lines = len(ignition_data.get('lines', []))
        print(f"  ATV ignition (Type 5) with {n_lines} lines")
        
        if args.verbose and n_lines > 0:
            for i, line in enumerate(ignition_data['lines'][:3]):
                print(f"    Line {i+1}: ({line['x_start']:.0f},{line['y_start']:.0f}) -> "
                      f"({line['x_end']:.0f},{line['y_end']:.0f}) "
                      f"t=[{line['t_start']:.0f},{line['t_end']:.0f}]s")
            if n_lines > 3:
                print(f"    ... and {n_lines-3} more lines")
        
        ig_payload = ignition_data
    else:
        print("  No valid ignite.dat. Using default center ignition")
        ig_payload = setup_default_ignition(nx, ny, nz)
        print(f"  Created {len(ig_payload)} ignition points at domain center")

    # =========================================================================
    # VALIDATE INPUTS
    # =========================================================================
    print("\n[Validation]")
    is_valid, warnings = validate_inputs(qf_config, fuel_density, terrain_meters, ignition_data)
    
    for w in warnings:
        print(f"  {w}")
    
    if not is_valid:
        print("\nERROR: Input validation failed. Check warnings above.")
        if not args.dry_run:
            return 1

    # =========================================================================
    # 6. RUN SIMULATION
    # =========================================================================
    if args.dry_run:
        print("\n[Dry Run] Skipping simulation.")
        return 0
    
    print("\n[6/6] Starting simulation...")
    print("=" * 60)
    
    # Build simulation parameters
    sim_params_payload = {
        'wind_schedule': wind_schedule,
        'moisture': default_moisture,
        'ignition': ig_payload,
        'custom_fuel': fuel_density,
        'custom_terrain': terrain_indices,
        'qf_config': qf_config
    }
    
    if fuel_moisture is not None:
        sim_params_payload['custom_moisture'] = fuel_moisture
    
    # Run!
    try:
        run_gpu.run_simulation(
            sim_params_payload,
            run_id="qf_interface",
            output_dir=args.output_dir
        )
        print("\n" + "=" * 60)
        print("Simulation completed successfully!")
        print(f"Output saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        print(f"\nERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())