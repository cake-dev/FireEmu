import streamlit as st
import numpy as np
import os
import sys
import time
from quicfire_io import QuicFireIO

# Import the realtime runner
# Assumes qf_realtime_runner.py is in the same directory
import qf_realtime_runner

st.set_page_config(page_title="QUIC-Fire Realtime", layout="wide", page_icon="🔥")

# --- Custom CSS for Dark Mode & Layout ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #464b5d;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 14px;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def render_frame(fuel_3d, rr_3d):
    """
    Composites 3D fuel and reaction rate into a 2D RGB image.
    Visual style: Top-down map.
    """
    nx, ny, nz = fuel_3d.shape
    
    # 1. Fuel Map (Green/Earth tones)
    # Sum fuel over Z or take max (depending on preference). 
    # Surface fuel (index 0) is usually most relevant for ground view.
    fuel_2d = fuel_3d[:, :, 0] 
    
    # Normalize fuel for display (assuming max fuel ~1.0 kg/m3 usually)
    fuel_norm = np.clip(fuel_2d / 1.0, 0, 1)
    
    # 2. Fire Map (Red/Orange)
    # Max project reaction rate to see flame intensity at any height
    fire_2d = np.max(rr_3d, axis=2)
    
    # Normalize fire (reaction rates are small, e.g., 0.0 to 0.1)
    # Using a non-linear scaling to make small fires visible
    fire_norm = np.clip(fire_2d * 20.0, 0, 1)
    
    # 3. Create RGB Buffer
    # Shape: (nx, ny, 3)
    # Background: Black
    img = np.zeros((nx, ny, 3), dtype=np.float32)
    
    # Fuel Channel (Greenish)
    img[:, :, 1] = fuel_norm * 0.6  # Green
    img[:, :, 0] = fuel_norm * 0.3  # Slight Red for brown tint
    
    # Fire Overlay (Red/Orange)
    # Where fire exists, it overwrites fuel
    mask_fire = fire_norm > 0.05
    img[mask_fire, 0] = 1.0        # Red
    img[mask_fire, 1] = 1.0 - fire_norm[mask_fire] * 0.5 # Yellow-to-Red gradient
    img[mask_fire, 2] = 0.0
    
    # Rotate for display (Streamlit images are usually (Height, Width))
    # QUIC-Fire arrays are (x, y), so we often need to rotate 90 deg
    img = np.rot90(img)
    
    return img

def main():
    st.sidebar.title("🔥 QUIC-Fire Live")
    
    # --- Input Configuration ---
    input_dir = st.sidebar.text_input("Input Directory", value="qf_inp")
    
    if not os.path.exists(input_dir):
        st.sidebar.error("Directory not found.")
        st.stop()
        
    st.sidebar.success(f"Found inputs in: {input_dir}")
    
    # --- Simulation Params ---
    st.sidebar.markdown("### Playback Controls")
    speed_factor = st.sidebar.slider("Simulation Steps per Frame", 1, 50, 5, 
                                     help="Run N physics steps for every 1 frame shown.")
    target_fps = st.sidebar.slider("Target FPS", 1, 30, 10, 
                                   help="Slow down the playback to making it readable.")
    
    # --- Session State for control ---
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Start Simulation"):
        st.session_state.running = True
    if col2.button("Stop"):
        st.session_state.running = False
        
    # --- Main Display Area ---
    # Placeholders for dynamic content
    status_text = st.empty()
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1: p_time = st.empty()
    with m2: p_wind = st.empty()
    with m3: p_fuel = st.empty()
    with m4: p_fire = st.empty()
    
    # Main Image
    image_display = st.empty()
    
    if st.session_state.running:
        try:
            # 1. Load Data (Using existing QuicFireIO)
            status_text.info("Loading inputs...")
            
            # =========================================================================
            # 1. PARSE GRID CONFIGURATION
            # =========================================================================
            print("[1/6] Reading grid configuration...")
            
            simparams_path = os.path.join(input_dir, "QU_simparams.inp")
            if not os.path.exists(simparams_path):
                print(f"ERROR: Required file not found: {simparams_path}")
                return 1
            
            sim_params = QuicFireIO.read_simparams(simparams_path)
            
            fire_params_path = os.path.join(input_dir, "QUIC_fire.inp")
            if os.path.exists(fire_params_path):
                fire_params = QuicFireIO.read_quic_fire_inp(fire_params_path)
            else:
                print("WARNING: QUIC_fire.inp not found. Using defaults.")
                fire_params = {'sim_time': 300, 'dt': 1, 'out_int_fire': 10, 'nz_fire': 32}
            
            # Get georeferencing
            origin_x, origin_y = QuicFireIO.read_raster_origin(input_dir)
            
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
            
            sensor_path = os.path.join(input_dir, "sensor1.inp")
            wind_schedule = QuicFireIO.read_sensor1(sensor_path)
            
            if wind_schedule:
                print(f"  Loaded {len(wind_schedule)} wind setpoints")
            else:
                print("  No wind schedule found. Using default: 5 m/s from 225°")
                wind_schedule = [(0, 5.0, 225.0)]

            # =========================================================================
            # 3. READ FUEL DATA
            # =========================================================================
            print("\n[3/6] Reading fuel data...")
            
            fuel_path = os.path.join(input_dir, "treesrhof.dat")
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
            moist_path = os.path.join(input_dir, "treesmoist.dat")
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
                os.path.join(input_dir, "QU_TopoInputs.inp")
            )
            topo_flag = topo_config.get('topo_flag', 0)
            
            terrain_meters = np.zeros((nx, ny), dtype=np.float32)
            
            if topo_flag == 5:
                # Custom terrain file
                fname = topo_config.get('filename', 'usgs_dem.dat')
                topo_path = os.path.join(input_dir, fname)
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
            
            ignite_path = os.path.join(input_dir, "ignite.dat")
            ignition_data = QuicFireIO.read_ignite_dat(ignite_path)
            
            if ignition_data and ignition_data.get('type') == 5:
                n_lines = len(ignition_data.get('lines', []))
                print(f"  ATV ignition (Type 5) with {n_lines} lines")
                
                ig_payload = ignition_data
            else:
                print("  No valid ignite.dat. Using default center ignition")
                ig_payload = setup_default_ignition(nx, ny, nz)
                print(f"  Created {len(ig_payload)} ignition points at domain center")
            
            # --- Prepare Payload ---
            sim_payload = {
                'wind_schedule': wind_schedule,
                'moisture': default_moisture,
                'ignition': ig_payload,
                'custom_fuel': fuel_density,
                'custom_terrain': terrain_indices,
                'qf_config': qf_config
            }
            
            # --- Run Iterator ---
            status_text.text("Simulation Running...")
            
            # Initialize Generator
            sim_gen = qf_realtime_runner.simulation_iterator(sim_payload)
            
            # Loop
            initial_fuel = np.sum(fuel_density)
            
            for step_data in sim_gen:
                if not st.session_state.running:
                    status_text.warning("Stopped by user.")
                    break
                
                step = step_data['step']
                
                # Update visuals every N steps
                if step % speed_factor == 0:
                    # Update Metrics
                    t = step_data['time']
                    ws = step_data['wind_speed']
                    wd = step_data['wind_dir']
                    
                    # Calc stats
                    rr = step_data['reaction_rate']
                    active_cells = np.sum(rr > 0.0001)
                    curr_fuel = np.sum(step_data['fuel'])
                    consumed = (1 - (curr_fuel/initial_fuel)) * 100
                    
                    # HTML Metrics
                    p_time.markdown(f"<div class='metric-card'><div class='metric-value'>{t:.1f}s</div><div class='metric-label'>Sim Time</div></div>", unsafe_allow_html=True)
                    p_wind.markdown(f"<div class='metric-card'><div class='metric-value'>{ws:.1f} m/s</div><div class='metric-label'>Wind {wd:.0f}°</div></div>", unsafe_allow_html=True)
                    p_fuel.markdown(f"<div class='metric-card'><div class='metric-value'>{consumed:.1f}%</div><div class='metric-label'>Fuel Burned</div></div>", unsafe_allow_html=True)
                    p_fire.markdown(f"<div class='metric-card'><div class='metric-value'>{active_cells}</div><div class='metric-label'>Active Cells</div></div>", unsafe_allow_html=True)
                    
                    # Render Image
                    img = render_frame(step_data['fuel'], step_data['reaction_rate'])
                    image_display.image(img, caption=f"Top-Down View (Step {step})", clamp=True, width=600)
                    
            status_text.success("Simulation Complete.")
            
        except Exception as e:
            st.error(f"Simulation Error: {e}")
            # Usually implies CUDA error or missing file
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()