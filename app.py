import streamlit as st
import numpy as np
import os
import sys
import time
import plotly.graph_objects as go
from quicfire_io import QuicFireIO

# Import the realtime runner
# Assumes qf_realtime_runner.py is in the same directory
import qf_realtime_runner

st.set_page_config(page_title="Py-Fire Realtime", layout="wide", page_icon="🔥")

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
    /* Fix Plotly margins to align with image */
    .stPlotlyChart {
        height: 100%;
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
    # Surface fuel (index 0) is usually most relevant for ground view.
    fuel_2d = fuel_3d[:, :, 0] 
    
    # Normalize fuel for display (assuming max fuel ~1.0 kg/m3 usually)
    fuel_norm = np.clip(fuel_2d / 1.0, 0, 1)
    
    # 2. Fire Map (Red/Orange)
    # Max project reaction rate to see flame intensity at any height
    fire_2d = np.max(rr_3d, axis=2)
    
    # Normalize fire (reaction rates are small, e.g., 0.0 to 0.1)
    fire_norm = np.clip(fire_2d * 20.0, 0, 1)
    
    # 3. Create RGB Buffer
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

def create_surface_trace(terrain, fuel_3d, rr_3d, dx, dy, dz, x_offset=0):
    """
    Helper function to generate a Plotly Surface trace.
    Allows offsetting the X coordinates to place multiple surfaces in one scene.
    """
    # 1. Calculate Fuel Height (Canopy)
    # Create Z-level array: [0, dz, 2dz, ...]
    nz = fuel_3d.shape[2]
    z_levels = np.arange(nz) * dz
    
    # Mask where fuel exists (threshold 0.1 kg/m3 to ignore thin grass/noise in height calc)
    has_fuel = fuel_3d > 0.1
    
    # Find the maximum height index where fuel exists for each cell
    fuel_height_map = np.max(has_fuel * z_levels[np.newaxis, np.newaxis, :], axis=2)
    
    # 2. Geometry: Terrain + Canopy
    # Transpose for Plotly (y, x) expectation
    z_data = (terrain + fuel_height_map).T
    
    # Calculate Coordinate Grids
    ny_grid, nx_grid = z_data.shape
    x_coords = np.arange(nx_grid) * dx + x_offset
    y_coords = np.arange(ny_grid) * dy
    
    # 3. Color Calculation
    fuel_load = np.sum(fuel_3d, axis=2).T
    fire_intensity = np.max(rr_3d, axis=2).T
    
    # Composite Color Map
    norm_fuel = np.clip(fuel_load / 2.0, 0, 1)
    color_values = norm_fuel * 0.65  # Base vegetation range
    
    # Active fire override
    mask_fire = fire_intensity > 0.001
    color_values[mask_fire] = 0.8 + np.clip(fire_intensity[mask_fire]*10, 0, 0.2)
    
    # Custom Colorscale
    colorscale = [
        [0.0, 'rgb(30, 30, 30)'],    # Charcoal (Burnt/No Fuel)
        [0.1, 'rgb(101, 67, 33)'],   # Dark Earth
        [0.2, 'rgb(34, 139, 34)'],   # Forest Green
        [0.7, 'rgb(124, 252, 0)'],   # Lawn Green
        [0.8, 'rgb(255, 69, 0)'],    # Orange Red (Fire)
        [1.0, 'rgb(255, 255, 0)']    # Yellow (Intense)
    ]
    
    trace = go.Surface(
        z=z_data,
        x=x_coords,
        y=y_coords,
        surfacecolor=color_values,
        colorscale=colorscale,
        cmin=0, cmax=1.0,
        showscale=False,
        lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.8, specular=0.1)
    )
    return trace

def render_3d_single(terrain, fuel_3d, rr_3d, dx, dy, dz):
    """
    Renders a single 3D view.
    """
    trace = create_surface_trace(terrain, fuel_3d, rr_3d, dx, dy, dz, x_offset=0)
    fig = go.Figure(data=[trace])
    
    fig.update_layout(
        autosize=True,
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=True, title='Elev (m)'),
            aspectmode='data',
            bgcolor='#0e1117'
        ),
        paper_bgcolor='#0e1117',
    )
    return fig

def render_3d_comparison_unified(terrain, fuel_init, rr_init, fuel_final, rr_final, dx, dy, dz):
    """
    Renders two 3D views side-by-side in the SAME scene.
    This synchronizes camera controls perfectly.
    """
    # Calculate offset
    nx = terrain.shape[0]
    total_width_m = nx * dx
    gap_m = total_width_m * 0.2 # 20% gap
    offset_x = total_width_m + gap_m
    
    # Create traces
    trace1 = create_surface_trace(terrain, fuel_init, rr_init, dx, dy, dz, x_offset=0)
    trace2 = create_surface_trace(terrain, fuel_final, rr_final, dx, dy, dz, x_offset=offset_x)
    
    fig = go.Figure(data=[trace1, trace2])
    
    # Add Text Annotations to the scene
    # We place them above the center of each terrain block
    center_y = (terrain.shape[1] * dy) / 2
    center_x1 = total_width_m / 2
    center_x2 = offset_x + (total_width_m / 2)
    max_z = np.max(terrain) * 1.5 + 20 # Float above terrain
    
    annotations = [
        dict(
            showarrow=False,
            x=center_x1, y=center_y, z=max_z,
            text="Initial State",
            font=dict(color="white", size=16),
            xanchor="center", yanchor="middle"
        ),
        dict(
            showarrow=False,
            x=center_x2, y=center_y, z=max_z,
            text="Final Burn Scar",
            font=dict(color="white", size=16),
            xanchor="center", yanchor="middle"
        )
    ]
    
    fig.update_layout(
        autosize=True,
        height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=True, title='Elev (m)'),
            aspectmode='data',
            bgcolor='#0e1117',
            annotations=annotations
        ),
        paper_bgcolor='#0e1117',
    )
    
    return fig

def main():
    st.sidebar.title("🔥 Py-Fire Live")
    
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
    
    # --- Dual View Columns ---
    # Col 1: 2D Image, Col 2: 3D Interactive
    view_c1, view_c2 = st.columns([1, 1])
    
    with view_c1:
        st.markdown("**Top-Down View**")
        image_display = st.empty()
        
    with view_c2:
        st.markdown("**3D View**")
        plotly_display = st.empty()
    
    if st.session_state.running:
        try:
            # 1. Load Data (Using existing QuicFireIO)
            status_text.info("Loading inputs...")
            
            # =========================================================================
            # 1. PARSE GRID CONFIGURATION
            # =========================================================================
            simparams_path = os.path.join(input_dir, "QU_simparams.inp")
            if not os.path.exists(simparams_path):
                st.error("Missing QU_simparams.inp")
                return 1
            
            sim_params = QuicFireIO.read_simparams(simparams_path)
            
            fire_params_path = os.path.join(input_dir, "QUIC_fire.inp")
            if os.path.exists(fire_params_path):
                fire_params = QuicFireIO.read_quic_fire_inp(fire_params_path)
            else:
                fire_params = {'sim_time': 300, 'dt': 1, 'out_int_fire': 10, 'nz_fire': 32}
            
            # Get georeferencing
            origin_x, origin_y = QuicFireIO.read_raster_origin(input_dir)
            
            # Combine configuration
            qf_config = {**sim_params, **fire_params}
            qf_config['origin_x'] = origin_x
            qf_config['origin_y'] = origin_y
            
            if 'nz_fire' in qf_config:
                qf_config['nz'] = qf_config['nz_fire']
            
            nx, ny, nz = qf_config['nx'], qf_config['ny'], qf_config['nz']
            dx = qf_config.get('dx', 2.0)
            dy = qf_config.get('dy', 2.0)
            dz = qf_config.get('dz', 1.0)
            
            # =========================================================================
            # 2. READ WIND SCHEDULE
            # =========================================================================
            sensor_path = os.path.join(input_dir, "sensor1.inp")
            wind_schedule = QuicFireIO.read_sensor1(sensor_path)
            if not wind_schedule:
                wind_schedule = [(0, 5.0, 225.0)]

            # =========================================================================
            # 3. READ FUEL DATA
            # =========================================================================
            fuel_path = os.path.join(input_dir, "treesrhof.dat")
            file_format = qf_config.get('fuel_file_format', 1)
            
            if os.path.exists(fuel_path):
                fuel_density = QuicFireIO.read_fuel_dat(fuel_path, nx, ny, nz, file_format)
            else:
                fuel_density = np.zeros((nx, ny, nz), dtype=np.float32)
                fuel_density[:, :, 0:3] = 0.5
            
            moist_path = os.path.join(input_dir, "treesmoist.dat")
            default_moisture = 0.05
            if os.path.exists(moist_path):
                fuel_moisture = QuicFireIO.read_fuel_dat(moist_path, nx, ny, nz, file_format)
            else:
                fuel_moisture = None

            # =========================================================================
            # 4. READ TERRAIN
            # =========================================================================
            topo_config = QuicFireIO.read_topo_inputs(
                os.path.join(input_dir, "QU_TopoInputs.inp")
            )
            topo_flag = topo_config.get('topo_flag', 0)
            terrain_meters = np.zeros((nx, ny), dtype=np.float32)
            
            if topo_flag == 5:
                fname = topo_config.get('filename', 'usgs_dem.dat')
                topo_path = os.path.join(input_dir, fname)
                if os.path.exists(topo_path):
                    terrain_meters = QuicFireIO.read_topo_dat(topo_path, nx, ny)
            
            # Indices for sim, meters for viz
            terrain_indices = terrain_meters / dz

            # =========================================================================
            # 5. READ IGNITION DATA
            # =========================================================================
            ignite_path = os.path.join(input_dir, "ignite.dat")
            ignition_data = QuicFireIO.read_ignite_dat(ignite_path)
            
            if ignition_data and ignition_data.get('type') == 5:
                ig_payload = ignition_data
            else:
                try:
                    ig_payload = setup_default_ignition(nx, ny, nz)
                except:
                    ig_payload = {'type': 1, 'x': nx//2, 'y': ny//2}
            
            # --- Prepare Payload ---
            sim_payload = {
                'wind_schedule': wind_schedule,
                'moisture': default_moisture,
                'ignition': ig_payload,
                'custom_fuel': fuel_density,
                'custom_terrain': terrain_indices,
                'qf_config': qf_config
            }
            
            # --- Render Static 3D View (Initial State) ---
            # Shows initial fuel structure
            initial_rr = np.zeros_like(fuel_density)
            fig_initial = render_3d_single(terrain_meters, fuel_density, initial_rr, dx, dy, dz)
            plotly_display.plotly_chart(fig_initial, width=600)

            # --- Run Iterator ---
            status_text.text("Simulation Running...")
            
            sim_gen = qf_realtime_runner.simulation_iterator(sim_payload)
            initial_fuel_total = np.sum(fuel_density)
            if initial_fuel_total == 0: initial_fuel_total = 1.0

            # Store final state
            final_fuel = None
            final_rr = None
            
            for step_data in sim_gen:
                if not st.session_state.running:
                    status_text.warning("Stopped by user.")
                    break
                
                step = step_data['step']
                final_fuel = step_data['fuel']
                final_rr = step_data['reaction_rate']
                
                # Update visuals every N steps
                if step % speed_factor == 0:
                    t = step_data['time']
                    ws = step_data['wind_speed']
                    wd = step_data['wind_dir']
                    
                    rr = step_data['reaction_rate']
                    active_cells = np.sum(rr > 0.0001)
                    curr_fuel_total = np.sum(step_data['fuel'])
                    consumed = (1 - (curr_fuel_total/initial_fuel_total)) * 100
                    
                    p_time.markdown(f"<div class='metric-card'><div class='metric-value'>{t:.1f}s</div><div class='metric-label'>Sim Time</div></div>", unsafe_allow_html=True)
                    p_wind.markdown(f"<div class='metric-card'><div class='metric-value'>{ws:.1f} m/s</div><div class='metric-label'>Wind {wd:.0f}°</div></div>", unsafe_allow_html=True)
                    p_fuel.markdown(f"<div class='metric-card'><div class='metric-value'>{consumed:.1f}%</div><div class='metric-label'>Fuel Burned</div></div>", unsafe_allow_html=True)
                    p_fire.markdown(f"<div class='metric-card'><div class='metric-value'>{active_cells}</div><div class='metric-label'>Active Cells</div></div>", unsafe_allow_html=True)
                    
                    img = render_frame(step_data['fuel'], step_data['reaction_rate'])
                    image_display.image(img, caption=f"Top-Down View (Step {step})", clamp=True, width=600)
            
            # --- Final 3D Comparison ---
            if final_fuel is not None:
                status_text.success("Simulation Complete. Showing Synchronized Comparison...")
                
                # Use unified rendering (same scene, offset geometry) to lock controls
                fig_compare = render_3d_comparison_unified(
                    terrain_meters, 
                    fuel_density, initial_rr, 
                    final_fuel, final_rr, 
                    dx, dy, dz
                )
                
                plotly_display.plotly_chart(fig_compare, width=600, key="3d_end_compare_unified")
            else:
                status_text.success("Simulation Complete (No steps run).")
            
        except Exception as e:
            st.error(f"Simulation Error: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()