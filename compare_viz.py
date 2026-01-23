import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image
import tempfile
import imageio.v3 as iio
import matplotlib.pyplot as plt
import os

# Import modules
from model import UNetFireEmulator3D 
import config
import world_gen
import run_gpu # Import the actual physics engine

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas-fix")
    st.stop()

# --- CONSTANTS ---
RR_SCALE = 1000.0 

st.set_page_config(page_title="3D Fire Emulator vs Simulator", layout="wide", page_icon="🔥")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetFireEmulator3D(in_channels=5, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None
    model.eval()
    return model

def render_comparison_frame(fuel_vol, emu_fire, sim_fire, step):
    """
    Renders side-by-side comparison (Emulator vs Simulator)
    """
    # Max Projection
    fuel_2d = np.mean(fuel_vol, axis=0)
    emu_2d = np.max(emu_fire, axis=0)
    sim_2d = np.max(sim_fire, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    
    # EMULATOR (Left)
    ax1.imshow(fuel_2d, cmap='Greens', vmin=0, vmax=1, alpha=0.5)
    emu_masked = np.ma.masked_where(emu_2d < 0.05, emu_2d)
    ax1.imshow(emu_masked, cmap='hot', vmin=0, vmax=1.0, alpha=0.9)
    ax1.set_title(f"Emulator (AI) - Step {step}")
    ax1.axis('off')

    # SIMULATOR (Right)
    ax2.imshow(fuel_2d, cmap='Greens', vmin=0, vmax=1, alpha=0.5)
    sim_masked = np.ma.masked_where(sim_2d < 0.05, sim_2d)
    ax2.imshow(sim_masked, cmap='hot', vmin=0, vmax=1.0, alpha=0.9)
    ax2.set_title(f"Simulator (Physics) - Step {step}")
    ax2.axis('off')
    
    # Convert to RGB
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_flat = np.frombuffer(rgba_buffer, dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_rgb = image_rgba[:, :, :3]
    plt.close(fig)
    return image_rgb

# --- STATE ---
if 'generated_data' not in st.session_state: st.session_state['generated_data'] = None

# --- SIDEBAR ---
st.sidebar.title("🔥 Controls")
nx = st.sidebar.number_input("Grid X", 100, 256, 128)
ny = st.sidebar.number_input("Grid Y", 100, 256, 128)
nz = st.sidebar.number_input("Grid Z", 5, 64, 32)
steps = st.sidebar.slider("Steps", 10, 200, 50)

st.title("Fire Emulator vs Physics Simulator")

# --- GENERATOR ---
st.subheader("1. Generate Scenario")
if st.button("Generate Random World"):
    with st.spinner("Generating..."):
        fuel_grid, terrain_grid = world_gen.generate_world(nx, ny, nz)
        speed = np.random.uniform(5.0, 25.0) 
        direction = np.random.uniform(0.0, 360.0)
        moisture = np.random.uniform(0.1, 0.8) 
        
        # Smart Ignition
        ig_x, ig_y, ig_z = nx//2, ny//2, 0
        for _ in range(50):
            rx = np.random.randint(nx * 0.2, nx * 0.8)
            ry = np.random.randint(ny * 0.2, ny * 0.8)
            ground_z = int(terrain_grid[rx, ry])
            if ground_z < nz and fuel_grid[ground_z, rx, ry] > 0.5:
                ig_x, ig_y, ig_z = rx, ry, ground_z
                break
        
        ignition_list = [{'x': int(ig_x), 'y': int(ig_y), 'z': int(ig_z)}]
        
        rr_grid = np.zeros_like(fuel_grid)
        if 0 <= ig_z < nz: rr_grid[ig_z, ig_x, ig_y] = 1.0

        st.session_state['generated_data'] = {
            'fuel': fuel_grid,
            'terrain': terrain_grid,
            'rr': rr_grid,
            'ignition_points': ignition_list,
            'params': {'wind_speed': speed, 'wind_dir': direction, 'moisture': moisture}
        }
        st.success("World Generated.")

if st.session_state['generated_data']:
    p = st.session_state['generated_data']['params']
    st.info(f"Wind: {p['wind_speed']:.1f} m/s @ {p['wind_dir']:.0f}° | Moisture: {p['moisture']:.2f}")

# --- RUNNER ---
st.divider()
if st.button("Run Comparison", type="primary"):
    if not st.session_state['generated_data']:
        st.error("Generate a world first.")
        st.stop()
        
    data = st.session_state['generated_data']
    fuel_vol = data['fuel']
    
    # ---------------------------
    # 1. RUN SIMULATOR (Synchronized)
    # ---------------------------
    with st.spinner("Running Physics Simulator..."):
        sim_params = {
            'wind_speed': data['params']['wind_speed'],
            'wind_dir': data['params']['wind_dir'],
            'moisture': data['params']['moisture'],
            'ignition': data['ignition_points'],
            'custom_fuel': data['fuel'],
            'custom_terrain': data['terrain']
        }
        
        # Override config temporarily to match dashboard STEPS exactly
        old_total_time = config.TOTAL_TIME
        old_save_interval = getattr(config, 'SAVE_INTERVAL', 1)
        
        # Force exact synchronization
        config.TOTAL_TIME = float(steps * config.DT) 
        config.SAVE_INTERVAL = 1 # Save every frame for smooth video
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            run_gpu.run_simulation(sim_params, run_id=999, output_dir=tmpdirname)
            sim_result = np.load(os.path.join(tmpdirname, "run_999.npz"))
            # Transpose to (Time, Z, X, Y)
            sim_rr_history = sim_result['reaction_rate'].transpose(0, 3, 1, 2)
        
        # Restore Config
        config.TOTAL_TIME = old_total_time
        config.SAVE_INTERVAL = old_save_interval

    # ---------------------------
    # 2. RUN EMULATOR
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    
    p = data['params']
    w_rad = np.radians(p['wind_dir'])
    wx = np.cos(w_rad)*(p['wind_speed']/30.0)
    wy = np.sin(w_rad)*(p['wind_speed']/30.0)
    
    curr_rr = torch.from_numpy(data['rr']).float().unsqueeze(0).to(device) * RR_SCALE
    curr_fuel = torch.from_numpy(data['fuel']).float().unsqueeze(0).to(device)
    t_wx = torch.full((1, nz, nx, ny), wx, device=device).float()
    t_wy = torch.full((1, nz, nx, ny), wy, device=device).float()
    t_mst = torch.full((1, nz, nx, ny), p['moisture'], device=device).float()

    emu_rr_history = []
    video_frames = []
    
    progress_bar = st.progress(0)
    
    # Ensure we don't crash if sim produced fewer frames
    max_frames = min(steps, len(sim_rr_history))
    
    with torch.no_grad():
        for t in range(max_frames):
            # EMULATOR STEP
            inputs = torch.stack([curr_fuel[0], curr_rr[0], t_wx[0], t_wy[0], t_mst[0]], dim=0).unsqueeze(0)
            pred = model(inputs).squeeze(1)
            curr_rr = torch.clamp(curr_rr + pred, 0, 1.0 * RR_SCALE)
            
            emu_frame = curr_rr.cpu().numpy()[0] / RR_SCALE
            sim_frame = sim_rr_history[t] # Perfectly synced index
            
            video_frames.append(render_comparison_frame(fuel_vol, emu_frame, sim_frame, t))
            progress_bar.progress((t+1)/max_frames)

    # 3. Save Video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    iio.imwrite(tfile.name, np.stack(video_frames, axis=0), fps=10)
    st.video(tfile.name)