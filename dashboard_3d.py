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
RR_SCALE = 1000.0 # must match dataset scaling

st.set_page_config(page_title="3D Fire Emulator vs Simulator", layout="wide", page_icon="🔥")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetFireEmulator3D(in_channels=5, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    except Exception as e:
        st.error(f"Failed to load model checkpoint: {e}")
        st.stop()
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

def canvas_to_mask(img_data, target_shape):
    if img_data is not None and isinstance(img_data, np.ndarray):
        alpha = img_data[:, :, 3] 
        mask = alpha.astype(np.float32) / 255.0
        img = Image.fromarray(mask)
        img = img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
        return np.array(img)
    return np.zeros(target_shape, dtype=np.float32)

def render_frame_to_array(fuel_vol, fire_vol, step):
    fire_2d = np.max(fire_vol, axis=0)
    fuel_2d = np.mean(fuel_vol, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(fuel_2d, cmap='Greens', vmin=0, vmax=1, alpha=0.5)
    fire_masked = np.ma.masked_where(fire_2d < 0.05, fire_2d)
    ax.imshow(fire_masked, cmap='hot', vmin=0, vmax=1.0, alpha=0.9)
    ax.set_title(f"Step {step}")
    ax.axis('off')
    
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    image_flat = np.frombuffer(rgba_buffer, dtype='uint8')
    image_rgba = image_flat.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_rgb = image_rgba[:, :, :3]
    plt.close(fig)
    return image_rgb

if 'layer_history' not in st.session_state: st.session_state['layer_history'] = {}
if 'generated_data' not in st.session_state: st.session_state['generated_data'] = None

st.sidebar.title("🔥 Controls")
input_mode = st.sidebar.radio("Input Mode", ["Procedural Generator", "Manual Painting"])
st.sidebar.divider()
nx = st.sidebar.number_input("Grid X", 100, 256, 128)
ny = st.sidebar.number_input("Grid Y", 100, 256, 128)
nz = st.sidebar.number_input("Grid Z", 5, 64, 32)
st.sidebar.divider()

if input_mode == "Manual Painting":
    wind_speed = st.sidebar.slider("Wind Speed", 0.0, 30.0, 10.0)
    wind_dir = st.sidebar.slider("Wind Dir", 0, 360, 45)
    moisture = st.sidebar.slider("Moisture", 0.1, 1.5, 0.5)
else:
    st.sidebar.markdown("### Generated Conditions")
    if st.session_state['generated_data']:
        p = st.session_state['generated_data']['params']
        st.sidebar.metric("Wind Speed", f"{p['wind_speed']:.1f} m/s")
        st.sidebar.metric("Wind Dir", f"{p['wind_dir']:.0f}°")
        st.sidebar.metric("Moisture", f"{p['moisture']:.2f}")
    else:
        st.sidebar.info("Generate a world to see params.")
steps = st.sidebar.slider("Simulation Steps", 10, 200, 50)

st.title("Interactive 3D Fire Emulator")

if input_mode == "Procedural Generator":
    st.subheader("🌍 World Generator")
    if st.button("Generate Random Scenario"):
        with st.spinner("Generating World (Terrain + Fuel)..."):
            fuel_grid, terrain_grid = world_gen.generate_world(nx, ny, nz)
            speed = np.random.uniform(0.0, 25.0)
            direction = np.random.uniform(0.0, 360.0)
            moisture = np.random.uniform(0.1, 1.5)
            
            ig_x, ig_y, ig_z = 0, 0, 0
            for _ in range(20):
                rx = np.random.randint(nx * 0.1, nx * 0.9)
                ry = np.random.randint(ny * 0.1, ny * 0.9)
                ground_z = int(terrain_grid[rx, ry])
                if ground_z < nz and fuel_grid[ground_z, rx, ry] > 0:
                    ig_x, ig_y, ig_z = rx, ry, ground_z
                    break
            else:
                ig_x, ig_y = nx // 2, ny // 2
                ig_z = int(terrain_grid[ig_x, ig_y])
                
            rr_grid = np.zeros_like(fuel_grid)
            if 0 <= ig_z < nz: rr_grid[ig_z, ig_x, ig_y] = 1.0
                
            st.session_state['generated_data'] = {
                'fuel': fuel_grid,
                'rr': rr_grid,
                'params': {'wind_speed': speed, 'wind_dir': direction, 'moisture': moisture}
            }
            st.success("New world generated!")

    if st.session_state['generated_data']:
        data = st.session_state['generated_data']
        fuel = data['fuel']
        rr = data['rr']
        col1, col2 = st.columns(2)
        with col1:
            fuel_proj = np.max(fuel, axis=0)
            fig_fuel = plt.figure(figsize=(4, 4))
            plt.imshow(fuel_proj, cmap="Greens", vmin=0, vmax=2.0)
            plt.axis('off')
            st.pyplot(fig_fuel)
        with col2:
            fig_ign = plt.figure(figsize=(4, 4))
            plt.imshow(fuel_proj, cmap="Greens", vmin=0, vmax=2.0, alpha=0.5)
            z_idx, x_idx, y_idx = np.where(rr > 0)
            if len(x_idx) > 0: plt.scatter(y_idx, x_idx, c='red', s=100, marker='x')
            plt.axis('off')
            st.pyplot(fig_ign)

elif input_mode == "Manual Painting":
    current_layer = st.slider("Select Layer Slice (Z-Axis)", 0, nz-1, 0)
    DISPLAY_SIZE = 400
    pixel_scale = DISPLAY_SIZE / max(nx, ny)
    brush_cells = st.slider("Brush Size", 1, 10, 2)
    stroke_w = brush_cells * pixel_scale
    col1, col2 = st.columns(2)
    
    if current_layer not in st.session_state['layer_history']:
        st.session_state['layer_history'][current_layer] = {'fuel': None, 'ign': None}
    layer_data = st.session_state['layer_history'][current_layer]

    with col1:
        st.subheader(f"Paint Fuel (Layer {current_layer})")
        fuel_canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.5)",
            stroke_width=stroke_w,
            stroke_color="rgba(0, 100, 0, 1.0)",
            background_color="#eee",
            height=DISPLAY_SIZE, width=DISPLAY_SIZE,
            drawing_mode="freedraw",
            initial_drawing=layer_data['fuel']['json'] if layer_data['fuel'] else None,
            key=f"fc_{current_layer}",
            update_streamlit=True
        )
    with col2:
        st.subheader(f"Set Ignition (Layer {current_layer})")
        bg_img = None
        if fuel_canvas.image_data is not None:
            bg_img = Image.fromarray(fuel_canvas.image_data.astype('uint8'))
        elif layer_data['fuel']:
            bg_img = Image.fromarray(layer_data['fuel']['img'].astype('uint8'))
        ign_canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 1.0)",
            stroke_width=stroke_w,
            stroke_color="rgba(255, 0, 0, 1.0)",
            background_color="#eee",
            background_image=bg_img,
            height=DISPLAY_SIZE, width=DISPLAY_SIZE,
            drawing_mode="freedraw",
            initial_drawing=layer_data['ign']['json'] if layer_data['ign'] else None,
            key=f"ic_{current_layer}",
            update_streamlit=True
        )

    if fuel_canvas.image_data is not None:
        st.session_state['layer_history'][current_layer]['fuel'] = {'json': fuel_canvas.json_data, 'img': fuel_canvas.image_data}
    if ign_canvas.image_data is not None:
        st.session_state['layer_history'][current_layer]['ign'] = {'json': ign_canvas.json_data, 'img': ign_canvas.image_data}

st.divider()
if st.button("Run Simulation & Generate Video", type="primary"):
    fuel_vol = np.zeros((nz, nx, ny), dtype=np.float32)
    rr_vol = np.zeros((nz, nx, ny), dtype=np.float32)
    
    if input_mode == "Procedural Generator":
        if st.session_state['generated_data'] is None:
            st.error("Please generate a scenario first!")
            st.stop()
        data = st.session_state['generated_data']
        fuel_vol = data['fuel']
        rr_vol = data['rr']
        p_speed = data['params']['wind_speed']
        p_dir = data['params']['wind_dir']
        p_moist = data['params']['moisture']
    else: 
        found_ign = False
        with st.spinner("Compiling Manual Canvas Layers..."):
            for z in range(nz):
                if z in st.session_state['layer_history']:
                    ld = st.session_state['layer_history'][z]
                    if ld['fuel']: fuel_vol[z] = canvas_to_mask(ld['fuel']['img'], (nx, ny))
                    if ld['ign']: 
                        mask = canvas_to_mask(ld['ign']['img'], (nx, ny))
                        if np.sum(mask) > 0:
                            rr_vol[z] = mask
                            found_ign = True
        if not found_ign and np.sum(rr_vol) == 0:
            st.error("No ignition painted!")
            st.stop()
        p_speed, p_dir, p_moist = wind_speed, wind_dir, moisture

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    
    w_rad = np.radians(p_dir)
    wx, wy = np.cos(w_rad)*(p_speed/30.0), np.sin(w_rad)*(p_speed/30.0)
    
    curr_rr = torch.from_numpy(rr_vol).float().unsqueeze(0).to(device)
    
    # Scale RR UP for the model (so it sees "big" numbers)
    curr_rr = curr_rr * RR_SCALE
    
    curr_fuel = torch.from_numpy(fuel_vol).float().unsqueeze(0).to(device)
    t_wx = torch.full((1, nz, nx, ny), wx, device=device).float()
    t_wy = torch.full((1, nz, nx, ny), wy, device=device).float()
    t_mst = torch.full((1, nz, nx, ny), p_moist, device=device).float()

    progress_bar = st.progress(0)
    # Save unscaled history for viz
    history_rr = [(curr_rr.cpu().numpy()[0] / RR_SCALE)] 
    video_frames = []
    video_frames.append(render_frame_to_array(fuel_vol, rr_vol, 0))

    with torch.no_grad():
        for t in range(steps):
            inputs = torch.stack([curr_fuel[0], curr_rr[0], t_wx[0], t_wy[0], t_mst[0]], dim=0).unsqueeze(0)
            
            # Prediction is DELTA * SCALE
            pred = model(inputs).squeeze(1)
            
            # Apply delta
            curr_rr = torch.clamp(curr_rr + pred, 0, 1.0 * RR_SCALE)
            
            # Convert back to 0.0-1.0 range for visualization
            np_rr = curr_rr.cpu().numpy()[0] / RR_SCALE
            history_rr.append(np_rr)
            video_frames.append(render_frame_to_array(fuel_vol, np_rr, t+1))
            
            progress_bar.progress((t+1)/steps)

    with st.spinner("Encoding MP4..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_array = np.stack(video_frames, axis=0)
        iio.imwrite(tfile.name, video_array, fps=10)
        st.session_state['video_path'] = tfile.name
        st.session_state['history_rr'] = history_rr
        st.session_state['run_complete'] = True
        st.session_state['run_dims'] = (nx, ny, nz)

if st.session_state.get('run_complete'):
    st.divider()
    st.subheader("Results")
    col_vid, col_dl = st.columns([3, 1])
    with col_vid: st.video(st.session_state['video_path'])
    with col_dl: 
        with open(st.session_state['video_path'], 'rb') as v: st.download_button("Download Video", v, "fire.mp4")
    st.divider()
    st.markdown("#### 3D Volume Explorer")
    history_rr = st.session_state['history_rr']
    vnx, vny, vnz = st.session_state['run_dims']
    time_step = st.slider("Time Step View", 0, len(history_rr)-1, len(history_rr)-1)
    frame_data = history_rr[time_step]
    step_size = 2 if vnx >= 100 else 1
    Z_grid, X_grid, Y_grid = np.mgrid[0:vnz:1, 0:vnx:step_size, 0:vny:step_size]
    vol_data = frame_data[::1, ::step_size, ::step_size]
    fig = go.Figure(data=go.Volume(x=X_grid.flatten(), y=Y_grid.flatten(), z=Z_grid.flatten(), value=vol_data.flatten(), isomin=0.05, isomax=1.0, opacity=0.15, surface_count=10, colorscale='Hot', caps=dict(x_show=False, y_show=False, z_show=False)))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.4)), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)