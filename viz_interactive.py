import streamlit as st
import numpy as np
import math
from numba import cuda
import config
import wind_gpu
import fire_gpu
import gpu_utils
import world_gen
import scipy.ndimage
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Interactive Fire Sim")

# --- SIMULATION ENGINE CLASS ---
class FireSimEngine:
    def __init__(self, nx, ny, nz, dx, dy, dz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.shape = (nx, ny, nz)
        self.grid_shape = (nx, ny)
        
        # Physics State
        self.t = 0
        self.dt = config.DT
        
        # 1. Generate World (Host)
        fuel_raw, terrain_raw = world_gen.generate_world(nx, ny, nz)
        self.fuel_host = np.ascontiguousarray(fuel_raw.transpose(1, 2, 0)) # (NX, NY, NZ)
        self.terrain_host = np.ascontiguousarray(terrain_raw, dtype=np.float32)
        
        # Apply physics smoothing to terrain (Slope Effect fix)
        self.terrain_physics = scipy.ndimage.gaussian_filter(self.terrain_host, sigma=1.5)
        
        self.z_coords = np.arange(nz) * dz
        
        # 2. Allocate GPU Memory
        # Use a try-except to catch context errors if Streamlit switches threads
        try:
            self.stream = cuda.stream()
            with self.stream.auto_synchronize():
                self.elevation_dev = cuda.to_device(self.terrain_physics)
                self.z_coords_dev = cuda.to_device(self.z_coords)
                self.fuel_0_dev = cuda.to_device(self.fuel_host)
                self.fuel_dev = cuda.to_device(self.fuel_host)
                
                # Physics Fields
                self.u_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.v_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.w_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.rr_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.time_since_ign_dev = cuda.device_array(self.shape, dtype=np.float32)
                
                # Lagrangian / EP Fields
                self.cx_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.cy_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.cz_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.ep_hist_dev = cuda.device_array(self.shape, dtype=np.int32)
                
                # Initialize Centroids
                tpb = (8, 8, 8)
                bpg = ((nx + 7)//8, (ny + 7)//8, (nz + 7)//8)
                gpu_utils.init_centroid_kernel[bpg, tpb](self.cx_dev, self.cy_dev, self.cz_dev)
                gpu_utils.zero_array_3d[bpg, tpb](self.ep_hist_dev)
                
                # Flux Buffers
                self.inc_x_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.inc_y_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.inc_z_dev = cuda.device_array(self.shape, dtype=np.float32)
                self.n_ep_recv_dev = cuda.device_array(self.shape, dtype=np.int32)
                self.ep_counts_dev = cuda.device_array(self.shape, dtype=np.int32)
                
                # Moisture (Dynamic)
                self.moisture_dev = cuda.to_device(np.ones(self.shape, dtype=np.float32) * 0.5)
                
                # RNG
                self.rng_states = gpu_utils.init_rng(nx * ny * nz, seed=1234)
                
                # Zero out initial state
                gpu_utils.zero_array_3d[bpg, tpb](self.rr_dev)
                gpu_utils.zero_array_3d[bpg, tpb](self.time_since_ign_dev)
                gpu_utils.zero_array_3d[bpg, tpb](self.n_ep_recv_dev)
        except cuda.CudaSupportError:
            st.error("CUDA Error: Context lost. Please reset the world.")

    def add_ignition(self, x, y, radius=1):
        """Adds fire at a specific location on the CPU then pushes to GPU."""
        # We need to find the ground Z at this X,Y
        z_idx = int(self.terrain_host[x, y])
        z_idx = max(0, min(z_idx, self.nz - 1))
        
        current_eps = self.n_ep_recv_dev.copy_to_host()
        
        # Add 5000 EPs (Ignition)
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                xi, yj = x+i, y+j
                if 0 <= xi < self.nx and 0 <= yj < self.ny:
                    z_g = int(self.terrain_host[xi, yj])
                    if 0 <= z_g < self.nz:
                        current_eps[xi, yj, z_g] += 5000
        
        self.n_ep_recv_dev = cuda.to_device(current_eps)

    def update_environment(self, wind_speed, wind_dir, moisture):
        """Updates global parameters."""
        self.wind_speed = wind_speed
        self.wind_rad = np.radians(270 - wind_dir)
        m_arr = np.ones(self.shape, dtype=np.float32) * moisture
        self.moisture_dev = cuda.to_device(m_arr)

    def step(self):
        """Advances the simulation by one DT."""
        tpb = (8, 8, 8)
        bpg = ((self.nx + 7)//8, (self.ny + 7)//8, (self.nz + 7)//8)
        tpb_2d = (8, 8)
        bpg_2d = ((self.nx + 7)//8, (self.ny + 7)//8)

        # 1. Wind
        wind_gpu.apply_drag_kernel[bpg, tpb](
            self.u_dev, self.fuel_dev, self.fuel_0_dev, self.z_coords_dev, 
            self.wind_speed, 10.0, config.K_VON_KARMAN, config.Z0, config.DZ
        )
        wind_gpu.rotate_wind_kernel[bpg, tpb](self.u_dev, self.v_dev, self.wind_rad)
        wind_gpu.reset_w_kernel[bpg, tpb](self.w_dev)
        wind_gpu.project_wind_over_terrain_kernel[bpg, tpb](
            self.u_dev, self.v_dev, self.w_dev, self.elevation_dev, self.dx, self.dy
        )
        wind_gpu.apply_buoyancy_column_kernel[bpg_2d, tpb_2d](
            self.w_dev, self.rr_dev, self.dx, self.dy, self.dz, 
            config.G, config.RHO_AIR, config.CP_AIR, config.T_AMBIENT, config.H_WOOD
        )

        # 2. Reaction
        fire_gpu.compute_reaction_and_fuel_kernel[bpg, tpb](
            self.fuel_dev, self.moisture_dev,
            self.n_ep_recv_dev, self.inc_x_dev, self.inc_y_dev, self.inc_z_dev,
            self.cx_dev, self.cy_dev, self.cz_dev, self.ep_hist_dev,
            self.time_since_ign_dev, self.rr_dev, self.ep_counts_dev,
            self.dt, config.CM, config.T_BURNOUT, config.H_WOOD, 
            (self.dx*self.dy*self.dz), config.C_RAD_LOSS, config.EEP,
            config.CP_WOOD, config.T_CRIT, config.T_AMBIENT
        )

        # Clear Flux Buffers
        gpu_utils.zero_array_3d[bpg, tpb](self.n_ep_recv_dev)
        gpu_utils.zero_array_3d[bpg, tpb](self.inc_x_dev)
        gpu_utils.zero_array_3d[bpg, tpb](self.inc_y_dev)
        gpu_utils.zero_array_3d[bpg, tpb](self.inc_z_dev)
        cuda.synchronize()

        # 3. Transport
        fire_gpu.transport_eps_kernel[bpg, tpb](
            self.ep_counts_dev,
            self.n_ep_recv_dev, self.inc_x_dev, self.inc_y_dev, self.inc_z_dev,
            self.cx_dev, self.cy_dev, self.cz_dev,
            self.u_dev, self.v_dev, self.w_dev, self.elevation_dev,
            self.rng_states, self.dx, self.dy, self.dz, self.dt, config.SLOPE_FACTOR, config.JUMP_HACK, config.MOD_DT
        )
        
        self.t += 1

    def get_visualization_data(self):
        """Downloads current state to CPU for rendering."""
        fuel = self.fuel_dev.copy_to_host()
        rr = self.rr_dev.copy_to_host()
        fuel_top = np.max(fuel, axis=2)
        fire_top = np.max(rr, axis=2)
        return fuel_top, fire_top, self.terrain_host

# --- STREAMLIT UI ---

# 1. Initialize Engine State
if 'engine' not in st.session_state:
    st.session_state.engine = FireSimEngine(128, 128, 32, 2.0, 2.0, 1.0)
    st.session_state.running = False
    st.session_state.step_count = 0

engine = st.session_state.engine

# 2. Sidebar Controls
st.sidebar.header("Controls")

col1, col2 = st.sidebar.columns(2)
if col1.button("Start/Resume"):
    st.session_state.running = True
if col2.button("Pause"):
    st.session_state.running = False

if st.sidebar.button("Reset World"):
    st.session_state.engine = FireSimEngine(128, 128, 32, 2.0, 2.0, 1.0)
    st.session_state.step_count = 0
    st.session_state.running = False
    st.rerun()

st.sidebar.markdown("---")
view_width = st.sidebar.slider("Map View Size (px)", 300, 1200, 700, step=50)
sim_speed = st.sidebar.slider("Simulation Speed (Steps/Frame)", 1, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Environment")
wind_spd = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 10.0)
wind_dir = st.sidebar.slider("Wind Direction (deg)", 0.0, 360.0, 90.0)
moisture = st.sidebar.slider("Fuel Moisture", 0.05, 1.0, 0.1)

engine.update_environment(wind_spd, wind_dir, moisture)

st.sidebar.markdown("---")
st.sidebar.subheader("Tactical Actions")
action_mode = st.sidebar.radio("Action", ["Ignite", "Firebreak"])
act_x = st.sidebar.slider("X Coord", 0, engine.nx-1, engine.nx//2)
act_y = st.sidebar.slider("Y Coord", 0, engine.ny-1, engine.ny//2)
act_rad = st.sidebar.slider("Radius", 1, 10, 2)

if st.sidebar.button("Apply Action"):
    if action_mode == "Ignite":
        engine.add_ignition(act_x, act_y, radius=act_rad)
    elif action_mode == "Firebreak":
        f_host = engine.fuel_dev.copy_to_host()
        f_host[max(0, act_x-act_rad):min(engine.nx, act_x+act_rad), 
               max(0, act_y-act_rad):min(engine.ny, act_y+act_rad), :] = 0.0
        engine.fuel_dev = cuda.to_device(f_host)

# 3. Main Visualization Area
st.title("Interactive Fire Simulator")

c1, c2, c3 = st.columns([1, 10, 1])
with c2:
    map_placeholder = st.empty()
stats_placeholder = st.empty()

def render_frame():
    fuel, fire, terrain = engine.get_visualization_data()
    
    fuel_norm = np.clip(fuel / 2.0, 0, 1)
    fire_norm = np.clip(fire / 1.0, 0, 1) 
    terr_norm = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-6)
    
    rgb = np.zeros((engine.nx, engine.ny, 3), dtype=np.float32)
    rgb[:, :, 1] = fuel_norm * (0.5 + 0.5 * terr_norm)
    
    fire_mask = fire_norm > 0.05
    rgb[fire_mask, 0] = 1.0 
    rgb[fire_mask, 1] = np.clip(rgb[fire_mask, 1] + 0.5 * fire_norm[fire_mask], 0, 1)
    rgb[fire_mask, 2] = 0.0
    
    cx, cy = act_x, act_y
    r = 2
    rgb[max(0, cx-r):min(engine.nx, cx+r), max(0, cy-r):min(engine.ny, cy+r), 2] = 1.0 
    
    rgb = np.flipud(np.rot90(rgb))
    map_placeholder.image(rgb, caption=f"Step: {st.session_state.step_count}", width=view_width)
    stats_placeholder.markdown(f"**Status:** {'RUNNING' if st.session_state.running else 'PAUSED'} | **Steps:** {st.session_state.step_count}")

# 4. Hybrid Loop Strategy
if st.session_state.running:
    # Run a "Batch" of frames to get smooth animation
    FRAMES_PER_BATCH = 10
    
    for _ in range(FRAMES_PER_BATCH):
        # Physics Steps per Frame
        for _ in range(sim_speed):
            engine.step()
            st.session_state.step_count += 1
        
        # Render
        render_frame()
        
        # Throttle to ~20 FPS so browser doesn't freeze
        time.sleep(0.1)
    
    # Force a rerun to sync sliders/buttons and clear memory
    # This causes a momentary "flash" but keeps the app responsive and stable
    st.rerun()
else:
    render_frame()