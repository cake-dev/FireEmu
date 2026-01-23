import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Placeholder for your actual trained model import
from model import UNetFireEmulator 

st.set_page_config(page_title="Fire Emulator V1", layout="wide")

st.title("🔥 AI Fire Emulator (Jetstream2)")

# --- Sidebar: Controls ---
st.sidebar.header("Environment Conditions")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 10.0)
wind_dir = st.sidebar.slider("Wind Direction (deg)", 0, 360, 45)
moisture = st.sidebar.slider("Fuel Moisture", 0.1, 1.5, 0.5)

st.sidebar.header("Simulation Control")
steps = st.sidebar.slider("Time Steps to Emulate", 10, 200, 50)
run_btn = st.sidebar.button("Run Emulator")

# --- Helper Functions ---
def get_model():
    # Load trained model (cached)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetFireEmulator(in_channels=5, out_channels=1).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    return model

def init_state(nx=100, ny=100):
    # Initialize fuel and a center ignition
    fuel = np.ones((nx, ny), dtype=np.float32)
    rr = np.zeros((nx, ny), dtype=np.float32)
    
    # Ignition in center
    cx, cy = nx//2, ny//2
    rr[cx-2:cx+2, cy-2:cy+2] = 1.0 # Active fire
    return fuel, rr

# --- Main Viz Logic ---
col1, col2 = st.columns([1, 1])

if run_btn:
    model = get_model()
    fuel, rr = init_state()
    
    # Pre-calculate wind vectors
    w_rad = np.radians(wind_dir)
    wx = np.cos(w_rad) * (wind_speed / 30.0)
    wy = np.sin(w_rad) * (wind_speed / 30.0)
    
    # Setup Plot
    chart_placeholder = st.empty()
    
    # Inference Loop
    history = []
    
    for t in range(steps):
        # 1. Prepare Input Tensor
        # [Fuel, RR, Wx, Wy, Moisture]
        h, w = fuel.shape
        wx_c = np.full((h, w), wx)
        wy_c = np.full((h, w), wy)
        mst_c = np.full((h, w), moisture)
        
        inputs = np.stack([fuel, rr, wx_c, wy_c, mst_c], axis=0)
        tensor_in = torch.from_numpy(inputs).float().unsqueeze(0) # Batch dim
        
        # 2. Run Model Inference
        with torch.no_grad():
           pred_rr = model(tensor_in.cuda()).cpu().numpy()[0, 0]
        
        # MOCK INFERENCE for visualization demo without weights
        # Simple diffusion to show how it looks
        # pred_rr = rr * 0.9 + np.random.normal(0, 0.1, rr.shape)
        # pred_rr = np.clip(pred_rr, 0, 1)
        
        # 3. Update State (Simple physics integration step)
        # In a real emulator, the model might predict the *new* state directly, 
        # or the *derivative* (dRR/dt).
        rr = pred_rr 
        fuel = fuel - (rr * 0.1) # Burn fuel based on RR
        fuel = np.clip(fuel, 0, 1)
        
        # 4. Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Fuel
        ax1.set_title("Fuel Load")
        im1 = ax1.imshow(fuel, cmap="Greens_r", vmin=0, vmax=1)
        
        # Plot Fire Intensity
        ax2.set_title("Fire Intensity (AI Prediction)")
        im2 = ax2.imshow(rr, cmap="inferno", vmin=0, vmax=1)
        
        # Update Streamlit container
        chart_placeholder.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.05) # Control framerate