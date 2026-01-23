import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm

class FireEmulationDataset(Dataset):
    def __init__(self, data_dir, cache_in_ram=True):
        self.files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.cache_in_ram = cache_in_ram
        self.data_cache = []

        if self.cache_in_ram:
            print(f"Pre-loading {len(self.files)} files into RAM...")
            for f in tqdm(self.files):
                self.data_cache.append(self._load_file(f))
        
    def _load_file(self, filepath):
        """Helper to load and process a single file"""
        with np.load(filepath) as data:
            # Inputs: (Time, X, Y, Z) -> Transpose to (Time, Z, X, Y)
            fuel_map = data['fuel'].transpose(0, 3, 1, 2).astype(np.float32)
            rr_map = data['reaction_rate'].transpose(0, 3, 1, 2).astype(np.float32)
            
            # Global scalars
            w_speed = data['wind_speed'][0] / 30.0
            w_dir = np.radians(data['wind_dir'][0])
            wx = np.cos(w_dir) * w_speed
            wy = np.sin(w_dir) * w_speed
            moist = data['moisture'][0]
            
            return {
                'fuel': fuel_map,
                'rr': rr_map,
                'wx': wx,
                'wy': wy,
                'moist': moist
            }

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.cache_in_ram:
            data = self.data_cache[idx]
        else:
            data = self._load_file(self.files[idx])

        fuel_map = data['fuel']
        rr_map = data['rr']
        
        # Select random time step t (predict t+1)
        # fuel_map shape is (Time, Z, X, Y)
        max_t = fuel_map.shape[0] - 2 # Ensure t+1 exists
        if max_t < 0:
            # Fallback for very short simulations
            t = 0
        else:
            t = np.random.randint(0, max_t + 1)
        
        # Dimensions
        d, h, w = fuel_map.shape[1], fuel_map.shape[2], fuel_map.shape[3]
        
        # Broadcast scalars to 3D volumes
        wx_vol = np.full((d, h, w), data['wx'], dtype=np.float32)
        wy_vol = np.full((d, h, w), data['wy'], dtype=np.float32)
        mst_vol = np.full((d, h, w), data['moist'], dtype=np.float32)
        
        # Stack inputs (5 channels)
        input_stack = np.stack([
            fuel_map[t], 
            rr_map[t], 
            wx_vol, 
            wy_vol, 
            mst_vol
        ], axis=0)
        
        # Target: Reaction Rate at t+1
        delta_rr = rr_map[t+1] - rr_map[t]
        target = delta_rr[None, :, :, :] 

        return torch.from_numpy(input_stack), torch.from_numpy(target)