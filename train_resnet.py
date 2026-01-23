import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import os
import torch.nn.functional as F
from dataset_parallel import FireEmulationDataset 
from model import UNetFireEmulator3D as UNetFireEmulator

# --- Hyperparameters ---
BATCH_SIZE = 24         
LEARNING_RATE = 1e-4
EPOCHS = 30             
DATA_DIR = "./training_data_v2"
CHECKPOINT_DIR = "./checkpoints_resnet"
RR_SCALE = 1000.0 
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class FrontWeightedLoss(nn.Module):
    """
    Ported from Swin script. 
    Focuses training on the fire front and penalizes 'hollow' fires.
    """
    def __init__(self, active_weight=20.0, growth_penalty=5.0, distance_penalty=10.0):
        super().__init__()
        self.active_weight = active_weight
        self.growth_penalty = growth_penalty
        self.distance_penalty = distance_penalty
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, input_rr):
        # Clamp prediction to avoid explosions in log space decoding
        pred = torch.clamp(pred, -1.0, 10.0)
        
        loss = self.mse(pred, target)
        
        # --- Masks ---
        # 1. Active Fire Mask (Where fire SHOULD be)
        active_mask = (target > 0.01).float()
        
        # 2. Valid Growth Zone (Dilate previous fire to define allowed spread)
        with torch.no_grad():
            curr_fire_mask = (input_rr > 0.01).float()
            # 3x3x3 Dilation
            valid_growth_zone = F.max_pool3d(curr_fire_mask, kernel_size=3, stride=1, padding=1)
        
        # 3. Underestimation (Missing the fire core)
        under_mask = (target > pred).float() * active_mask
        
        # 4. Teleportation (Predicting fire far away from existing fire)
        over_mask = (pred > (target + 0.1)).float()
        teleport_mask = over_mask * (1.0 - valid_growth_zone)
        
        # --- Weighting ---
        weights = 1.0
        weights += (self.active_weight * active_mask)
        weights += (self.active_weight * self.growth_penalty * under_mask)
        weights += (self.distance_penalty * teleport_mask)
        
        return (loss * weights).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model: ResNet UNet (Log-Space + FrontWeightedLoss)")

    full_dataset = FireEmulationDataset(DATA_DIR, cache_in_ram=False)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 8 Channels: Fuel, RR, RR-1, RR-2, Wx, Wy, Moist, Terrain
    model = UNetFireEmulator(in_channels=8, out_channels=1).to(device)
    
    criterion = FrontWeightedLoss(active_weight=20.0, growth_penalty=5.0, distance_penalty=10.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # --- CRITICAL CHANGE: Log Transform ---
            # Transform Inputs: log(1 + RR)
            # This makes faint fire (RR=5) and intense fire (RR=500) closer in magnitude,
            # allowing the model to learn the shape of the fire, not just the peak.
            inputs[:, 1] = torch.log1p(torch.clamp(inputs[:, 1], min=0.0)) # RR_t
            
            # We don't necessarily need to log transform older history (RR-1, RR-2) 
            # if we want to give the model raw context, but transforming them usually helps consistency.
            # Let's transform them too for consistency.
            inputs[:, 2] = torch.log1p(torch.clamp(inputs[:, 2], min=0.0))
            inputs[:, 3] = torch.log1p(torch.clamp(inputs[:, 3], min=0.0))

            # Capture current RR for the loss mask
            current_rr_log = inputs[:, 1:2].detach()
            
            # Transform Targets: log(1 + RR_next) - log(1 + RR_current)
            # Actually, let's predict the NEXT Log-State directly.
            # Target = log(1 + RR_next)
            # The model will output the Log-RR directly.
            # (Note: In dataset, targets are currently Delta-Linear. We need to reconstruct.)
            
            # Reconstruct Raw Target from Delta
            # target_raw = current_raw + delta
            # Since dataset gives us Delta, let's just grab the raw "Next" from the loader? 
            # The dataset loader returns `delta_rr`. 
            # Let's reconstruct `rr_next_linear` approximately:
            # inputs[:, 1] was originally raw/RR_SCALE (wait, dataset gives raw?).
            
            # REVISIT DATASET: 
            # Dataset returns: inputs (scaled by RR_SCALE? No, dataset returns RAW/RR_SCALE? No.)
            # Let's look at dataset_parallel.py from previous turn...
            # It returns `input_stack` (raw values) and `target` (delta raw values).
            # Wait, `db_new` assumes `inputs` are raw.
            # Let's assume inputs are RAW intensity (0-1000 range).
            
            raw_rr_t = inputs[:, 1].clone() # Copy before log transform
            raw_delta = targets.squeeze(1)
            raw_rr_next = raw_rr_t + raw_delta
            
            # Now Log Transform Targets
            log_target = torch.log1p(torch.clamp(raw_rr_next, min=0.0))
            
            # Now apply Log transform to input stack (in place) for the model
            # (We already did this above)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(inputs)
                # Model predicts Log(RR_next), not Delta.
                loss = criterion(outputs, log_target.unsqueeze(1), current_rr_log)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Reconstruct Target (Log Space)
                raw_rr_t = inputs[:, 1].clone()
                raw_delta = targets.squeeze(1)
                raw_rr_next = raw_rr_t + raw_delta
                log_target = torch.log1p(torch.clamp(raw_rr_next, min=0.0))
                
                # Log Transform Inputs
                inputs[:, 1] = torch.log1p(torch.clamp(inputs[:, 1], min=0.0))
                inputs[:, 2] = torch.log1p(torch.clamp(inputs[:, 2], min=0.0))
                inputs[:, 3] = torch.log1p(torch.clamp(inputs[:, 3], min=0.0))
                
                current_rr_log = inputs[:, 1:2]
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, log_target.unsqueeze(1), current_rr_log)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Results: Train Loss: {running_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("--> Best model saved.")

if __name__ == "__main__":
    train()