import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import os
from dataset_parallel import FireEmulationDataset 
from model import UNetFireEmulator3D as UNetFireEmulator

# --- Hyperparameters ---
BATCH_SIZE = 24         
LEARNING_RATE = 1e-4
EPOCHS = 30             
DATA_DIR = "./training_data_test"
CHECKPOINT_DIR = "./checkpoints"
RR_SCALE = 1000.0 # Match global config
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class ContiguousGrowthLoss(nn.Module):
    def __init__(self, active_weight=10.0, growth_penalty=1.0, tv_weight=0.5):
        super().__init__()
        self.active_weight = active_weight
        self.growth_penalty = growth_penalty
        self.tv_weight = tv_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        
        active_mask = (torch.abs(target) > 0.05).float()
        underestimation_mask = (target > pred).float() * active_mask
        
        weight_map = 1.0 + (self.active_weight * active_mask)
        weight_map = weight_map + (self.active_weight * self.growth_penalty * underestimation_mask)
        
        mse_loss = (loss * weight_map).mean()
        
        # Total Variation Loss (Smoothness)
        diff_d = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]).mean()
        diff_h = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]).mean()
        diff_w = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]).mean()
        tv_loss = diff_d + diff_h + diff_w
        
        return mse_loss + (self.tv_weight * tv_loss)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    full_dataset = FireEmulationDataset(DATA_DIR, cache_in_ram=False)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = UNetFireEmulator(in_channels=7, out_channels=1).to(device)
    
    criterion = ContiguousGrowthLoss(active_weight=10.0, growth_penalty=1.0, tv_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # --- CRITICAL NEW PHYSICS FIX: Normalize Inputs ---
            # Inputs come in scaled by RR_SCALE (0-1000). 
            # We normalize to 0-1 so the UNet weights don't explode.
            inputs[:, 1:4] = inputs[:, 1:4] / RR_SCALE
            
            # Normalize Targets too (-1 to +1 range)
            targets = targets / RR_SCALE

            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
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
                
                # Apply same normalization in validation
                inputs[:, 1:4] = inputs[:, 1:4] / RR_SCALE
                targets = targets / RR_SCALE
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Results: Train Loss: {running_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("--> Best model saved.")

if __name__ == "__main__":
    train()