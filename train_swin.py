import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import os
import config
from dataset_parallel import FireEmulationDataset 
from model_swin import SwinUNetFireEmulator
import torch.nn.functional as F

# --- Hyperparameters ---
BATCH_SIZE = 4          
LEARNING_RATE = 2e-5    
EPOCHS = 30             
DATA_DIR = "./training_data_v2"
CHECKPOINT_DIR = "./checkpoints_swin_2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class HeightWeightedResidualLoss(nn.Module):
    def __init__(self, growth_weight=10.0, height_scale=5.0, nz=32):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.growth_weight = growth_weight
        
        # Create a Z-Weight vector: [1.0, 1.1, 1.2 ... 5.0]
        # This makes the loss 5x stronger at the top of the domain.
        # Shape: (1, 1, D, 1, 1) for broadcasting
        # NOTE: We register this as a buffer so it moves with .to(device)
        z_weights = torch.linspace(1.0, height_scale, steps=nz)
        self.register_buffer('z_weights', z_weights.view(1, 1, nz, 1, 1))

    def forward(self, pred_delta, target_delta):
        """
        pred_delta: Model output (Log space change)
        target_delta: Actual change (Log(Next) - Log(Curr))
        """
        loss = self.mse(pred_delta, target_delta)
        
        # 1. Growth Weighting (Prioritize spread over decay)
        growth_mask = (target_delta > 0.01).float()
        growth_w = 1.0 + (self.growth_weight * growth_mask)
        
        # 2. Height Weighting (Prioritize canopy fire)
        # Apply z_weights broadcasting across batch, height, width
        
        total_weights = growth_w * self.z_weights
        
        return (loss * total_weights).mean()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model: SwinUNetFireEmulator (Height-Weighted Residual)")

    full_dataset = FireEmulationDataset(DATA_DIR, cache_in_ram=False)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 8 Channels (Fuel, RR, RR-1, RR-2, Wx, Wy, Moist, Terrain)
    model = SwinUNetFireEmulator(
        in_channels=8, 
        out_channels=1,
        img_size=(config.NZ, config.NX, config.NY) 
    ).to(device)
    
    # Updated Loss - MOVED TO DEVICE
    criterion = HeightWeightedResidualLoss(growth_weight=5.0, height_scale=5.0, nz=config.NZ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # 1. Log Transform Inputs
            inputs[:, 1] = torch.log1p(torch.clamp(inputs[:, 1], min=0.0))
            inputs[:, 2] = torch.log1p(torch.clamp(inputs[:, 2], min=0.0))
            inputs[:, 3] = torch.log1p(torch.clamp(inputs[:, 3], min=0.0))
            
            # 2. Prepare Target (RESIDUAL)
            log_rr_t = inputs[:, 1].clone()
            raw_rr_t = torch.expm1(log_rr_t)
            raw_delta = targets.squeeze(1) 
            
            raw_rr_next = raw_rr_t + raw_delta
            log_rr_next = torch.log1p(torch.clamp(raw_rr_next, min=0.0))
            
            # Target is the CHANGE in log space
            target_log_delta = log_rr_next - log_rr_t

            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                pred_log_delta = model(inputs) 
                loss = criterion(pred_log_delta, target_log_delta.unsqueeze(1))
            
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
                
                inputs[:, 1] = torch.log1p(torch.clamp(inputs[:, 1], min=0.0))
                inputs[:, 2] = torch.log1p(torch.clamp(inputs[:, 2], min=0.0))
                inputs[:, 3] = torch.log1p(torch.clamp(inputs[:, 3], min=0.0))
                
                log_rr_t = inputs[:, 1].clone()
                raw_rr_t = torch.expm1(log_rr_t)
                raw_delta = targets.squeeze(1)
                raw_rr_next = raw_rr_t + raw_delta
                log_rr_next = torch.log1p(torch.clamp(raw_rr_next, min=0.0))
                target_log_delta = log_rr_next - log_rr_t
                
                with autocast('cuda'):
                    pred_log_delta = model(inputs)
                    loss = criterion(pred_log_delta, target_log_delta.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Results: Train Loss: {running_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_swin_model.pth"))
            print("--> Best model saved.")

if __name__ == "__main__":
    train()