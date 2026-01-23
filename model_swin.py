import torch
import torch.nn as nn
import warnings

# IMPORT FROM LOCAL FILE
from monai_custom_swin import SwinUNETR

class SwinUNetFireEmulator(nn.Module):
    """
    3D Swin-UNet implementation for Fire Emulation.
    Wraps the custom SwinUNETR implementation.
    """
    def __init__(
        self,
        in_channels: int = 8, # Updated to 8 (Fuel, RR, RR-1, RR-2, Wx, Wy, Moist, Terrain)
        out_channels: int = 1,
        img_size: tuple = (32, 128, 128), # (Depth, Height, Width) match config.NZ, NX, NY
        flatten_temporal_dimension: bool = False,
        encoder_weights: str = None
    ):
        super().__init__()
        self.flatten_temporal_dimension = flatten_temporal_dimension
        
        # --- Config Class ---
        class Config:
            class DATA:
                IMG_SIZE = img_size
            
            class MODEL:
                DROP_RATE = 0.0
                DROP_PATH_RATE = 0.3 # Increase drop path slightly for deeper models
                
                class SWIN:
                    PATCH_SIZE = (2, 2, 2) 
                    IN_CHANS = in_channels
                    EMBED_DIM = 48 
                    # INCREASE DEPTH
                    # Old: (2, 2, 2, 2) -> 8 layers total
                    # New: (2, 4, 4, 2) -> 12 layers total. 
                    # The middle layers are crucial for spatial mixing.
                    DEPTHS = (2, 4, 4, 2) 
                    NUM_HEADS = (3, 6, 12, 24)
                    WINDOW_SIZE = (7, 7, 7)
        
        self.config = Config()
        
        # Initialize SwinUNETR using the custom class signature
        self.swin_unet = SwinUNETR(
            in_channels=self.config.MODEL.SWIN.IN_CHANS,
            out_channels=out_channels,
            feature_size=self.config.MODEL.SWIN.EMBED_DIM,
            use_checkpoint=False, 
            spatial_dims=3,
            depths=self.config.MODEL.SWIN.DEPTHS,
            num_heads=self.config.MODEL.SWIN.NUM_HEADS,
            drop_rate=self.config.MODEL.DROP_RATE,
            norm_name="instance",
            use_v2=True 
        )

    def forward(self, x: torch.Tensor, doys=None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor. Can be (B, T, C, D, H, W) or (B, C, D, H, W).
        """
        # Handle (B, T, C, D, H, W)
        if x.ndim == 6: 
            x_in = x[:, -1, :, :, :, :]
        elif x.ndim == 5: 
            x_in = x
        else:
            raise ValueError(f"SwinUNetFireEmulator expects 5D or 6D input, got {x.shape}")

        logits = self.swin_unet(x_in)
        return logits

    def load_from(self, config):
        pass