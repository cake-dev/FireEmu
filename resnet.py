import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConv3D(nn.Module):
    """
    Residual Block: x + Conv(Conv(x))
    Helps flow gradients in deep 3D networks.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.conv_block = nn.Sequential(
            nn.Conv3d(out_channels if self.projection else in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels)
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        
        if self.projection:
            residual = self.projection(x)
            x = residual # Start convolution with projected dimensions
            
        out = self.conv_block(x)
        out = out + residual # Skip connection
        return self.activation(out)

class UNetFireEmulator3D(nn.Module):
    def __init__(self, in_channels=8, out_channels=1): 
        super(UNetFireEmulator3D, self).__init__()
        
        # Encoder
        self.inc = ResidualConv3D(in_channels, 32)
        self.pool = nn.MaxPool3d(2)
        
        self.down1 = ResidualConv3D(32, 64)
        self.down2 = ResidualConv3D(64, 128)
        
        # Bottleneck
        self.bot = ResidualConv3D(128, 256)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = ResidualConv3D(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = ResidualConv3D(128 + 64, 64)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = ResidualConv3D(64 + 32, 32)
        
        # Output
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (Batch, 8, 32, 128, 128)
        
        x1 = self.inc(x)      # -> (32, 32, 128, 128)
        p1 = self.pool(x1)    # -> (32, 16, 64, 64)
        
        x2 = self.down1(p1)   # -> (64, 16, 64, 64)
        p2 = self.pool(x2)    # -> (64, 8, 32, 32)
        
        x3 = self.down2(p2)   # -> (128, 8, 32, 32)
        p3 = self.pool(x3)    # -> (128, 4, 16, 16)
        
        x4 = self.bot(p3)     # -> (256, 4, 16, 16)
        
        # Up 1
        x = self.up1(x4)      # -> (256, 8, 32, 32)
        x = torch.cat([x3, x], dim=1) # Concat with 128 -> 384
        x = self.conv1(x)     # -> 128
        
        # Up 2
        x = self.up2(x)       # -> (128, 16, 64, 64)
        x = torch.cat([x2, x], dim=1) # Concat with 64 -> 192
        x = self.conv2(x)     # -> 64
        
        # Up 3
        x = self.up3(x)       # -> (64, 32, 128, 128)
        x = torch.cat([x1, x], dim=1) # Concat with 32 -> 96
        x = self.conv3(x)     # -> 32
        
        logits = self.outc(x) # -> (1, 32, 128, 128)
        
        # CRITICAL: No ReLU here. 
        # Regression targets (Delta RR) can be negative (decay).
        return logits