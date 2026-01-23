import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(Conv3d => GroupNorm => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetFireEmulator3D(nn.Module):
    def __init__(self, in_channels=7, out_channels=1): 
        super(UNetFireEmulator3D, self).__init__()
        
        # Encoder
        self.inc = DoubleConv3D(in_channels, 32)
        self.down1 = DoubleConv3D(32, 64)
        self.down2 = DoubleConv3D(64, 128)
        
        # Bottleneck
        self.bot = DoubleConv3D(128, 256)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = DoubleConv3D(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = DoubleConv3D(128 + 64, 64)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = DoubleConv3D(64 + 32, 32)
        
        self.outc = nn.Conv3d(32, out_channels, kernel_size=1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x1 = self.inc(x)
        p1 = self.pool(x1)
        
        x2 = self.down1(p1)
        p2 = self.pool(x2)
        
        x3 = self.down2(p2)
        p3 = self.pool(x3)
        
        x4 = self.bot(p3)
        
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        
        # CRITICAL FIX: Removed F.relu(logits). 
        # Regression targets (Delta RR) can be negative (decay). 
        # ReLU was forcing outputs to be >= 0, causing "Dead ReLU" on initialization 
        # and preventing the model from predicting fire dying down.
        return logits