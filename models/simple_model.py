import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class ConvBlock(nn.Module):
    """Basic convolutional block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Residual connection
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class SimpleVideoRestoration(BaseModel):
    """A simple CNN-based model for video restoration."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=32, num_blocks=4):
        super(SimpleVideoRestoration, self).__init__()

        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(num_blocks):
            self.encoder_blocks.append(ConvBlock(current_channels, current_channels * 2))
            current_channels *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(current_channels, current_channels)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(current_channels, current_channels // 2, kernel_size=2, stride=2),
                    ConvBlock(current_channels // 2, current_channels // 2)
                )
            )
            current_channels //= 2

        # Final convolution
        self.final_conv = nn.Conv3d(current_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input tensor with shape (B, C, T, H, W)

        Returns:
            Output tensor with shape (B, C, T, H, W)
        """
        # Initial convolution
        x = self.init_conv(x)

        # Store encoder features for skip connections
        encoder_features = []

        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
            encoder_features.append(x)
            x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            if i < len(encoder_features):
                # Add skip connection
                encoder_feature = encoder_features[-(i+1)]
                # Resize to match dimensions
                x = F.interpolate(x, size=(encoder_feature.size(2), encoder_feature.size(3), encoder_feature.size(4)),
                                 mode='trilinear', align_corners=False)
                x = x + encoder_feature

        # Final convolution
        x = self.final_conv(x)

        return x
