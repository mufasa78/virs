import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class BasicVideoRestoration(BaseModel):
    """A very basic CNN model for video restoration."""
    
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64):
        super(BasicVideoRestoration, self).__init__()
        
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.middle = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x: Input tensor with shape (B, C, T, H, W)
        
        Returns:
            Output tensor with shape (B, C, T, H, W)
        """
        # Encoder
        features = self.encoder(x)
        
        # Middle
        features = self.middle(features)
        
        # Decoder
        output = self.decoder(features)
        
        return output
