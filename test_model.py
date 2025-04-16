import os
import yaml
import torch
from models.vrt_model import VRT
from utils.data_utils import create_dataloader

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load config
    config = load_config('config/small_config.yaml')
    
    # Set device
    device = torch.device('cpu')  # Use CPU for testing
    print(f"Using device: {device}")
    
    # Create data loaders
    try:
        print("Creating data loaders...")
        train_loader = create_dataloader(config, is_training=True)
        print(f"Train dataset size: {len(train_loader.dataset)}")
        
        # Get a sample batch
        print("Getting a sample batch...")
        sample_batch = next(iter(train_loader))
        print(f"Sample batch shapes:")
        print(f"  Degraded: {sample_batch['degraded'].shape}")
        print(f"  Clean: {sample_batch['clean'].shape}")
    except Exception as e:
        print(f"Error creating data loaders: {str(e)}")
        return
    
    # Build model
    try:
        print("Building model...")
        model_config = config['model']['vrt']
        model = VRT(
            img_size=(config['data']['sequence_length'], *config['data']['frame_size']),
            patch_size=(1, 4, 4),
            in_chans=3,
            out_chans=3,
            embed_dim=model_config['embed_dim'],
            depths=model_config['depths'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size'],
            mlp_ratio=model_config['mlp_ratio'],
            qkv_bias=model_config['qkv_bias'],
            qk_scale=model_config['qk_scale'],
            drop_rate=model_config['drop_rate'],
            attn_drop_rate=model_config['attn_drop_rate'],
            drop_path_rate=model_config['drop_path_rate']
        )
        model = model.to(device)
        print(f"Model parameters: {model.count_parameters():,}")
    except Exception as e:
        print(f"Error building model: {str(e)}")
        return
    
    # Test forward pass
    try:
        print("Testing forward pass...")
        # Reshape to [B, C, T, H, W] for model
        degraded = sample_batch['degraded'].permute(0, 2, 1, 3, 4).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(degraded)
        
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        return
    
    print("All tests passed!")

if __name__ == '__main__':
    main()
