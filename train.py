import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
from models.vrt_model import VRT
from utils.data_utils import create_dataloader
from utils.metrics import evaluate_metrics
from utils.visualization import visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description='Train Video Restoration Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for resuming')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(config):
    model_type = config['model']['type']
    
    if model_type == 'vrt':
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
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    return model

def build_loss_fn(config):
    loss_fns = {}
    
    # L1 loss
    if config['loss']['l1_weight'] > 0:
        loss_fns['l1'] = {
            'weight': config['loss']['l1_weight'],
            'fn': nn.L1Loss()
        }
    
    # Perceptual loss (VGG)
    if config['loss']['perceptual_weight'] > 0:
        try:
            from torchvision.models import vgg19, VGG19_Weights
            from torch.nn import functional as F
            
            class VGGPerceptualLoss(nn.Module):
                def __init__(self, device='cuda'):
                    super(VGGPerceptualLoss, self).__init__()
                    if not torch.cuda.is_available() and device == 'cuda':
                        device = 'cpu'
                        print('CUDA not available, using CPU for VGG perceptual loss')
                    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
                    self.vgg_layers = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27]])
                    for param in self.parameters():
                        param.requires_grad = False
                    self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                    self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                
                def forward(self, x, y):
                    # Normalize to ImageNet stats
                    x = (x + 1) / 2  # [-1, 1] -> [0, 1]
                    y = (y + 1) / 2
                    x = (x - self.mean) / self.std
                    y = (y - self.mean) / self.std
                    
                    # Extract features
                    loss = 0.0
                    for layer in self.vgg_layers:
                        x = layer(x)
                        y = layer(y)
                        loss += F.l1_loss(x, y)
                    
                    return loss
            
            loss_fns['perceptual'] = {
                'weight': config['loss']['perceptual_weight'],
                'fn': VGGPerceptualLoss(device=config['general']['device'])
            }
        except ImportError:
            print("Warning: Could not import VGG for perceptual loss. Skipping.")
    
    # Adversarial loss (GAN)
    if config['loss']['adversarial_weight'] > 0:
        # This would require implementing a discriminator network
        # For simplicity, we'll skip this for now
        print("Warning: Adversarial loss not implemented. Skipping.")
    
    return loss_fns

def train_epoch(model, train_loader, optimizer, loss_fns, device, epoch, writer):
    model.train()
    total_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Get data
            degraded = batch['degraded'].to(device)  # [B, T, C, H, W]
            clean = batch['clean'].to(device)  # [B, T, C, H, W]
            
            # Reshape to [B, C, T, H, W] for model
            degraded = degraded.permute(0, 2, 1, 3, 4)
            clean = clean.permute(0, 2, 1, 3, 4)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(degraded)
            
            # Calculate losses
            loss = 0.0
            loss_dict = {}
            
            for loss_name, loss_info in loss_fns.items():
                loss_fn = loss_info['fn']
                weight = loss_info['weight']
                
                if loss_name == 'perceptual':
                    # For perceptual loss, we need to handle each frame separately
                    curr_loss = 0.0
                    for t in range(clean.size(2)):
                        curr_loss += loss_fn(output[:, :, t], clean[:, :, t])
                    curr_loss /= clean.size(2)
                else:
                    curr_loss = loss_fn(output, clean)
                
                weighted_loss = weight * curr_loss
                loss += weighted_loss
                loss_dict[loss_name] = weighted_loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=avg_loss, **{k: v for k, v in loss_dict.items()})
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), global_step)
            for loss_name, loss_val in loss_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_val, global_step)
            
            # Visualize results occasionally
            if batch_idx % 100 == 0:
                # Convert back to [B, T, C, H, W] for visualization
                output_vis = output.permute(0, 2, 1, 3, 4)
                degraded_vis = degraded.permute(0, 2, 1, 3, 4)
                clean_vis = clean.permute(0, 2, 1, 3, 4)
                
                # Visualize middle frame
                t_idx = clean_vis.size(1) // 2
                visualize_results(
                    degraded_vis[:, t_idx],
                    output_vis[:, t_idx],
                    clean_vis[:, t_idx],
                    save_path=os.path.join(config['general']['log_dir'], f'train_vis_epoch{epoch}_batch{batch_idx}.png')
                )
                
                # Log images to TensorBoard
                writer.add_images('Train/Input', degraded_vis[:, t_idx], global_step)
                writer.add_images('Train/Output', output_vis[:, t_idx], global_step)
                writer.add_images('Train/Target', clean_vis[:, t_idx], global_step)
    
    return avg_loss

def validate(model, val_loader, loss_fns, device, epoch, writer):
    model.eval()
    total_loss = 0.0
    metrics_sum = {metric: 0.0 for metric in ['psnr', 'ssim']}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            # Get data
            degraded = batch['degraded'].to(device)  # [B, T, C, H, W]
            clean = batch['clean'].to(device)  # [B, T, C, H, W]
            
            # Reshape to [B, C, T, H, W] for model
            degraded = degraded.permute(0, 2, 1, 3, 4)
            clean = clean.permute(0, 2, 1, 3, 4)
            
            # Forward pass
            output = model(degraded)
            
            # Calculate losses
            loss = 0.0
            for loss_name, loss_info in loss_fns.items():
                loss_fn = loss_info['fn']
                weight = loss_info['weight']
                
                if loss_name == 'perceptual':
                    # For perceptual loss, we need to handle each frame separately
                    curr_loss = 0.0
                    for t in range(clean.size(2)):
                        curr_loss += loss_fn(output[:, :, t], clean[:, :, t])
                    curr_loss /= clean.size(2)
                else:
                    curr_loss = loss_fn(output, clean)
                
                loss += weight * curr_loss
            
            total_loss += loss.item()
            
            # Calculate metrics for each frame
            for t in range(clean.size(2)):
                # Convert to [B, C, H, W] for metrics calculation
                output_t = output[:, :, t]
                clean_t = clean[:, :, t]
                
                # Calculate metrics
                metrics = evaluate_metrics(output_t, clean_t, metrics=['psnr', 'ssim'])
                
                # Accumulate metrics
                for metric, value in metrics.items():
                    metrics_sum[metric] += value
    
    # Calculate average loss and metrics
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {metric: value / (len(val_loader) * clean.size(2)) for metric, value in metrics_sum.items()}
    
    # Log to TensorBoard
    writer.add_scalar('Loss/val', avg_loss, epoch)
    for metric, value in avg_metrics.items():
        writer.add_scalar(f'Metrics/{metric}', value, epoch)
    
    # Visualize validation results
    if val_loader.dataset:
        # Get a random batch
        batch = next(iter(val_loader))
        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)
        
        # Reshape to [B, C, T, H, W] for model
        degraded = degraded.permute(0, 2, 1, 3, 4)
        clean = clean.permute(0, 2, 1, 3, 4)
        
        # Forward pass
        with torch.no_grad():
            output = model(degraded)
        
        # Convert back to [B, T, C, H, W] for visualization
        output = output.permute(0, 2, 1, 3, 4)
        degraded = degraded.permute(0, 2, 1, 3, 4)
        clean = clean.permute(0, 2, 1, 3, 4)
        
        # Visualize middle frame
        t_idx = clean.size(1) // 2
        visualize_results(
            degraded[:, t_idx],
            output[:, t_idx],
            clean[:, t_idx],
            save_path=os.path.join(config['general']['log_dir'], f'val_vis_epoch{epoch}.png')
        )
        
        # Log images to TensorBoard
        writer.add_images('Val/Input', degraded[:, t_idx], epoch)
        writer.add_images('Val/Output', output[:, t_idx], epoch)
        writer.add_images('Val/Target', clean[:, t_idx], epoch)
    
    return avg_loss, avg_metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['general']['seed'])
    
    # Create directories
    os.makedirs(config['general']['save_dir'], exist_ok=True)
    os.makedirs(config['general']['log_dir'], exist_ok=True)
    
    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = create_dataloader(config, is_training=True)
    val_loader = create_dataloader(config, is_training=False)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Build optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Build scheduler
    if config['train']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['num_epochs']
        )
    elif config['train']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['train']['lr_step_size'],
            gamma=config['train']['lr_gamma']
        )
    elif config['train']['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train']['lr_gamma'],
            patience=config['train']['lr_patience'],
            verbose=True
        )
    else:
        scheduler = None
    
    # Build loss functions
    loss_fns = build_loss_fn(config)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=config['general']['log_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and args.checkpoint:
        start_epoch = model.load_checkpoint(args.checkpoint, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config['train']['num_epochs']):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_fns, device, epoch, writer)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, loss_fns, device, epoch, writer)
        
        # Update learning rate
        if scheduler is not None:
            if config['train']['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        model.save_checkpoint(
            config['general']['save_dir'],
            epoch + 1,
            optimizer,
            scheduler
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                config['general']['save_dir'],
                epoch + 1,
                optimizer,
                scheduler,
                best=True
            )
            print(f"New best model saved at epoch {epoch + 1}")
        
        # Print epoch summary
        print(f"Epoch {epoch + 1}/{config['train']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        for metric, value in val_metrics.items():
            print(f"Val {metric.upper()}: {value:.4f}")
        print("-" * 50)
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed!")

if __name__ == '__main__':
    main()
