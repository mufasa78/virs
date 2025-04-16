import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from models.basic_model import BasicVideoRestoration
from utils.data_utils import create_dataloader

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
    # Use the basic model for training
    model = BasicVideoRestoration(
        in_channels=3,
        out_channels=3,
        hidden_channels=64  # Use a moderate number of channels
    )

    return model

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
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

            # Calculate loss
            loss = criterion(output, clean)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=avg_loss)

    return avg_loss

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0

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

            # Calculate loss
            loss = criterion(output, clean)
            total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss

def main():
    # Load config
    config = load_config('config/small_config.yaml')

    # Set seed for reproducibility
    set_seed(config['general']['seed'])

    # Create directories
    os.makedirs(config['general']['save_dir'], exist_ok=True)
    os.makedirs(config['general']['log_dir'], exist_ok=True)

    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(config, is_training=True)
    val_loader = create_dataloader(config, is_training=False)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    # Build model
    print("Building model...")
    model = build_model(config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # Training loop
    num_epochs = 3  # Train for 3 epochs since we're on CPU
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # Save checkpoint
        model.save_checkpoint(
            config['general']['save_dir'],
            epoch + 1,
            optimizer
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                config['general']['save_dir'],
                epoch + 1,
                optimizer,
                best=True
            )
            print(f"New best model saved at epoch {epoch + 1}")

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print("-" * 50)

    print("Training completed!")

if __name__ == '__main__':
    main()
