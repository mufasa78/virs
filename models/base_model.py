import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    """Base model class for all restoration models."""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save_checkpoint(self, save_path, epoch, optimizer=None, scheduler=None, best=False):
        """Save model checkpoint.
        
        Args:
            save_path: Directory to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state
            best: Whether this is the best checkpoint
        """
        os.makedirs(save_path, exist_ok=True)
        
        if best:
            checkpoint_path = os.path.join(save_path, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(save_path, f'model_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer to load state
            scheduler: Scheduler to load state
        
        Returns:
            epoch: Epoch of the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
        return epoch
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_train(self):
        """Set model to training mode."""
        self.train()
    
    def set_eval(self):
        """Set model to evaluation mode."""
        self.eval()
    
    def test(self, x):
        """Test mode forward pass."""
        self.set_eval()
        with torch.no_grad():
            return self.forward(x)
