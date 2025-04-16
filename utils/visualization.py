import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import cv2

def tensor_to_numpy(tensor):
    """Convert a torch tensor to numpy array."""
    if tensor.ndim == 4:  # Batch of images
        return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:  # Single image
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize a tensor from [-1, 1] to [0, 1]."""
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)
    
    # Clone the tensor to avoid modifying the original
    tensor = tensor.clone()
    
    # Denormalize
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp to [0, 1]
    tensor.clamp_(0, 1)
    
    return tensor

def visualize_results(input_tensor, output_tensor, target_tensor, save_path=None, max_samples=4):
    """Visualize input, output, and target images.
    
    Args:
        input_tensor: Input degraded images, tensor of shape [B, C, H, W]
        output_tensor: Output restored images, tensor of shape [B, C, H, W]
        target_tensor: Target ground truth images, tensor of shape [B, C, H, W]
        save_path: Path to save the visualization
        max_samples: Maximum number of samples to visualize
    """
    # Limit the number of samples
    batch_size = min(input_tensor.size(0), max_samples)
    
    # Denormalize tensors
    input_tensor = denormalize(input_tensor[:batch_size])
    output_tensor = denormalize(output_tensor[:batch_size])
    target_tensor = denormalize(target_tensor[:batch_size])
    
    # Create a grid of images
    grid_input = make_grid(input_tensor, nrow=batch_size, padding=2, normalize=False)
    grid_output = make_grid(output_tensor, nrow=batch_size, padding=2, normalize=False)
    grid_target = make_grid(target_tensor, nrow=batch_size, padding=2, normalize=False)
    
    # Convert to numpy for visualization
    grid_input = tensor_to_numpy(grid_input)
    grid_output = tensor_to_numpy(grid_output)
    grid_target = tensor_to_numpy(grid_target)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot input images
    plt.subplot(3, 1, 1)
    plt.imshow(grid_input)
    plt.title('Input (Degraded)')
    plt.axis('off')
    
    # Plot output images
    plt.subplot(3, 1, 2)
    plt.imshow(grid_output)
    plt.title('Output (Restored)')
    plt.axis('off')
    
    # Plot target images
    plt.subplot(3, 1, 3)
    plt.imshow(grid_target)
    plt.title('Target (Ground Truth)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_sequence(sequence_tensor, title=None, save_path=None):
    """Visualize a sequence of frames.
    
    Args:
        sequence_tensor: Sequence of frames, tensor of shape [T, C, H, W]
        title: Title for the visualization
        save_path: Path to save the visualization
    """
    # Denormalize tensor
    sequence_tensor = denormalize(sequence_tensor)
    
    # Create a grid of images
    grid = make_grid(sequence_tensor, nrow=sequence_tensor.size(0), padding=2, normalize=False)
    
    # Convert to numpy for visualization
    grid = tensor_to_numpy(grid)
    
    # Create figure
    plt.figure(figsize=(15, 3))
    
    # Plot sequence
    plt.imshow(grid)
    if title:
        plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_comparison_video(input_video_path, output_video_path, save_path, fps=30):
    """Create a side-by-side comparison video.
    
    Args:
        input_video_path: Path to input degraded video
        output_video_path: Path to output restored video
        save_path: Path to save the comparison video
        fps: Frames per second
    """
    # Open input and output videos
    cap_input = cv2.VideoCapture(input_video_path)
    cap_output = cv2.VideoCapture(output_video_path)
    
    # Check if videos opened successfully
    if not cap_input.isOpened() or not cap_output.isOpened():
        print("Error: Could not open videos.")
        return
    
    # Get video properties
    width_input = int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_input = int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_output = int(cap_output.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_output = int(cap_output.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width_input + width_output, max(height_input, height_output)))
    
    while True:
        # Read frames
        ret_input, frame_input = cap_input.read()
        ret_output, frame_output = cap_output.read()
        
        # Break if either video ends
        if not ret_input or not ret_output:
            break
        
        # Resize output frame if dimensions don't match
        if height_input != height_output or width_input != width_output:
            frame_output = cv2.resize(frame_output, (width_input, height_input))
        
        # Create side-by-side comparison
        comparison = np.hstack((frame_input, frame_output))
        
        # Add labels
        cv2.putText(comparison, 'Input (Degraded)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Output (Restored)', (width_input + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame
        out.write(comparison)
    
    # Release resources
    cap_input.release()
    cap_output.release()
    out.release()
    
    print(f"Comparison video saved to {save_path}")
