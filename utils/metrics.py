import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

def calculate_psnr(img1, img2, data_range=1.0):
    """Calculate PSNR between two images.
    
    Args:
        img1, img2: Images in range [0, 1] or [0, 255]
        data_range: The data range of the input image (1.0 or 255.0)
    
    Returns:
        PSNR value
    """
    # Convert torch tensor to numpy array
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:  # B, C, H, W
        psnr_values = []
        for i in range(img1.shape[0]):
            # Move channel dimension to the end for skimage function
            img1_i = np.transpose(img1[i], (1, 2, 0))
            img2_i = np.transpose(img2[i], (1, 2, 0))
            psnr_values.append(peak_signal_noise_ratio(img1_i, img2_i, data_range=data_range))
        return np.mean(psnr_values)
    else:  # C, H, W
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        return peak_signal_noise_ratio(img1, img2, data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0):
    """Calculate SSIM between two images.
    
    Args:
        img1, img2: Images in range [0, 1] or [0, 255]
        data_range: The data range of the input image (1.0 or 255.0)
    
    Returns:
        SSIM value
    """
    # Convert torch tensor to numpy array
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:  # B, C, H, W
        ssim_values = []
        for i in range(img1.shape[0]):
            # Move channel dimension to the end for skimage function
            img1_i = np.transpose(img1[i], (1, 2, 0))
            img2_i = np.transpose(img2[i], (1, 2, 0))
            ssim_values.append(structural_similarity(img1_i, img2_i, data_range=data_range, multichannel=True))
        return np.mean(ssim_values)
    else:  # C, H, W
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
        return structural_similarity(img1, img2, data_range=data_range, multichannel=True)

class LPIPSMetric:
    """LPIPS perceptual metric."""
    
    def __init__(self, net='alex', device='cuda'):
        """Initialize LPIPS metric.
        
        Args:
            net: Network for LPIPS ('alex', 'vgg', or 'squeeze')
            device: Device to run the network on
        """
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device
    
    def __call__(self, img1, img2):
        """Calculate LPIPS between two images.
        
        Args:
            img1, img2: Images in range [-1, 1] with shape [B, C, H, W]
        
        Returns:
            LPIPS value
        """
        # Ensure inputs are torch tensors
        if not torch.is_tensor(img1):
            img1 = torch.from_numpy(img1).to(self.device)
        if not torch.is_tensor(img2):
            img2 = torch.from_numpy(img2).to(self.device)
        
        # Ensure inputs are on the correct device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = self.loss_fn(img1, img2)
        
        # Return mean value if batch size > 1
        if lpips_value.ndim > 0:
            return lpips_value.mean().item()
        return lpips_value.item()

def evaluate_metrics(pred, target, metrics=['psnr', 'ssim']):
    """Evaluate multiple metrics between prediction and target.
    
    Args:
        pred: Predicted images, tensor of shape [B, C, H, W]
        target: Target images, tensor of shape [B, C, H, W]
        metrics: List of metrics to calculate
    
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    # Ensure inputs are in the correct range
    if pred.max() > 1.0 or target.max() > 1.0:
        pred = pred / 255.0
        target = target / 255.0
    
    for metric in metrics:
        if metric.lower() == 'psnr':
            results['psnr'] = calculate_psnr(pred, target)
        elif metric.lower() == 'ssim':
            results['ssim'] = calculate_ssim(pred, target)
        elif metric.lower() == 'lpips':
            # LPIPS expects inputs in range [-1, 1]
            pred_lpips = pred * 2 - 1
            target_lpips = target * 2 - 1
            lpips_fn = LPIPSMetric(device=pred.device)
            results['lpips'] = lpips_fn(pred_lpips, target_lpips)
    
    return results
