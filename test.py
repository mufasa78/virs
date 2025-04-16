import os
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.vrt_model import VRT
from utils.data_utils import VideoFramesDataset
from utils.metrics import evaluate_metrics
from utils.visualization import visualize_results, create_comparison_video

def parse_args():
    parser = argparse.ArgumentParser(description='Test Video Restoration Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing test videos')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--save_frames', action='store_true', help='Save individual frames')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model(config, checkpoint_path, device):
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
    
    # Load checkpoint
    model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    return model

def test(model, test_loader, device, output_dir, save_frames=False, save_video=False, visualize=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    metrics_sum = {metric: 0.0 for metric in ['psnr', 'ssim']}
    metrics_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Get data
            degraded = batch['degraded'].to(device)  # [B, T, C, H, W]
            clean = batch['clean'].to(device) if 'clean' in batch else None  # [B, T, C, H, W]
            video_name = batch['video_name'][0]  # Assuming batch size 1
            frame_names = batch['frame_names']  # List of frame names
            
            # Create output directory for this video
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Reshape to [B, C, T, H, W] for model
            degraded = degraded.permute(0, 2, 1, 3, 4)
            if clean is not None:
                clean = clean.permute(0, 2, 1, 3, 4)
            
            # Forward pass
            output = model(degraded)
            
            # Convert back to [B, T, C, H, W] for saving/visualization
            output = output.permute(0, 2, 1, 3, 4)
            degraded = degraded.permute(0, 2, 1, 3, 4)
            if clean is not None:
                clean = clean.permute(0, 2, 1, 3, 4)
            
            # Calculate metrics if ground truth is available
            if clean is not None:
                for t in range(clean.size(1)):
                    # Convert to [B, C, H, W] for metrics calculation
                    output_t = output[:, t]
                    clean_t = clean[:, t]
                    
                    # Calculate metrics
                    metrics = evaluate_metrics(output_t, clean_t, metrics=['psnr', 'ssim'])
                    
                    # Accumulate metrics
                    for metric, value in metrics.items():
                        metrics_sum[metric] += value
                    metrics_count += 1
            
            # Save frames if requested
            if save_frames:
                frames_dir = os.path.join(video_output_dir, 'frames')
                os.makedirs(frames_dir, exist_ok=True)
                
                # Denormalize and convert to numpy
                output_np = ((output[0] + 1) / 2).clamp(0, 1).cpu().numpy()
                output_np = (output_np * 255).astype(np.uint8)
                
                # Save each frame
                for t, frame_name in enumerate(frame_names[0]):
                    frame = output_np[t].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                    frame = frame[:, :, ::-1]  # RGB -> BGR for OpenCV
                    frame_path = os.path.join(frames_dir, f"{os.path.splitext(frame_name)[0]}_restored.png")
                    cv2.imwrite(frame_path, frame)
            
            # Visualize results if requested
            if visualize and clean is not None:
                # Visualize middle frame
                t_idx = clean.size(1) // 2
                visualize_results(
                    degraded[:, t_idx],
                    output[:, t_idx],
                    clean[:, t_idx],
                    save_path=os.path.join(video_output_dir, f'comparison.png')
                )
            
            # Save video if requested
            if save_video:
                # Create video from output frames
                if save_frames:
                    # Use the frames we already saved
                    output_video_path = os.path.join(video_output_dir, f"{video_name}_restored.mp4")
                    frames_to_video(frames_dir, output_video_path)
                else:
                    # Save frames temporarily and create video
                    temp_frames_dir = os.path.join(video_output_dir, 'temp_frames')
                    os.makedirs(temp_frames_dir, exist_ok=True)
                    
                    # Denormalize and convert to numpy
                    output_np = ((output[0] + 1) / 2).clamp(0, 1).cpu().numpy()
                    output_np = (output_np * 255).astype(np.uint8)
                    
                    # Save each frame
                    for t in range(output_np.shape[0]):
                        frame = output_np[t].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                        frame = frame[:, :, ::-1]  # RGB -> BGR for OpenCV
                        frame_path = os.path.join(temp_frames_dir, f"frame_{t:04d}.png")
                        cv2.imwrite(frame_path, frame)
                    
                    # Create video
                    output_video_path = os.path.join(video_output_dir, f"{video_name}_restored.mp4")
                    frames_to_video(temp_frames_dir, output_video_path)
                    
                    # Remove temporary frames
                    import shutil
                    shutil.rmtree(temp_frames_dir)
                
                # Create comparison video if ground truth is available
                if clean is not None and os.path.exists(output_video_path):
                    # Create input video
                    input_frames_dir = os.path.join(video_output_dir, 'input_frames')
                    os.makedirs(input_frames_dir, exist_ok=True)
                    
                    # Denormalize and convert to numpy
                    input_np = ((degraded[0] + 1) / 2).clamp(0, 1).cpu().numpy()
                    input_np = (input_np * 255).astype(np.uint8)
                    
                    # Save each frame
                    for t in range(input_np.shape[0]):
                        frame = input_np[t].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                        frame = frame[:, :, ::-1]  # RGB -> BGR for OpenCV
                        frame_path = os.path.join(input_frames_dir, f"frame_{t:04d}.png")
                        cv2.imwrite(frame_path, frame)
                    
                    # Create input video
                    input_video_path = os.path.join(video_output_dir, f"{video_name}_input.mp4")
                    frames_to_video(input_frames_dir, input_video_path)
                    
                    # Create comparison video
                    comparison_video_path = os.path.join(video_output_dir, f"{video_name}_comparison.mp4")
                    create_comparison_video(input_video_path, output_video_path, comparison_video_path)
                    
                    # Remove temporary frames
                    import shutil
                    shutil.rmtree(input_frames_dir)
    
    # Calculate average metrics
    if metrics_count > 0:
        avg_metrics = {metric: value / metrics_count for metric, value in metrics_sum.items()}
        print("Average metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            for metric, value in avg_metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
    
    print(f"Results saved to {output_dir}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    model = build_model(config, args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")
    
    # Create test dataset
    test_dataset = VideoFramesDataset(
        root_dir=args.input_dir,
        sequence_length=config['data']['sequence_length'],
        frame_size=tuple(config['data']['frame_size']),
        is_training=False,
        degradation_type='all'
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one video at a time
        shuffle=False,
        num_workers=config['general']['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Run test
    test(
        model,
        test_loader,
        device,
        args.output_dir,
        save_frames=args.save_frames,
        save_video=args.save_video,
        visualize=args.visualize
    )

if __name__ == '__main__':
    import cv2
    from utils.data_utils import frames_to_video
    main()
