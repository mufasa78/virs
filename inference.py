import os
import yaml
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from models.vrt_model import VRT
from utils.data_utils import extract_frames, frames_to_video

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Video Restoration Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='./results/output.mp4', help='Path to output video file')
    parser.add_argument('--frame_size', type=int, nargs=2, default=None, help='Output frame size (height width)')
    parser.add_argument('--sequence_length', type=int, default=None, help='Number of frames to process at once')
    parser.add_argument('--overlap', type=int, default=2, help='Number of overlapping frames between sequences')
    parser.add_argument('--save_frames', action='store_true', help='Save individual frames')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model(config, checkpoint_path, device, sequence_length=None, frame_size=None):
    model_type = config['model']['type']
    
    # Override sequence_length and frame_size if provided
    if sequence_length is None:
        sequence_length = config['data']['sequence_length']
    if frame_size is None:
        frame_size = config['data']['frame_size']
    
    if model_type == 'vrt':
        model_config = config['model']['vrt']
        model = VRT(
            img_size=(sequence_length, *frame_size),
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

def preprocess_frame(frame, frame_size):
    """Preprocess a frame for model input."""
    # Resize frame
    frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    frame = frame.astype(np.float32) / 255.0
    frame = frame * 2 - 1
    
    # Convert to tensor [C, H, W]
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    
    return frame

def postprocess_frame(tensor):
    """Convert model output tensor to numpy image."""
    # Denormalize
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy [H, W, C]
    frame = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Convert to uint8
    frame = (frame * 255).astype(np.uint8)
    
    return frame

def process_video(model, input_path, output_path, frame_size, sequence_length, overlap, save_frames, device):
    """Process a video file with the restoration model."""
    # Create temporary directories
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp')
    input_frames_dir = os.path.join(temp_dir, 'input_frames')
    output_frames_dir = os.path.join(temp_dir, 'output_frames')
    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # Extract frames from input video
    print(f"Extracting frames from {input_path}...")
    num_frames = extract_frames(input_path, input_frames_dir)
    print(f"Extracted {num_frames} frames")
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Process frames in sequences
    print(f"Processing frames with sequence length {sequence_length} and overlap {overlap}...")
    
    # Create sequences with overlap
    sequences = []
    for i in range(0, len(frame_files), sequence_length - overlap):
        seq = frame_files[i:i + sequence_length]
        if len(seq) < sequence_length:
            # Pad the last sequence if needed
            seq = seq + [seq[-1]] * (sequence_length - len(seq))
        sequences.append((i, seq))
    
    # Process each sequence
    with torch.no_grad():
        for seq_idx, (start_idx, seq) in enumerate(tqdm(sequences, desc="Processing sequences")):
            # Load and preprocess frames
            frames = []
            for frame_file in seq:
                frame_path = os.path.join(input_frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                frame = preprocess_frame(frame, frame_size)
                frames.append(frame)
            
            # Stack frames into a batch
            frames_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # [1, T, C, H, W]
            
            # Reshape to [B, C, T, H, W] for model
            frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).to(device)
            
            # Forward pass
            output = model(frames_tensor)
            
            # Reshape back to [B, T, C, H, W]
            output = output.permute(0, 2, 1, 3, 4)[0]  # [T, C, H, W]
            
            # Save output frames, accounting for overlap
            for i, frame_tensor in enumerate(output):
                # Skip overlapping frames except for the first sequence
                if seq_idx > 0 and i < overlap:
                    continue
                
                # Calculate the global frame index
                global_idx = start_idx + i
                if global_idx >= num_frames:
                    continue
                
                # Postprocess and save frame
                frame = postprocess_frame(frame_tensor)
                output_path = os.path.join(output_frames_dir, f"frame_{global_idx:04d}.png")
                cv2.imwrite(output_path, frame)
    
    # Create output video
    print(f"Creating output video {output_path}...")
    frames_to_video(output_frames_dir, output_path)
    
    # Save frames if requested
    if save_frames:
        frames_dir = os.path.join(os.path.dirname(output_path), 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Copy frames to output directory
        import shutil
        for frame_file in os.listdir(output_frames_dir):
            src = os.path.join(output_frames_dir, frame_file)
            dst = os.path.join(frames_dir, frame_file)
            shutil.copy2(src, dst)
        
        print(f"Frames saved to {frames_dir}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Video processing completed. Output saved to {output_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['general']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set frame size and sequence length
    frame_size = args.frame_size if args.frame_size else config['data']['frame_size']
    sequence_length = args.sequence_length if args.sequence_length else config['data']['sequence_length']
    
    # Build model
    model = build_model(config, args.checkpoint, device, sequence_length, frame_size)
    print(f"Model loaded from {args.checkpoint}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process video
    process_video(
        model,
        args.input,
        args.output,
        frame_size,
        sequence_length,
        args.overlap,
        args.save_frames,
        device
    )

if __name__ == '__main__':
    main()
