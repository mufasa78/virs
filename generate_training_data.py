import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import random
from utils.data_utils import apply_degradation

def create_video_frames(output_dir, num_frames=30, resolution=(256, 256), fps=30):
    """Create a synthetic video with moving shapes for training."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'GT'), exist_ok=True)
    
    # Create a black canvas
    width, height = resolution
    
    # Generate random shapes
    shapes = []
    for _ in range(5):  # Generate 5 random shapes
        shape_type = random.choice(['circle', 'rectangle'])
        color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        if shape_type == 'circle':
            center = (random.randint(50, width-50), random.randint(50, height-50))
            radius = random.randint(20, 40)
            velocity = (random.uniform(-3, 3), random.uniform(-3, 3))
            shapes.append({
                'type': shape_type,
                'center': center,
                'radius': radius,
                'color': color,
                'velocity': velocity
            })
        else:  # rectangle
            top_left = (random.randint(10, width-100), random.randint(10, height-100))
            bottom_right = (top_left[0] + random.randint(40, 80), top_left[1] + random.randint(40, 80))
            velocity = (random.uniform(-3, 3), random.uniform(-3, 3))
            shapes.append({
                'type': shape_type,
                'top_left': top_left,
                'bottom_right': bottom_right,
                'color': color,
                'velocity': velocity
            })
    
    # Generate frames
    for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
        # Create a black canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw and update shapes
        for shape in shapes:
            if shape['type'] == 'circle':
                cv2.circle(frame, (int(shape['center'][0]), int(shape['center'][1])), shape['radius'], shape['color'], -1)
                
                # Update position
                shape['center'] = (
                    shape['center'][0] + shape['velocity'][0],
                    shape['center'][1] + shape['velocity'][1]
                )
                
                # Bounce off walls
                if shape['center'][0] - shape['radius'] < 0 or shape['center'][0] + shape['radius'] > width:
                    shape['velocity'] = (-shape['velocity'][0], shape['velocity'][1])
                if shape['center'][1] - shape['radius'] < 0 or shape['center'][1] + shape['radius'] > height:
                    shape['velocity'] = (shape['velocity'][0], -shape['velocity'][1])
            
            else:  # rectangle
                cv2.rectangle(frame, 
                             (int(shape['top_left'][0]), int(shape['top_left'][1])), 
                             (int(shape['bottom_right'][0]), int(shape['bottom_right'][1])), 
                             shape['color'], -1)
                
                # Update position
                shape['top_left'] = (
                    shape['top_left'][0] + shape['velocity'][0],
                    shape['top_left'][1] + shape['velocity'][1]
                )
                shape['bottom_right'] = (
                    shape['bottom_right'][0] + shape['velocity'][0],
                    shape['bottom_right'][1] + shape['velocity'][1]
                )
                
                # Bounce off walls
                if shape['top_left'][0] < 0 or shape['bottom_right'][0] > width:
                    shape['velocity'] = (-shape['velocity'][0], shape['velocity'][1])
                if shape['top_left'][1] < 0 or shape['bottom_right'][1] > height:
                    shape['velocity'] = (shape['velocity'][0], -shape['velocity'][1])
        
        # Save the clean frame
        clean_frame_path = os.path.join(output_dir, 'GT', f'frame_{frame_idx:04d}.png')
        cv2.imwrite(clean_frame_path, frame)
    
    return output_dir

def create_degraded_frames(video_dir, degradation_types=['blur', 'noise', 'low_res']):
    """Create degraded versions of the clean frames."""
    gt_dir = os.path.join(video_dir, 'GT')
    frame_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Create directories for each degradation type
    for degradation_type in degradation_types:
        os.makedirs(os.path.join(video_dir, degradation_type), exist_ok=True)
    
    # Process each frame
    for frame_file in tqdm(frame_files, desc="Creating degraded frames"):
        # Load clean frame
        clean_frame_path = os.path.join(gt_dir, frame_file)
        clean_frame = cv2.imread(clean_frame_path)
        
        # Create degraded versions
        for degradation_type in degradation_types:
            if degradation_type == 'blur':
                params = {'kernel_size': random.choice([3, 5, 7])}
            elif degradation_type == 'noise':
                params = {'noise_level': random.uniform(10, 50)}
            elif degradation_type == 'low_res':
                params = {'scale_factor': random.choice([2, 4])}
            else:
                params = {}
            
            # Apply degradation
            degraded_frame = apply_degradation(clean_frame, degradation_type, params)
            
            # Save degraded frame
            degraded_frame_path = os.path.join(video_dir, degradation_type, frame_file)
            cv2.imwrite(degraded_frame_path, degraded_frame)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data for video restoration')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--num_videos', type=int, default=5, help='Number of videos to generate')
    parser.add_argument('--num_frames', type=int, default=30, help='Number of frames per video')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256], help='Frame resolution (height width)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset split')
    args = parser.parse_args()
    
    # Create output directory for the specified split
    output_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate videos
    for video_idx in range(args.num_videos):
        video_dir = os.path.join(output_dir, f'video_{video_idx:03d}')
        create_video_frames(video_dir, args.num_frames, tuple(args.resolution))
        create_degraded_frames(video_dir)
    
    print(f"Generated {args.num_videos} videos with {args.num_frames} frames each in {output_dir}")

if __name__ == '__main__':
    main()
