import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VideoFramesDataset(Dataset):
    """Dataset for loading video frames for restoration tasks."""

    def __init__(self, root_dir, sequence_length=7, frame_size=(256, 256),
                 is_training=True, degradation_type='all'):
        """
        Args:
            root_dir (str): Directory with all the video frames.
            sequence_length (int): Number of consecutive frames to load.
            frame_size (tuple): Size to resize frames to (height, width).
            is_training (bool): Whether this is for training or validation/testing.
            degradation_type (str): Type of degradation ('all', 'blur', 'noise', 'low_res').
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.is_training = is_training
        self.degradation_type = degradation_type

        # Get all video directories
        self.video_dirs = [d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))]

        # Create a list of all valid sequences
        self.sequences = []
        for video_dir in self.video_dirs:
            # Check if the video directory has the expected structure
            gt_dir = os.path.join(root_dir, video_dir, 'GT')
            if not os.path.exists(gt_dir):
                print(f"Warning: GT directory not found in {os.path.join(root_dir, video_dir)}")
                continue

            # Get clean frames
            frames = sorted([f for f in os.listdir(gt_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))])

            # Create sequences of consecutive frames
            for i in range(len(frames) - sequence_length + 1):
                self.sequences.append({
                    'video_dir': video_dir,
                    'start_idx': i,
                    'frames': frames[i:i+sequence_length]
                })

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        video_dir = sequence['video_dir']
        frames = sequence['frames']

        # Load frames
        clean_frames = []
        degraded_frames = []

        for frame_name in frames:
            # Load clean frame
            clean_path = os.path.join(self.root_dir, video_dir, 'GT', frame_name)
            clean_frame = cv2.imread(clean_path)
            clean_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
            clean_frame = cv2.resize(clean_frame, (self.frame_size[1], self.frame_size[0]))

            # Load or create degraded frame based on degradation type
            if self.degradation_type == 'all':
                # For training, we might want to randomly choose a degradation
                if self.is_training:
                    degradation = np.random.choice(['blur', 'noise', 'low_res'])
                else:
                    degradation = 'blur'  # Default for testing
            else:
                degradation = self.degradation_type

            # Apply degradation
            if degradation == 'blur':
                degraded_path = os.path.join(self.root_dir, video_dir, 'blur', frame_name)
                if os.path.exists(degraded_path):
                    degraded_frame = cv2.imread(degraded_path)
                    degraded_frame = cv2.cvtColor(degraded_frame, cv2.COLOR_BGR2RGB)
                else:
                    # Apply synthetic blur
                    kernel_size = np.random.choice([3, 5, 7])
                    degraded_frame = cv2.GaussianBlur(clean_frame, (kernel_size, kernel_size), 0)

            elif degradation == 'noise':
                degraded_path = os.path.join(self.root_dir, video_dir, 'noise', frame_name)
                if os.path.exists(degraded_path):
                    degraded_frame = cv2.imread(degraded_path)
                    degraded_frame = cv2.cvtColor(degraded_frame, cv2.COLOR_BGR2RGB)
                else:
                    # Apply synthetic noise
                    noise_level = np.random.uniform(5, 50)
                    noise = np.random.normal(0, noise_level, clean_frame.shape).astype(np.uint8)
                    degraded_frame = np.clip(clean_frame + noise, 0, 255).astype(np.uint8)

            elif degradation == 'low_res':
                degraded_path = os.path.join(self.root_dir, video_dir, 'low_res', frame_name)
                if os.path.exists(degraded_path):
                    degraded_frame = cv2.imread(degraded_path)
                    degraded_frame = cv2.cvtColor(degraded_frame, cv2.COLOR_BGR2RGB)
                else:
                    # Apply synthetic downsampling and upsampling
                    scale_factor = np.random.choice([2, 4])
                    h, w = clean_frame.shape[:2]
                    small = cv2.resize(clean_frame, (w//scale_factor, h//scale_factor),
                                      interpolation=cv2.INTER_CUBIC)
                    degraded_frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

            # Resize to target size
            degraded_frame = cv2.resize(degraded_frame, (self.frame_size[1], self.frame_size[0]))

            # Apply transformations
            clean_frame = self.transform(clean_frame)
            degraded_frame = self.transform(degraded_frame)

            clean_frames.append(clean_frame)
            degraded_frames.append(degraded_frame)

        # Stack frames along a new dimension
        clean_frames = torch.stack(clean_frames, dim=0)
        degraded_frames = torch.stack(degraded_frames, dim=0)

        return {'degraded': degraded_frames, 'clean': clean_frames,
                'video_name': video_dir, 'frame_names': frames}


def create_dataloader(config, is_training=True):
    """Create data loaders for training and validation."""
    if is_training:
        dataset = VideoFramesDataset(
            root_dir=config['data']['train_path'],
            sequence_length=config['data']['sequence_length'],
            frame_size=tuple(config['data']['frame_size']),
            is_training=True,
            degradation_type='all'
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['general']['num_workers'],
            pin_memory=True
        )
    else:
        dataset = VideoFramesDataset(
            root_dir=config['data']['val_path'],
            sequence_length=config['data']['sequence_length'],
            frame_size=tuple(config['data']['frame_size']),
            is_training=False,
            degradation_type='all'
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['general']['num_workers'],
            pin_memory=True
        )

    return dataloader


def extract_frames(video_path, output_dir, frame_size=None):
    """Extract frames from a video file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_size:
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))

        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return frame_count


def frames_to_video(frames_dir, output_path, fps=30, frame_size=None):
    """Convert a directory of frames to a video file."""
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    if frame_size:
        h, w = frame_size
    else:
        h, w = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        if frame_size:
            frame = cv2.resize(frame, (w, h))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")


def apply_degradation(frame, degradation_type, params=None):
    """Apply synthetic degradation to a frame."""
    if degradation_type == 'blur':
        kernel_size = params.get('kernel_size', 5)
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    elif degradation_type == 'noise':
        noise_level = params.get('noise_level', 25)
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
        return np.clip(frame + noise, 0, 255).astype(np.uint8)

    elif degradation_type == 'low_res':
        scale_factor = params.get('scale_factor', 4)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//scale_factor, h//scale_factor), interpolation=cv2.INTER_CUBIC)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    else:
        return frame  # Return original frame if degradation type is not recognized
