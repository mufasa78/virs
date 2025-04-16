# Deep Learning Video Image Repair System

A comprehensive deep learning system for repairing and enhancing degraded videos, including features for frame alignment, super-resolution, denoising, deblurring, and color enhancement.

## Features

- **Frame Alignment**: Utilizes temporal information from adjacent frames to improve restoration quality
- **Super-Resolution**: Enhances the resolution of video frames
- **Denoising and Deblurring**: Removes noise and motion blur from frames
- **Color and Contrast Enhancement**: Improves visual appeal by adjusting color balance and contrast
- **Real-Time Processing**: Optimized for efficient processing of high-resolution videos

## Project Structure

```
Deep Learning Video Image Repair System
├── data/
│   ├── raw/                  # Raw video data
│   ├── processed/            # Processed frames
│   └── results/              # Output results
├── models/
│   ├── alignment/            # Frame alignment models
│   ├── super_resolution/     # Super-resolution models
│   ├── denoising/            # Denoising models
│   ├── deblurring/           # Deblurring models
│   └── color_enhancement/    # Color enhancement models
├── utils/
│   ├── data_utils.py         # Data loading and preprocessing utilities
│   ├── metrics.py            # Evaluation metrics (PSNR, SSIM)
│   └── visualization.py      # Visualization utilities
├── config/
│   └── config.yaml           # Configuration parameters
├── train.py                  # Training script
├── test.py                   # Testing script
├── inference.py              # Inference script for real-world videos
└── requirements.txt          # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-learning-video-repair.git
cd deep-learning-video-repair
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Web Interface

The system provides a user-friendly web interface using Streamlit. There are two versions of the app:

1. **Full Version** (requires PyTorch):
```bash
python -m streamlit run streamlit_app.py
```

2. **Simplified Version** (no PyTorch dependency):
```bash
python -m streamlit run fixed_app.py
```

The simplified version is recommended for deployment as it's more stable and has fewer dependencies.

### Training

To train the model:

```bash
python train.py --config config/config.yaml
```

To resume training from a checkpoint:

```bash
python train.py --config config/config.yaml --resume --checkpoint path/to/checkpoint.pth
```

### Testing

To evaluate the model on a test dataset:

```bash
python test.py --config config/config.yaml --checkpoint path/to/checkpoint.pth --input_dir path/to/test/data --output_dir path/to/results --save_video --visualize
```

### Inference

To process a single video:

```bash
python inference.py --config config/config.yaml --checkpoint path/to/checkpoint.pth --input path/to/video.mp4 --output path/to/output.mp4
```

## Model Architecture

The system implements the Video Restoration Transformer (VRT), a state-of-the-art model for video restoration tasks. VRT effectively captures long-range temporal dependencies between frames using a transformer-based architecture with the following key components:

1. **Temporal Mutual Self-Attention (TMSA)**: Divides the video into small clips and applies mutual attention for joint motion estimation, feature alignment, and feature fusion.

2. **Parallel Warping**: Further fuses information from neighboring frames through parallel feature warping.

3. **Multi-Scale Processing**: Processes video at multiple scales to capture both fine details and global context.

## Evaluation Metrics

The system evaluates restoration quality using the following metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the pixel-level accuracy of restoration
- **SSIM (Structural Similarity Index)**: Measures the structural similarity between restored and ground truth frames
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures the perceptual similarity using deep features

## Results

Example results will be shown here with before/after comparisons.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [VRT: Video Restoration Transformer](https://github.com/JingyunLiang/VRT)
- [EDVR: Enhanced Deformable Video Restoration](https://github.com/xinntao/EDVR)
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://github.com/swz30/Restormer)
