import os
import torch
import numpy as np
import cv2
import streamlit as st
import tempfile
from models.basic_model import BasicVideoRestoration
from utils.data_utils import extract_frames, frames_to_video

# Set page configuration
st.set_page_config(
    page_title="Deep Learning Video Image Repair System",
    page_icon="🎬",
    layout="wide"
)

# Create directories for uploads and results
UPLOAD_DIR = "streamlit_uploads"
RESULTS_DIR = "streamlit_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@st.cache_resource
def load_model(checkpoint_path=None, device_name='cpu'):
    """Load the model and cache it with Streamlit."""
    # Set device
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")
    
    # Build model
    model = BasicVideoRestoration(
        in_channels=3,
        out_channels=3,
        hidden_channels=64
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path)
        st.write(f"Model loaded from {checkpoint_path}")
    else:
        st.write("Using untrained model")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_frame(frame, frame_size=(128, 128)):
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

def process_video(model, device, input_path, output_path, sequence_length=5, progress_bar=None):
    """Process a video file with the restoration model."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_frames_dir = os.path.join(temp_dir, 'input_frames')
        output_frames_dir = os.path.join(temp_dir, 'output_frames')
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        
        # Extract frames from input video
        st.text(f"Extracting frames from video...")
        num_frames = extract_frames(input_path, input_frames_dir)
        st.text(f"Extracted {num_frames} frames")
        
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Process frames in sequences
        st.text(f"Processing frames with sequence length {sequence_length}...")
        
        # Create sequences
        sequences = []
        for i in range(0, len(frame_files), sequence_length):
            seq = frame_files[i:i + sequence_length]
            if len(seq) < sequence_length:
                # Pad the last sequence if needed
                seq = seq + [seq[-1]] * (sequence_length - len(seq))
            sequences.append((i, seq))
        
        # Process each sequence
        with torch.no_grad():
            for seq_idx, (start_idx, seq) in enumerate(sequences):
                # Update progress bar
                if progress_bar is not None:
                    progress_bar.progress((seq_idx + 1) / len(sequences))
                
                # Load and preprocess frames
                frames = []
                for frame_file in seq:
                    frame_path = os.path.join(input_frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    frame = preprocess_frame(frame)
                    frames.append(frame)
                
                # Stack frames into a batch
                frames_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # [1, T, C, H, W]
                
                # Reshape to [B, C, T, H, W] for model
                frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4).to(device)
                
                # Forward pass
                output = model(frames_tensor)
                
                # Reshape back to [B, T, C, H, W]
                output = output.permute(0, 2, 1, 3, 4)[0]  # [T, C, H, W]
                
                # Save output frames
                for i, frame_tensor in enumerate(output):
                    # Calculate the global frame index
                    global_idx = start_idx + i
                    if global_idx >= num_frames:
                        continue
                    
                    # Postprocess and save frame
                    frame = postprocess_frame(frame_tensor)
                    output_path_frame = os.path.join(output_frames_dir, f"frame_{global_idx:04d}.png")
                    cv2.imwrite(output_path_frame, frame)
        
        # Create output video
        st.text(f"Creating output video...")
        frames_to_video(output_frames_dir, output_path)
        
        st.text(f"Video processing completed!")
        return output_path

def main():
    # Display header
    st.title("Deep Learning Video Image Repair System")
    st.markdown("Upload a video to enhance its quality using our deep learning model")
    
    # Sidebar for model selection and parameters
    st.sidebar.title("Model Settings")
    
    # Check if checkpoint exists
    checkpoint_path = st.sidebar.text_input("Model Checkpoint Path", "checkpoints/best_model.pth")
    
    # Processing parameters
    st.sidebar.title("Processing Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", 3, 10, 5, 
                                       help="Number of frames to process at once")
    
    # Load model
    model, device = load_model(checkpoint_path if os.path.exists(checkpoint_path) else None)
    
    # File uploader
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm"])
    
    if uploaded_file is not None:
        # Save uploaded file
        input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display original video
        st.subheader("Original Video")
        st.video(input_path)
        
        # Process button
        if st.button("Process Video"):
            # Create output path
            output_filename = f"enhanced_{uploaded_file.name}"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            
            # Process video with progress bar
            st.subheader("Processing")
            progress_bar = st.progress(0)
            
            try:
                output_path = process_video(model, device, input_path, output_path, 
                                           sequence_length, progress_bar)
                
                # Display results
                st.subheader("Enhanced Video")
                st.video(output_path)
                
                # Download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Enhanced Video",
                        data=file,
                        file_name=output_filename,
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
