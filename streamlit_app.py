import os
import yaml
import torch
import tempfile
import cv2
import numpy as np
import streamlit as st
import asyncio
from models.basic_model import BasicVideoRestoration
from utils.data_utils import extract_frames, frames_to_video
from translations import translations

# Initialize asyncio event loop for Python 3.13 compatibility
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Initialize PyTorch with no gradients for inference
torch.set_grad_enabled(False)

# Set page configuration
st.set_page_config(
    page_title=translations['zh']['page_title'],
    page_icon="🎬",
    layout="wide"
)

# Language selector in sidebar
if 'lang_code' not in st.session_state:
    st.session_state.lang_code = 'zh'

lang = st.sidebar.selectbox(
    "语言/Language",
    ["中文", "English"],
    index=0 if st.session_state.lang_code == 'zh' else 1,
    key='language_selector'
)

# Update language code in session state
st.session_state.lang_code = 'zh' if lang == "中文" else 'en'
lang_code = st.session_state.lang_code

# Display page title and subtitle in selected language
st.title(translations[lang_code]['page_title'])
st.write(translations[lang_code]['page_subtitle'])

# Create directories for uploads and results
UPLOAD_DIR = "streamlit_uploads"
RESULTS_DIR = "streamlit_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@st.cache_resource
def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model(checkpoint_path, device_name='cpu'):
    """Load the model and cache it with Streamlit."""
    # Load configuration
    config = load_config()

    # Set device
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")

    try:
        # Build model - using BasicVideoRestoration model
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

        return model, config, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Return a dummy model for testing
        return None, config, device

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

def process_video(model, device, config, input_path, output_path, sequence_length, overlap, progress_bar=None):
    """Process a video file with the restoration model."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_frames_dir = os.path.join(temp_dir, 'input_frames')
        output_frames_dir = os.path.join(temp_dir, 'output_frames')
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)

        # Extract frames from input video
        num_frames = extract_frames(input_path, input_frames_dir)
        st.write(translations[st.session_state.lang_code]['extracted_frames'].format(num_frames))

        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Get frame size from config
        frame_size = config['data']['frame_size']

        # Process frames in sequences
        st.write(translations[st.session_state.lang_code]['processing_frames'].format(sequence_length, overlap))

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
            for seq_idx, (start_idx, seq) in enumerate(sequences):
                # Update progress bar
                if progress_bar is not None:
                    progress_bar.progress((seq_idx + 1) / len(sequences))

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
                if model is not None:
                    output = model(frames_tensor)
                    # Reshape back to [B, T, C, H, W]
                    output = output.permute(0, 2, 1, 3, 4)[0]  # [T, C, H, W]
                else:
                    # If model is None, just use the input frames as output (identity function)
                    output = frames_tensor.permute(0, 2, 1, 3, 4)[0]  # [T, C, H, W]

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
                    output_path_frame = os.path.join(output_frames_dir, f"frame_{global_idx:04d}.png")
                    cv2.imwrite(output_path_frame, frame)

        # Create output video
        st.write(translations[st.session_state.lang_code]['creating_video'])
        frames_to_video(output_frames_dir, output_path)

        st.success(translations[st.session_state.lang_code]['processing_complete'])
        return output_path

def main():
    # Model settings in sidebar
    st.sidebar.header(translations[st.session_state.lang_code]['model_settings'])

    # Check if checkpoint exists
    checkpoint_path = st.sidebar.text_input(translations[lang_code]['model_checkpoint'], "checkpoints/best_model.pth")

    # Load model
    model, config, device = load_model(checkpoint_path if os.path.exists(checkpoint_path) else None)
    if model is not None:
        if os.path.exists(checkpoint_path):
            st.sidebar.success(translations[lang_code]['model_loaded'])
        else:
            st.sidebar.warning(translations[lang_code]['using_untrained'])
    else:
        st.sidebar.warning("Using fallback processing (no model)")

    # Processing parameters
    st.sidebar.header(translations[st.session_state.lang_code]['processing_params'])
    sequence_length = st.sidebar.slider(translations[st.session_state.lang_code]['sequence_length'], 2, 16, 8,
                                      help=translations[st.session_state.lang_code]['sequence_length_help'])
    frame_overlap = st.sidebar.slider(translations[st.session_state.lang_code]['frame_overlap'], 0, 8, 2,
                                    help=translations[st.session_state.lang_code]['frame_overlap_help'])

    # File uploader
    st.subheader(translations[lang_code]['upload_video'])
    uploaded_file = st.file_uploader(translations[st.session_state.lang_code]['choose_video'],
                                    type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                                    help=translations[st.session_state.lang_code]['supported_formats'])

    if uploaded_file is not None:
        # Save uploaded file
        input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display original video
        st.subheader(translations[lang_code]['original_video'])
        st.video(input_path)

        # Process button
        if st.button(translations[lang_code]['process_another'] if 'process_another' in translations[lang_code] else 'Process Video'):
            # Create output path
            output_filename = f"enhanced_{uploaded_file.name}"
            output_path = os.path.join(RESULTS_DIR, output_filename)

            # Process video with progress bar
            st.subheader("Processing")
            progress_bar = st.progress(0)

            try:
                output_path = process_video(model, device, config, input_path, output_path,
                                           sequence_length, frame_overlap, progress_bar)

                # Display results
                st.subheader(translations[st.session_state.lang_code]['enhanced_video'])

                # Display enhanced video
                st.video(output_path)

                # Download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label=translations[st.session_state.lang_code]['download_enhanced'],
                        data=file,
                        file_name=output_filename,
                        mime="video/mp4"
                    )
            except Exception as e:
                st.error(translations[st.session_state.lang_code]['process_error'].format(str(e)))

if __name__ == "__main__":
    main()
