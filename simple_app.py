import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Video Enhancement App",
    page_icon="🎬",
    layout="wide"
)

# Create directories for uploads and results
UPLOAD_DIR = "streamlit_uploads"
RESULTS_DIR = "streamlit_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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
        st.error(f"No frames found in {frames_dir}")
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
    st.success(f"Video saved to {output_path}")

def process_video(input_path, output_path, progress_bar=None):
    """Simple video processing function that just adds a simple effect."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_frames_dir = os.path.join(temp_dir, 'input_frames')
        output_frames_dir = os.path.join(temp_dir, 'output_frames')
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)

        # Extract frames from input video
        num_frames = extract_frames(input_path, input_frames_dir)
        st.write(f"Extracted {num_frames} frames")

        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # Process each frame with a simple effect
        for i, frame_file in enumerate(frame_files):
            # Update progress bar
            if progress_bar is not None:
                progress_bar.progress((i + 1) / len(frame_files))

            # Read frame
            frame_path = os.path.join(input_frames_dir, frame_file)
            frame = cv2.imread(frame_path)

            # Apply a simple effect (increase brightness)
            enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

            # Save processed frame
            output_path_frame = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(output_path_frame, enhanced_frame)

        # Create output video
        st.write("Creating output video...")
        frames_to_video(output_frames_dir, output_path)

        st.success("Video processing completed!")
        return output_path

def main():
    # Display header
    st.title("Video Enhancement App")
    st.markdown("Upload a video to enhance its quality")

    # Sidebar for parameters
    st.sidebar.title("Processing Parameters")
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.2, 0.1)
    contrast = st.sidebar.slider("Contrast", 0, 50, 10, 5)

    # File uploader
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", 
                                    type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                                    help="Supported formats: MP4, AVI, MOV, MKV, WEBM")

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
                output_path = process_video(input_path, output_path, progress_bar)

                # Display results
                st.subheader("Enhanced Video")
                
                # Display enhanced video
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
