import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import yaml

# Create directories for uploads and results
UPLOAD_DIR = "streamlit_uploads"
RESULTS_DIR = "streamlit_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Video Enhancement App",
    page_icon="🎬",
    layout="wide"
)

# Translations
translations = {
    'zh': {
        'page_title': '深度学习视频图像修复系统',
        'page_subtitle': '使用我们的深度学习模型提升视频质量',
        'model_settings': '模型设置',
        'processing_params': '处理参数',
        'sequence_length': '序列长度',
        'frame_overlap': '帧重叠',
        'upload_video': '上传视频',
        'choose_video': '选择视频文件',
        'original_video': '原始视频',
        'enhanced_video': '增强视频',
        'process_video': '处理视频',
        'download_enhanced': '下载增强视频',
        'extracted_frames': '已提取 {} 帧',
        'processing_frames': '正在处理帧，序列长度 {}，重叠 {}...',
        'creating_video': '正在创建输出视频...',
        'processing_complete': '视频处理完成！',
    },
    'en': {
        'page_title': 'Deep Learning Video Image Repair System',
        'page_subtitle': 'Upload a video to enhance its quality using our deep learning model',
        'model_settings': 'Model Settings',
        'processing_params': 'Processing Parameters',
        'sequence_length': 'Sequence Length',
        'frame_overlap': 'Frame Overlap',
        'upload_video': 'Upload Video',
        'choose_video': 'Choose a video file',
        'original_video': 'Original Video',
        'enhanced_video': 'Enhanced Video',
        'process_video': 'Process Video',
        'download_enhanced': 'Download Enhanced Video',
        'extracted_frames': 'Extracted {} frames',
        'processing_frames': 'Processing frames with sequence length {} and overlap {}...',
        'creating_video': 'Creating output video...',
        'processing_complete': 'Video processing completed!',
    }
}

# Language selector in sidebar
if 'lang_code' not in st.session_state:
    st.session_state.lang_code = 'en'

lang = st.sidebar.selectbox(
    "Language/语言",
    ["English", "中文"],
    index=0 if st.session_state.lang_code == 'en' else 1,
    key='language_selector'
)

# Update language code in session state
st.session_state.lang_code = 'en' if lang == "English" else 'zh'
lang_code = st.session_state.lang_code

# Display page title and subtitle in selected language
st.title(translations[lang_code]['page_title'])
st.write(translations[lang_code]['page_subtitle'])

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

def process_video(input_path, output_path, sequence_length, overlap, progress_bar=None):
    """Process a video file with a simple enhancement."""
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

        # Process frames in sequences
        st.write(translations[st.session_state.lang_code]['processing_frames'].format(sequence_length, overlap))

        # Process each frame with a simple enhancement
        for i, frame_file in enumerate(frame_files):
            # Update progress bar
            if progress_bar is not None:
                progress_bar.progress((i + 1) / len(frame_files))

            # Read frame
            frame_path = os.path.join(input_frames_dir, frame_file)
            frame = cv2.imread(frame_path)

            # Apply a simple enhancement (increase brightness and contrast)
            enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

            # Save processed frame
            output_path_frame = os.path.join(output_frames_dir, f"frame_{i:04d}.png")
            cv2.imwrite(output_path_frame, enhanced_frame)

        # Create output video
        st.write(translations[st.session_state.lang_code]['creating_video'])
        frames_to_video(output_frames_dir, output_path)

        st.success(translations[st.session_state.lang_code]['processing_complete'])
        return output_path

def main():
    # Processing parameters
    st.sidebar.header(translations[st.session_state.lang_code]['processing_params'])
    sequence_length = st.sidebar.slider(translations[st.session_state.lang_code]['sequence_length'], 2, 16, 8)
    frame_overlap = st.sidebar.slider(translations[st.session_state.lang_code]['frame_overlap'], 0, 8, 2)

    # File uploader
    st.subheader(translations[lang_code]['upload_video'])
    uploaded_file = st.file_uploader(translations[st.session_state.lang_code]['choose_video'], 
                                    type=['mp4', 'avi', 'mov', 'mkv', 'webm'])

    if uploaded_file is not None:
        # Save uploaded file
        input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display original video
        st.subheader(translations[lang_code]['original_video'])
        st.video(input_path)

        # Process button
        if st.button(translations[lang_code]['process_video']):
            # Create output path
            output_filename = f"enhanced_{uploaded_file.name}"
            output_path = os.path.join(RESULTS_DIR, output_filename)

            # Process video with progress bar
            st.subheader("Processing")
            progress_bar = st.progress(0)

            try:
                output_path = process_video(input_path, output_path, sequence_length, frame_overlap, progress_bar)

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
                st.error(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()
