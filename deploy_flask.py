import os
import torch
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from models.basic_model import BasicVideoRestoration
from utils.data_utils import extract_frames, frames_to_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create upload and results directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for model and device
model = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(checkpoint_path=None, device_name='cpu'):
    global model, device
    
    # Set device
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    model = BasicVideoRestoration(
        in_channels=3,
        out_channels=3,
        hidden_channels=64
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_checkpoint(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("Using untrained model")
    
    model = model.to(device)
    model.eval()

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

def process_video(input_path, output_path, sequence_length=5):
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
    print(f"Processing frames with sequence length {sequence_length}...")
    
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
                output_path = os.path.join(output_frames_dir, f"frame_{global_idx:04d}.png")
                cv2.imwrite(output_path, frame)
    
    # Create output video
    print(f"Creating output video {output_path}...")
    frames_to_video(output_frames_dir, output_path)
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Video processing completed. Output saved to {output_path}")
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    base_name, extension = os.path.splitext(filename)
    unique_filename = f"{base_name}_{unique_id}{extension}"
    
    # Save uploaded file
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(input_path)
    
    # Generate output path
    output_filename = f"{base_name}_{unique_id}_restored.mp4"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
    
    # Get sequence length from form or use default
    sequence_length = int(request.form.get('sequence_length', 5))
    
    try:
        # Process video
        process_video(input_path, output_path, sequence_length)
        
        # Return paths for display
        return jsonify({
            'success': True,
            'input_video': f'/static/uploads/{unique_filename}',
            'output_video': f'/static/results/{output_filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Load model at startup
    checkpoint_path = 'checkpoints/best_model.pth'  # Update with your checkpoint path
    if os.path.exists(checkpoint_path):
        load_model(checkpoint_path)
    else:
        load_model()  # Use untrained model
    
    app.run(debug=True, host='0.0.0.0', port=5000)
