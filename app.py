import os
import uuid
import yaml
import torch
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify, url_for, session
from werkzeug.utils import secure_filename
from models.vrt_model import VRT
from utils.data_utils import extract_frames, frames_to_video
from translations import translations

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create upload and results directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.before_request
def before_request():
    # Set default language to Chinese if not set
    if 'lang_code' not in session:
        session['lang_code'] = 'zh'

# Global variables for model and device
model = None
device = None
config = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, device_name='cuda'):
    global model, device, config
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
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
    
    print(f"Model loaded from {checkpoint_path}")

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

def process_video(input_path, output_path, sequence_length, overlap):
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
    
    # Get frame size from config
    frame_size = config['data']['frame_size']
    
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
        for seq_idx, (start_idx, seq) in enumerate(sequences):
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
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"Video processing completed. Output saved to {output_path}")
    return output_path

@app.route('/')
def index():
    lang_code = session.get('lang_code', 'zh')
    return render_template('index.html', lang_code=lang_code, translations=translations)

@app.route('/set_language', methods=['POST'])
def set_language():
    lang_code = request.form.get('lang_code', 'zh')
    session['lang_code'] = lang_code
    return jsonify({'status': 'success'})

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
    
    # Get sequence length and overlap from form or use defaults
    sequence_length = int(request.form.get('sequence_length', config['data']['sequence_length']))
    overlap = int(request.form.get('overlap', 2))
    
    try:
        # Process video
        process_video(input_path, output_path, sequence_length, overlap)
        
        # Return paths for display
        return jsonify({
            'success': True,
            'input_video': url_for('static', filename=f'uploads/{unique_filename}'),
            'output_video': url_for('static', filename=f'results/{output_filename}')
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
        print(f"Warning: Checkpoint file {checkpoint_path} not found. Model will not be loaded.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
