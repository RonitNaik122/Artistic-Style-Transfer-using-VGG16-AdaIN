from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import uuid
import time
from PIL import Image
import cv2
import sys

# Import the neural style transfer code
# Assuming style_transfer.py contains the functions from your pasted code
from style_transfer import test_image, test_video, TransformerNet, device

app = Flask(__name__)
CORS(app)

# Create directories for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODELS_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Available models
ALLOWED_MODELS = {
    'geometric_painting.pth': os.path.join(MODELS_FOLDER, 'geometric_painting.pth'),
    'oil_painting.pth': os.path.join(MODELS_FOLDER, 'oil_painting.pth'),
    'van_gogh.pth': os.path.join(MODELS_FOLDER, 'van_gogh.pth')
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/process', methods=['POST'])
def process_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model = request.form.get('model')
    
    # Validate file and model
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    if model not in ALLOWED_MODELS:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    # Generate unique filename to prevent conflicts
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the uploaded file
    file.save(file_path)
    
    try:
        # Process file based on type
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Process image
            output_path = test_image(
                image_path=file_path,
                checkpoint_model=ALLOWED_MODELS[model],
                save_path=RESULT_FOLDER
            )
        else:
            # Process video
            output_path = test_video(
                video_path=file_path,
                checkpoint_model=ALLOWED_MODELS[model],
                save_path=RESULT_FOLDER,
                max_dim=480  # Can be made configurable through the request
            )
        
        # Return the path to the processed file
        result_filename = os.path.basename(output_path)
        return jsonify({
            'success': True,
            'output_url': f'/api/results/{result_filename}'
        })
    
    except Exception as e:
        print(f"Error during processing: {str(e)}", file=sys.stderr)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """Serve the processed files"""
    return send_file(os.path.join(RESULT_FOLDER, 'results', filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)