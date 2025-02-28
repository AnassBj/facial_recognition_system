import os
import cv2
import numpy as np
import time
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from face_detector import FaceDetector, VideoFaceDetector
from face_recognizer import FaceRecognizer, VideoFaceRecognizer

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['DATA_FOLDER'] = 'data'
app.config['MODEL_FOLDER'] = os.path.join(app.config['DATA_FOLDER'], 'model')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

detector = FaceDetector(method="dnn", min_confidence=0.5)
recognizer = FaceRecognizer(
    model_path="data/model/face_recognition_mobilenet.pth",
    method="deep_learning", 
    threshold=0.6
)

class_mapping_path = os.path.join(app.config['MODEL_FOLDER'], "class_mapping.json")
known_identities = []

# Try to load the class mapping file
if os.path.exists(class_mapping_path):
    try:
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
            # Extract just the names from the mapping (values of the dictionary)
            known_identities = list(class_mapping.values())
            print(f"Loaded {len(known_identities)} known identities from class mapping")
    except Exception as e:
        print(f"Error loading class mapping: {str(e)}")
        known_identities = ["Demo Person 1", "Demo Person 2"]  # Fallback names
else:
    print(f"Class mapping not found at {class_mapping_path}")
    known_identities = ["Demo Person 1", "Demo Person 2"]  # Fallback names

# Store session data
sessions = {}

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html', known_identities=known_identities)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads."""
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Generate a session ID
    session_id = str(uuid.uuid4())
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    base_filename, ext = os.path.splitext(filename)
    upload_filename = f"{base_filename}_{session_id}{ext}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    file.save(upload_path)
    
    # Create a dictionary to store session data
    sessions[session_id] = {
        'upload_path': upload_path,
        'original_filename': filename,
        'processed': False,
        'faces': [],
        'output_path': None,
        'start_time': time.time()
    }
    
    # Extract video information
    cap = cv2.VideoCapture(upload_path)
    if cap.isOpened():
        # Get basic video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Extract a preview frame (about 10% into the video)
        preview_frame_idx = int(frame_count * 0.1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Detect faces in the preview frame
            faces = detector.detect_faces(frame)
            
            # Save face thumbnails
            face_thumbnails = []
            for i, face in enumerate(faces):
                x, y, w, h = face[:4]
                
                # Expand the bounding box slightly
                padding = int(max(w, h) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                # Extract face thumbnail
                face_img = frame[y:y+h, x:x+w]
                
                # Save thumbnail
                face_id = f"face_{i}"
                face_thumbnail_path = os.path.join(
                    app.config['OUTPUT_FOLDER'], 
                    f"{session_id}_{face_id}.jpg"
                )
                cv2.imwrite(face_thumbnail_path, face_img)
                
                # Add face info to session
                face_info = {
                    'id': face_id,
                    'bbox': [x, y, w, h],
                    'thumbnail': os.path.basename(face_thumbnail_path),
                    'name': f"Unknown_{i+1}",
                    'is_known': False
                }
                face_thumbnails.append(face_info)
                sessions[session_id]['faces'].append(face_info)
        
        # Close the video
        cap.release()
        
        # Update session with video info
        sessions[session_id]['video_info'] = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
    
    # Redirect to the processing page
    return jsonify({
        'success': True,
        'session_id': session_id,
        'redirect': url_for('process_video_auto', session_id=session_id)  # New route
    })


@app.route('/process-video-auto/<session_id>')
def process_video_auto(session_id):
    """Process a video automatically without manual face naming."""
    if session_id not in sessions:
        return redirect(url_for('index'))
    
    # Process the video automatically
    session_data = sessions[session_id]
    input_path = session_data['upload_path']
    output_filename = f"processed_{os.path.basename(input_path).replace(' ', '_')}"
    
    # Make sure it's an mp4 for browser compatibility
    if not output_filename.lower().endswith('.mp4'):
        output_filename = os.path.splitext(output_filename)[0] + '.mp4'
        
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Initialize video recognizer with our trained model
    video_recognizer = VideoFaceRecognizer(
        detector=detector,
        recognizer=recognizer,
        track_faces=True,
        recognition_interval=5
    )
    
    # Process the video
    try:
        recognition_results = video_recognizer.process_video(input_path, output_path)
        
        # Update session data
        session_data['processed'] = True
        session_data['output_path'] = output_path
        session_data['output_filename'] = output_filename
        session_data['recognition_results'] = {
            'total_frames': len(recognition_results),
            'recognized_faces': video_recognizer.recognition_count,
            'known_faces': video_recognizer.known_count,
            'unknown_faces': video_recognizer.unknown_count
        }
        
        # Redirect to result page
        return redirect(url_for('result', session_id=session_id))
    except Exception as e:
        return render_template('error.html', error=str(e))

# Create a result page
@app.route('/result/<session_id>')
def result(session_id):
    """Show processing results."""
    if session_id not in sessions:
        return redirect(url_for('index'))
    
    # Get the output filename instead of the full path
    session_data = sessions[session_id].copy()
    if session_data.get('output_path'):
        output_filename = os.path.basename(session_data['output_path'])
        session_data['output_filename'] = output_filename
    
    return render_template('result.html', 
                          session_id=session_id, 
                          session_data=session_data)

@app.route('/process/<session_id>')
def process(session_id):
    """Render the processing page for a session."""
    if session_id not in sessions:
        return redirect(url_for('index'))
    
    return render_template('process.html', 
                           session_id=session_id, 
                           session_data=sessions[session_id])

@app.route('/identify', methods=['POST'])
def identify_face():
    """Handle face identification."""
    data = request.json
    session_id = data.get('session_id')
    face_id = data.get('face_id')
    name = data.get('name')
    
    if not session_id or not face_id or not name or session_id not in sessions:
        return jsonify({'error': 'Invalid request'}), 400
    
    # Update face info in the session
    for face in sessions[session_id]['faces']:
        if face['id'] == face_id:
            face['name'] = name
            face['is_known'] = True
            break
    
    return jsonify({'success': True})

@app.route('/process-video', methods=['POST'])
def process_video():
    """Process the video with face recognition."""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = sessions[session_id]
    
    if session_data['processed']:
        return jsonify({
            'success': True,
            'output_path': url_for('output', filename=os.path.basename(session_data['output_path']))
        })
    
    try:
        # Initialize video recognizer
        video_recognizer = VideoFaceRecognizer(
            detector=detector,
            recognizer=recognizer,
            track_faces=True,
            recognition_interval=5
        )
        
        # Add known faces to the recognizer
        for face_info in session_data['faces']:
            if face_info['is_known'] and not face_info['name'].startswith('Unknown'):
                # Load the face thumbnail
                thumbnail_path = os.path.join(
                    app.config['OUTPUT_FOLDER'], 
                    face_info['thumbnail']
                )
                if os.path.exists(thumbnail_path):
                    face_img = cv2.imread(thumbnail_path)
                    if face_img is not None:
                        # Add the face to the recognizer
                        recognizer.add_person(face_img, face_info['name'])
        
        # Process the video
        input_path = session_data['upload_path']
        output_filename = f"processed_{os.path.basename(input_path)}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        recognition_results = video_recognizer.process_video(input_path, output_path)
        
        # Update session data
        session_data['processed'] = True
        session_data['output_path'] = output_path
        session_data['recognition_results'] = {
            'total_frames': len(recognition_results),
            'recognized_faces': video_recognizer.recognition_count,
            'known_faces': video_recognizer.known_count,
            'unknown_faces': video_recognizer.unknown_count
        }
        
        return jsonify({
            'success': True,
            'output_path': url_for('output', filename=output_filename),
            'stats': session_data['recognition_results']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/<filename>')
def output(filename):
    """Serve processed video files with proper MIME types."""
    if filename.lower().endswith('.mp4'):
        return send_from_directory(
            app.config['OUTPUT_FOLDER'], 
            filename, 
            mimetype='video/mp4',
            as_attachment=False
        )
    else:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/thumbnails/<filename>')
def thumbnails(filename):
    """Serve face thumbnail images."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/session/<session_id>/status')
def session_status(session_id):
    """Get the status of a session."""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = sessions[session_id]
    
    return jsonify({
        'processed': session_data['processed'],
        'faces': session_data['faces'],
        'output_path': url_for('output', filename=os.path.basename(session_data['output_path'])) 
                      if session_data['output_path'] else None
    })

@app.route('/cleanup-sessions', methods=['POST'])
def cleanup_sessions():
    """Clean up old sessions."""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, session_data in sessions.items():
        # Remove sessions older than 1 hour
        if current_time - session_data['start_time'] > 3600:
            expired_sessions.append(session_id)
            
            # Delete uploaded and output files
            if 'upload_path' in session_data and os.path.exists(session_data['upload_path']):
                os.remove(session_data['upload_path'])
            
            if 'output_path' in session_data and session_data['output_path'] and os.path.exists(session_data['output_path']):
                os.remove(session_data['output_path'])
            
            # Delete face thumbnails
            for face in session_data.get('faces', []):
                if 'thumbnail' in face:
                    thumbnail_path = os.path.join(app.config['OUTPUT_FOLDER'], face['thumbnail'])
                    if os.path.exists(thumbnail_path):
                        os.remove(thumbnail_path)
    
    # Remove expired sessions from memory
    for session_id in expired_sessions:
        del sessions[session_id]
    
    return jsonify({'success': True, 'removed_sessions': len(expired_sessions)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)