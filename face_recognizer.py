import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import time
import json
from collections import defaultdict, Counter
import face_recognition

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, feature_extract=True, model_name='resnet50'):
        super(FaceRecognitionModel, self).__init__()
        
        if model_name == 'resnet50':
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            
            # Freeze parameters if feature extracting
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final fully connected layer
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        elif model_name == 'efficientnet':
            from torchvision.models import efficientnet_b0
            self.model = efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            
            # Freeze parameters if feature extracting
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        elif model_name == 'mobilenet':
            from torchvision.models import mobilenet_v2
            self.model = mobilenet_v2(pretrained=True)
            num_features = self.model.classifier[1].in_features
            
            # Freeze parameters if feature extracting
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)

class FaceRecognizer:
    
    def __init__(self, model_path=None, method="deep_learning", threshold=0.6, device=None):
        
        self.method = method
        self.threshold = threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self.class_mapping = {}
        self.unknown_counter = 0
        
        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        if method == "deep_learning":
            # Load the model
            if model_path is None:
                model_path = "data/model/face_recognition_mobilenet.pth"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize the model
            num_classes = checkpoint['num_classes']
            model_type = checkpoint.get('model_type', 'mobilenet')
            feature_extract = checkpoint.get('feature_extract', True)
            
            self.model = FaceRecognitionModel(
                num_classes=num_classes,
                feature_extract=feature_extract,
                model_name=model_type
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load class mapping
            mapping_file = os.path.join(os.path.dirname(model_path), "class_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    self.class_mapping = json.load(f)
                
            else:
                print("Warning: Class mapping file not found.")
                self.class_mapping = {str(i): f"Person_{i}" for i in range(num_classes)}
            
            # Define image transforms for inference
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        elif method == "face_recognition":
            # No pre-trained model needed for face_recognition library
            pass
        
        else:
            raise ValueError(f"Unsupported face recognition method: {method}")
        
        # Track performance
        self.recognition_times = []
    
    def add_person(self, face_image, name):
        
        if self.method == "face_recognition":
            try:
                # Find face encodings
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) != 1:
                    return False
                
                # Add to known faces
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                return True
            
            except Exception as e:
                print(f"Error adding person: {str(e)}")
                return False
        
        else:
            # For deep learning method, adding a person would require retraining
            # which is not supported in this implementation
            print("Adding a person is not supported for the deep learning method.")
            return False
    
    def recognize_face(self, face_image):
        
        start_time = time.time()
        
        if self.method == "deep_learning":
            try:
                # Preprocess the image
                img_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    max_prob, predicted_idx = torch.max(probabilities, 0)
                
                # Get the class label
                predicted_idx = predicted_idx.item()
                confidence = max_prob.item()
                
                # Check if confidence is above threshold
                if confidence >= self.threshold:
                    name = self.class_mapping.get(str(predicted_idx), f"Person_{predicted_idx}")
                else:
                    name = "Unknown"
                
                # Record recognition time
                end_time = time.time()
                self.recognition_times.append(end_time - start_time)
                
                return name, confidence
            
            except Exception as e:
                print(f"Error recognizing face: {str(e)}")
                return "Error", 0.0
        
        elif self.method == "face_recognition":
            try:
                # If no known faces, return unknown
                if len(self.known_face_encodings) == 0:
                    # Generate a unique ID for unknown faces
                    self.unknown_counter += 1
                    return f"Unknown_{self.unknown_counter}", 0.0
                
                # Find face encodings
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) != 1:
                    # Generate a unique ID for unrecognized faces
                    self.unknown_counter += 1
                    return f"Unknown_{self.unknown_counter}", 0.0
                
                # Compare with known faces
                face_encoding = face_encodings[0]
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                # Find the best match
                best_match_idx = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_idx]
                
                
                confidence = 1 - best_match_distance
                
                # Check if confidence is above threshold
                if confidence >= self.threshold:
                    name = self.known_face_names[best_match_idx]
                else:
                    # Generate a unique ID for unknown faces
                    self.unknown_counter += 1
                    name = f"Unknown_{self.unknown_counter}"
                
                # Record recognition time
                end_time = time.time()
                self.recognition_times.append(end_time - start_time)
                
                return name, confidence
            
            except Exception as e:
                print(f"Error recognizing face: {str(e)}")
                # Generate a unique ID for error faces
                self.unknown_counter += 1
                return f"Unknown_{self.unknown_counter}", 0.0
        
        else:
            raise ValueError(f"Unsupported face recognition method: {self.method}")
    
    def get_average_recognition_time(self):
        if not self.recognition_times:
            return 0
        return sum(self.recognition_times) / len(self.recognition_times) * 1000

class VideoFaceRecognizer:
    
    def __init__(self, detector=None, recognizer=None, track_faces=True, recognition_interval=5):
        
        from face_detector import FaceDetector, FaceTracker
        
        # Initialize detector if not provided
        if detector is None:
            detector = FaceDetector(method="dnn", min_confidence=0.5)
        self.detector = detector
        
        # Initialize recognizer if not provided
        if recognizer is None:
            recognizer = FaceRecognizer(method="face_recognition", threshold=0.6)
        self.recognizer = recognizer
        
        # Initialize face tracker
        self.track_faces = track_faces
        if track_faces:
            self.tracker = FaceTracker(max_disappeared=50, min_detection_confidence=0.5)
        
        # Recognition settings
        self.recognition_interval = recognition_interval
        self.frame_count = 0
        
        # Store recognition results
        self.face_identities = {}  # Map face IDs to identities
        self.face_confidences = {}  # Map face IDs to confidence scores
        self.face_timestamps = {}  # Map face IDs to last recognition timestamp
        
        # Track recognition statistics
        self.recognition_count = 0
        self.known_count = 0
        self.unknown_count = 0
    
    def process_frame(self, frame):
        
        self.frame_count += 1
        current_time = time.time()
        
        # Detect faces
        face_detections = self.detector.detect_faces(frame)
        
        # Track faces if enabled
        if self.track_faces:
            tracked_faces = self.tracker.update(face_detections)
            face_bboxes = {face_id: bbox for face_id, bbox in tracked_faces.items()}
        else:
            # Assign temporary IDs to detected faces
            face_bboxes = {i: detection[:4] for i, detection in enumerate(face_detections)}
        
        # Check if we should run recognition on this frame
        run_recognition = (self.frame_count % self.recognition_interval == 0)
        
        # Process each face
        faces_info = []
        for face_id, bbox in face_bboxes.items():
            x, y, w, h = bbox
            
            # Check if we already have an identity for this face
            if face_id in self.face_identities:
                name = self.face_identities[face_id]
                confidence = self.face_confidences[face_id]
                last_recognition = self.face_timestamps[face_id]
                
                # Re-run recognition if it's been a while
                if run_recognition and (current_time - last_recognition > 1.0):
                    # Extract face from the frame
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Recognize the face
                    name, confidence = self.recognizer.recognize_face(face_img)
                    
                    # Update face information
                    self.face_identities[face_id] = name
                    self.face_confidences[face_id] = confidence
                    self.face_timestamps[face_id] = current_time
                    
                    # Update statistics
                    self.recognition_count += 1
                    if name.startswith("Unknown"):
                        self.unknown_count += 1
                    else:
                        self.known_count += 1
            
            else:
                # New face, run recognition
                face_img = frame[y:y+h, x:x+w]
                name, confidence = self.recognizer.recognize_face(face_img)
                
                # Store face information
                self.face_identities[face_id] = name
                self.face_confidences[face_id] = confidence
                self.face_timestamps[face_id] = current_time
                
                # Update statistics
                self.recognition_count += 1
                if name.startswith("Unknown"):
                    self.unknown_count += 1
                else:
                    self.known_count += 1
            
            # Store face info
            faces_info.append((bbox, name, confidence))
        
        # Draw results on the frame
        processed_frame = self.draw_recognition_results(frame, faces_info)
        
        return processed_frame, faces_info
    
    def draw_recognition_results(self, frame, faces_info):
        
        result = frame.copy()
        
        for bbox, name, confidence in faces_info:
            x, y, w, h = bbox
            
            # Choose color based on recognition status
            if name.startswith("Unknown"):
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            
            # Add a background for the text for better readability
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x, y - 25), (x + text_size[0], y), color, -1)
            cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def process_video(self, input_path, output_path=None, display=False, max_frames=None):
       
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        # Create video writer if output path is specified
        if output_path:
            # Change the extension to .mp4 for better browser compatibility
            output_path = output_path.replace('.avi', '.mp4')
            
            # Use a web-compatible codec
            try:
                # Try H.264 codec first (most compatible with browsers)
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            except:
                try:
                    # Fallback to MP4V codec if H.264 is not available
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                except:
                    # Last resort - use a base codec that should be available
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_path = output_path.replace('.mp4', '.avi')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
        
        # Reset frame count and statistics
        self.frame_count = 0
        self.recognition_count = 0
        self.known_count = 0
        self.unknown_count = 0
        
        # Store results
        recognition_results = {}
        
        # Process frames
        try:
            while cap.isOpened() and (max_frames is None or self.frame_count < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame
                processed_frame, faces_info = self.process_frame(frame)
                
                # Store results
                recognition_results[self.frame_count] = faces_info
                
                # Write to output video
                if output_path:
                    out.write(processed_frame)
                
                # Display the frame
                if display:
                    cv2.imshow('Face Recognition', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress
                if self.frame_count % 100 == 0:
                    progress = self.frame_count / total_frames * 100
                    print(f"Processing: {progress:.1f}% complete ({self.frame_count}/{total_frames})")
        
        finally:
            # Release resources
            cap.release()
            if output_path:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        return recognition_results
    
    def set_identity(self, face_id, name):
        
        if face_id in self.face_identities:
            self.face_identities[face_id] = name
            current_time = time.time()
            self.face_timestamps[face_id] = current_time
            return True
        return False

def test_face_recognizer():
    """Test the face recognizer on a sample image, video, or webcam."""
    import argparse
    from face_detector import FaceDetector
    
    parser = argparse.ArgumentParser(description="Test the face recognizer")
    parser.add_argument("--input", type=str, default="0",
                        help="Path to an image, video file, or camera index (default: 0 for webcam)")
    parser.add_argument("--detector", type=str, default="dnn",
                        choices=["dnn", "hog", "haar"],
                        help="Face detection method to use")
    parser.add_argument("--recognizer", type=str, default="face_recognition",
                        choices=["deep_learning", "face_recognition"],
                        help="Face recognition method to use")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained model file (for deep_learning method)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Recognition threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the output (image or video)")
    parser.add_argument("--display", action="store_true",
                        help="Display the frames while processing")
    parser.add_argument("--no-tracking", dest="tracking", action="store_false",
                        help="Disable face tracking")
    parser.add_argument("--interval", type=int, default=5,
                        help="Recognition interval (frames)")
    
    args = parser.parse_args()
    
    # Initialize detector and recognizer
    detector = FaceDetector(method=args.detector, min_confidence=0.5)
    recognizer = FaceRecognizer(model_path=args.model, method=args.recognizer, threshold=args.threshold)
    
    # Initialize video face recognizer
    video_recognizer = VideoFaceRecognizer(
        detector=detector,
        recognizer=recognizer,
        track_faces=args.tracking,
        recognition_interval=args.interval
    )
    
    # Check if input is an image, video, or camera
    if args.input.isdigit():
        # Use webcam
        cap = cv2.VideoCapture(int(args.input))
        if not cap.isOpened():
            print(f"Failed to open camera: {args.input}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, faces_info = video_recognizer.process_frame(frame)
                
                cv2.imshow('Face Recognition', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video file
        video_recognizer.process_video(args.input, args.output, display=args.display)
    
    else:
        # Process image file
        image = cv2.imread(args.input)
        if image is None:
            print(f"Failed to load image: {args.input}")
            return
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        # Process each face
        for i, face in enumerate(faces):
            x, y, w, h = face[:4]
            face_img = image[y:y+h, x:x+w]
            
            # Recognize the face
            name, confidence = recognizer.recognize_face(face_img)
            
            # Choose color based on recognition status
            if name.startswith("Unknown"):
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if args.display:
            cv2.imshow('Face Recognition', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if args.output:
            cv2.imwrite(args.output, image)

if __name__ == "__main__":
    test_face_recognizer()