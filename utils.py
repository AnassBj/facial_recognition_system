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
    """PyTorch model for face recognition"""
    def __init__(self, num_classes, feature_extract=True, model_name='resnet50'):
        super(FaceRecognitionModel, self).__init__()
        
        if model_name == 'resnet50':
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
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
            
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        elif model_name == 'mobilenet':
            from torchvision.models import mobilenet_v2
            self.model = mobilenet_v2(pretrained=True)
            num_features = self.model.classifier[1].in_features
            
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)

class FaceRecognizer:
    """Class for recognizing faces in images/video frames."""
    
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
                print(f"Loaded class mapping: {self.class_mapping}")
            else:
                print("Warning: Class mapping file not found.")
                self.class_mapping = {str(i): f"Person_{i}" for i in range(num_classes)}
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        elif method == "face_recognition":
            pass
        
        else:
            raise ValueError(f"Unsupported face recognition method: {method}")
        
        self.recognition_times = []
    
    def add_person(self, face_image, name):
        
        if self.method == "face_recognition":
            try:
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) != 1:
                    return False
                
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                return True
            
            except Exception as e:
                print(f"Error adding person: {str(e)}")
                return False
        
        else:
            print("Adding a person is not supported for the deep learning method.")
            return False
    
    def recognize_face(self, face_image):
       
        start_time = time.time()
        
        if self.method == "deep_learning":
            try:
                # Preprocess the image
                img_tensor = self.transform(face_image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)[0]
                    max_prob, predicted_idx = torch.max(probabilities, 0)
                
                predicted_idx = predicted_idx.item()
                confidence = max_prob.item()
                
                if confidence >= self.threshold:
                    name = self.class_mapping.get(str(predicted_idx), f"Person_{predicted_idx}")
                else:
                    name = "Unknown"
                
                end_time = time.time()
                self.recognition_times.append(end_time - start_time)
                
                return name, confidence
            
            except Exception as e:
                print(f"Error recognizing face: {str(e)}")
                return "Error", 0.0
        
        elif self.method == "face_recognition":
            try:
                if len(self.known_face_encodings) == 0:
                    self.unknown_counter += 1
                    return f"Unknown_{self.unknown_counter}", 0.0
                
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) != 1:
                    self.unknown_counter += 1
                    return f"Unknown_{self.unknown_counter}", 0.0
                
                face_encoding = face_encodings[0]
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                best_match_idx = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_idx]
                
                
                confidence = 1 - best_match_distance
                
                if confidence >= self.threshold:
                    name = self.known_face_names[best_match_idx]
                else:
                    self.unknown_counter += 1
                    name = f"Unknown_{self.unknown_counter}"
                
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
        
        if detector is None:
            detector = FaceDetector(method="dnn", min_confidence=0.5)
        self.detector = detector
        
        if recognizer is None:
            recognizer = FaceRecognizer(method="face_recognition", threshold=0.6)
        self.recognizer = recognizer
        
        self.track_faces = track_faces
        if track_faces:
            self.tracker = FaceTracker(max_disappeared=50, min_detection_confidence=0.5)
        
        self.recognition_interval = recognition_interval
        self.frame_count = 0
        
        self.face_identities = {}  
        self.face_confidences = {}  
        self.face_timestamps = {}  
        
        self.recognition_count = 0
        self.known_count = 0
        self.unknown_count = 0
    
    def process_frame(self, frame):
        
        self.frame_count += 1
        current_time = time.time()
        
        face_detections = self.detector.detect_faces(frame)
        
        if self.track_faces:
            tracked_faces = self.tracker.update(face_detections)
            face_bboxes = {face_id: bbox for face_id, bbox in tracked_faces.items()}
        else:
            face_bboxes = {i: detection[:4] for i, detection in enumerate(face_detections)}
        
        run_recognition = (self.frame_count % self.recognition_interval == 0)
        
        faces_info = []
        for face_id, bbox in face_bboxes.items():
            x, y, w, h = bbox
            
            if face_id in self.face_identities:
                name = self.face_identities[face_id]
                confidence = self.face_confidences[face_id]
                last_recognition = self.face_timestamps[face_id]
                
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
        
        processed_frame = self.draw_recognition_results(frame, faces_info)
        
        return processed_frame, faces_info
    
    def draw_recognition_results(self, frame, faces_info):
        
        result = frame.copy()
        
        for bbox, name, confidence in faces_info:
            x, y, w, h = bbox
            
            if name.startswith("Unknown"):
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            label = f"{name} ({confidence:.2f})"
            
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
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.frame_count = 0
        self.recognition_count = 0
        self.known_count = 0
        self.unknown_count = 0
        
        recognition_results = {}
        
        try:
            while cap.isOpened() and (max_frames is None or self.frame_count < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, faces_info = self.process_frame(frame)
                
                recognition_results[self.frame_count] = faces_info
                
                if output_path:
                    out.write(processed_frame)
                
                if display:
                    cv2.imshow('Face Recognition', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if self.frame_count % 100 == 0:
                    progress = self.frame_count / total_frames * 100
                    print(f"Processing: {progress:.1f}% complete ({self.frame_count}/{total_frames})")
        
        finally:
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