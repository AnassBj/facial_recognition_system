import cv2
import numpy as np
import os
import torch
import time
from collections import defaultdict

class FaceDetector:
    """Class for detecting faces in images/video frames."""
    
    def __init__(self, method="dnn", min_confidence=0.5, scale_factor=1.0):
        
        self.method = method
        self.min_confidence = min_confidence
        self.scale_factor = scale_factor
        
        if method == "dnn":
            # Load the pre-trained DNN face detector
            self.model_file = "models/opencv_face_detector_uint8.pb"
            self.config_file = "models/opencv_face_detector.pbtxt"
            
            # Check if the model files exist, if not, download them
            if not os.path.exists(self.model_file) or not os.path.exists(self.config_file):
                os.makedirs("models", exist_ok=True)
                print("Downloading face detection model files...")
                
                # Download the model files from GitHub
                import urllib.request
                model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb"
                config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
                
                urllib.request.urlretrieve(model_url, self.model_file)
                urllib.request.urlretrieve(config_url, self.config_file)
            
            self.net = cv2.dnn.readNetFromTensorflow(self.model_file, self.config_file)
            
            # Use CUDA if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA for face detection")
            else:
                print("CUDA not available, using CPU for face detection")
        
        elif method == "haar":
            # Load the Haar cascade classifier
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        elif method == "hog":
            # Use dlib's HOG-based face detector (imported when needed)
            import dlib
            self.hog_detector = dlib.get_frontal_face_detector()
        
        else:
            raise ValueError(f"Unsupported face detection method: {method}")
        
        # Track the performance
        self.detection_times = []
    
    def detect_faces(self, image):
        
        start_time = time.time()
        faces = []
        
        # Make a copy of the image to avoid modifying the original
        img = image.copy()
        
        # Resize image if scale factor is not 1.0
        if self.scale_factor != 1.0:
            height, width = img.shape[:2]
            img = cv2.resize(img, (int(width * self.scale_factor), int(height * self.scale_factor)))
        
        if self.method == "dnn":
            # Prepare the image for the DNN model
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            
            # Pass the blob through the network
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Process the detections
            height, width = img.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.min_confidence:
                    # Get the bounding box coordinates
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # Convert to (x, y, w, h) format
                    x = max(0, x1)
                    y = max(0, y1)
                    w = min(width - x, x2 - x1)
                    h = min(height - y, y2 - y1)
                    
                    # Adjust for scaling
                    if self.scale_factor != 1.0:
                        x = int(x / self.scale_factor)
                        y = int(y / self.scale_factor)
                        w = int(w / self.scale_factor)
                        h = int(h / self.scale_factor)
                    
                    faces.append((x, y, w, h, confidence))
        
        elif self.method == "haar":
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rect = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Convert to list of (x, y, w, h, confidence) tuples
            for (x, y, w, h) in faces_rect:
                # Adjust for scaling
                if self.scale_factor != 1.0:
                    x = int(x / self.scale_factor)
                    y = int(y / self.scale_factor)
                    w = int(w / self.scale_factor)
                    h = int(h / self.scale_factor)
                
                faces.append((x, y, w, h, 1.0))  # Haar doesn't provide confidence scores
        
        elif self.method == "hog":
            # Convert to RGB (dlib uses RGB images)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            dlib_faces = self.hog_detector(rgb_img, 1)
            
            # Convert to list of (x, y, w, h, confidence) tuples
            for face in dlib_faces:
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()
                
                # Adjust for scaling
                if self.scale_factor != 1.0:
                    x = int(x / self.scale_factor)
                    y = int(y / self.scale_factor)
                    w = int(w / self.scale_factor)
                    h = int(h / self.scale_factor)
                
                faces.append((x, y, w, h, 1.0))  # HOG doesn't provide confidence scores
        
        # Record the detection time
        end_time = time.time()
        self.detection_times.append(end_time - start_time)
        
        # If we have more than 100 records, keep only the most recent 100
        if len(self.detection_times) > 100:
            self.detection_times = self.detection_times[-100:]
        
        return faces
    
    def get_average_detection_time(self):
        if not self.detection_times:
            return 0
        return sum(self.detection_times) / len(self.detection_times) * 1000
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2, show_confidence=True):
        
        img = image.copy()
        
        for face in faces:
            x, y, w, h = face[:4]
            confidence = face[4] if len(face) > 4 else None
            
            # Draw the bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Display confidence if available and requested
            if confidence is not None and show_confidence:
                label = f"{confidence:.2f}"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img

class FaceTracker:
    
    def __init__(self, max_disappeared=50, min_detection_confidence=0.5):
        
        self.next_face_id = 0
        self.faces = {}  # Dictionary mapping face IDs to centroids
        self.disappeared = defaultdict(int)  # Number of frames a face has disappeared
        self.max_disappeared = max_disappeared
        self.min_detection_confidence = min_detection_confidence
        self.face_bboxes = {}  # Dictionary mapping face IDs to bounding boxes
    
    def register(self, centroid, bbox):
        self.faces[self.next_face_id] = centroid
        self.face_bboxes[self.next_face_id] = bbox
        self.disappeared[self.next_face_id] = 0
        self.next_face_id += 1
    
    def deregister(self, face_id):
        del self.faces[face_id]
        del self.face_bboxes[face_id]
        del self.disappeared[face_id]
    
    def update(self, detections):
        
        # Filter detections based on confidence
        detections = [d for d in detections if len(d) <= 4 or d[4] >= self.min_detection_confidence]
        
        # If no faces were detected, increment disappeared counters and remove faces if necessary
        if len(detections) == 0:
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            
            return self.face_bboxes
        
        # If we're currently not tracking any faces, register all detections
        if len(self.faces) == 0:
            for detection in detections:
                x, y, w, h = detection[:4]
                centroid = (x + w // 2, y + h // 2)
                self.register(centroid, detection[:4])
        
        # Otherwise, find the closest detection to each tracked face
        else:
            # Get centroids of current faces
            face_ids = list(self.faces.keys())
            face_centroids = list(self.faces.values())
            
            # Get centroids of new detections
            new_centroids = []
            for detection in detections:
                x, y, w, h = detection[:4]
                centroid = (x + w // 2, y + h // 2)
                new_centroids.append(centroid)
            
            # Calculate distance between each pair of current and new centroids
            distances = np.zeros((len(face_centroids), len(new_centroids)))
            for i, face_centroid in enumerate(face_centroids):
                for j, new_centroid in enumerate(new_centroids):
                    distances[i, j] = np.sqrt(
                        (face_centroid[0] - new_centroid[0]) ** 2 +
                        (face_centroid[1] - new_centroid[1]) ** 2
                    )
            
            # Find the closest detection for each face
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            # Keep track of used rows and columns
            used_rows = set()
            used_cols = set()
            
            # Update the tracked faces
            for row, col in zip(rows, cols):
                # If this detection has already been used, skip it
                if row in used_rows or col in used_cols:
                    continue
                
                # Get the face ID and update its centroid and bbox
                face_id = face_ids[row]
                self.faces[face_id] = new_centroids[col]
                self.face_bboxes[face_id] = detections[col][:4]
                self.disappeared[face_id] = 0
                
                # Mark this row and column as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Check for unused rows (faces that didn't get assigned a detection)
            unused_rows = set(range(distances.shape[0])) - used_rows
            for row in unused_rows:
                face_id = face_ids[row]
                self.disappeared[face_id] += 1
                
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            
            # Check for unused columns (detections that didn't get assigned to a face)
            unused_cols = set(range(distances.shape[1])) - used_cols
            for col in unused_cols:
                centroid = new_centroids[col]
                self.register(centroid, detections[col][:4])
        
        return self.face_bboxes

    def get_tracked_face_count(self):
        """Get the number of currently tracked faces."""
        return len(self.faces)

class VideoFaceDetector:
    
    
    def __init__(self, detector_method="dnn", min_confidence=0.5, use_tracking=True,
                 track_timeout=30, scale_factor=1.0, detect_interval=1):
        
        self.detector = FaceDetector(method=detector_method, min_confidence=min_confidence,
                                    scale_factor=scale_factor)
        self.use_tracking = use_tracking
        self.tracker = FaceTracker(max_disappeared=track_timeout,
                                  min_detection_confidence=min_confidence) if use_tracking else None
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.detected_faces = {}  # Map of frame indices to detected faces
        self.detection_speed = 0  # Average detection speed (ms)
    
    def process_frame(self, frame):
        
        self.frame_count += 1
        faces = []
        
        # Run detection at intervals or if not using tracking
        if not self.use_tracking or self.frame_count % self.detect_interval == 0:
            # Detect faces
            faces = self.detector.detect_faces(frame)
            self.detection_speed = self.detector.get_average_detection_time()
            
            # Update the tracker if using tracking
            if self.use_tracking:
                self.tracked_faces = self.tracker.update(faces)
                faces = [(x, y, w, h, 1.0) for (x, y, w, h) in self.tracked_faces.values()]
        
        # Get faces from tracker for frames between detections
        elif self.use_tracking:
            self.tracked_faces = self.tracker.update([])
            faces = [(x, y, w, h, 1.0) for (x, y, w, h) in self.tracked_faces.values()]
        
        # Save the detected faces for this frame
        self.detected_faces[self.frame_count] = faces
        
        # Draw faces on the frame
        processed_frame = frame.copy()
        for face in faces:
            x, y, w, h = face[:4]
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Add detection information
        cv2.putText(processed_frame, f"Detected Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Detection Speed: {self.detection_speed:.1f} ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return processed_frame, faces
    
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset frame count and detected faces
        self.frame_count = 0
        self.detected_faces = {}
        
        # Process frames
        try:
            while cap.isOpened() and (max_frames is None or self.frame_count < max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame
                processed_frame, faces = self.process_frame(frame)
                
                # Write to output video
                if output_path:
                    out.write(processed_frame)
                
                # Display the frame
                if display:
                    cv2.imshow('Face Detection', processed_frame)
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
        
        return self.detected_faces

def test_face_detector():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the face detector")
    parser.add_argument("--input", type=str, default="0",
                        help="Path to an image, video file, or camera index (default: 0 for webcam)")
    parser.add_argument("--method", type=str, default="dnn",
                        choices=["dnn", "hog", "haar"],
                        help="Face detection method to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum confidence threshold")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the output (image or video)")
    parser.add_argument("--display", action="store_true",
                        help="Display the frames while processing")
    parser.add_argument("--tracking", action="store_true",
                        help="Use face tracking")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for the input image")
    
    args = parser.parse_args()
    
    # Check if input is an image, video, or camera
    if args.input.isdigit():
        # Use webcam
        cap = cv2.VideoCapture(int(args.input))
        if not cap.isOpened():
            print(f"Failed to open camera: {args.input}")
            return
        
        detector = VideoFaceDetector(
            detector_method=args.method,
            min_confidence=args.confidence,
            use_tracking=args.tracking,
            scale_factor=args.scale
        )
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, faces = detector.process_frame(frame)
                
                cv2.imshow('Face Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video file
        detector = VideoFaceDetector(
            detector_method=args.method,
            min_confidence=args.confidence,
            use_tracking=args.tracking,
            scale_factor=args.scale
        )
        
        detector.process_video(args.input, args.output, display=args.display)
    
    else:
        # Process image file
        image = cv2.imread(args.input)
        if image is None:
            print(f"Failed to load image: {args.input}")
            return
        
        detector = FaceDetector(
            method=args.method,
            min_confidence=args.confidence,
            scale_factor=args.scale
        )
        
        faces = detector.detect_faces(image)
        processed_image = detector.draw_faces(image, faces)
        
        if args.display:
            cv2.imshow('Face Detection', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if args.output:
            cv2.imwrite(args.output, processed_image)

if __name__ == "__main__":
    test_face_detector()