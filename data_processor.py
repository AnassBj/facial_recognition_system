import os
import cv2
import numpy as np
import face_recognition
import argparse
import concurrent.futures
from tqdm import tqdm
import shutil
import json
import random

def detect_face(image_path):
    try:
        print(f"Loading image: {image_path}")
        # Load the image
        image = face_recognition.load_image_file(image_path)
        print(f"Image shape: {image.shape}")
        
        # Limit image size to prevent excessive processing time
        if max(image.shape[0], image.shape[1]) > 1500:
            height, width = image.shape[:2]
            scale = 1500 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"Resized image to {new_width}x{new_height}")
        
        # Find face locations with a timeout
        print("Detecting face locations...")
        
        # Use HOG as it's faster than CNN
        face_locations = face_recognition.face_locations(image, model="hog")
        print(f"Found {len(face_locations)} faces")
        
        if not face_locations:
            return None, "No faces detected"
        
        print("Computing face encodings...")
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # If no faces are detected, return None
        if not face_encodings:
            return None, "No faces detected"
        
        # If multiple faces are detected, return None (we want clean data)
        if len(face_encodings) > 1:
            return None, "Multiple faces detected"
        
        # Return the face encoding and location
        return (face_encodings[0], face_locations[0]), None
    
    except Exception as e:
        print(f"Exception in detect_face: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error processing image: {str(e)}"

def crop_face(image_path, face_location, output_path, padding=30):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return False, "Failed to load image"
        
        # Get the face location
        top, right, bottom, left = face_location
        
        # Add padding
        height, width = image.shape[:2]
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(height, bottom + padding)
        right = min(width, right + padding)
        
        # Crop the face
        face_image = image[top:bottom, left:right]
        
        # Resize to a standard size (e.g., 160x160)
        face_image = cv2.resize(face_image, (160, 160))
        
        # Save the cropped face
        cv2.imwrite(output_path, face_image)
        
        return True, output_path
    
    except Exception as e:
        return False, f"Error cropping face: {str(e)}"

def process_image(image_path, output_dir, person_name):
    try:
        print(f"Processing image: {image_path}")
        # Create a filename for the processed image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        # Detect faces
        print("Detecting faces...")
        result, error = detect_face(image_path)
        if result is None:
            print(f"Face detection failed: {error}")
            return False, error
        
        face_encoding, face_location = result
        
        # Crop and save the face
        print("Cropping face...")
        success, result = crop_face(image_path, face_location, output_path)
        if not success:
            print(f"Face cropping failed: {result}")
            return False, result
        
        # Save the face encoding
        encoding_path = os.path.splitext(output_path)[0] + ".npy"
        np.save(encoding_path, face_encoding)
        
        print(f"Successfully processed: {output_path}")
        return True, output_path
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing image: {str(e)}"

def process_person_images(person_dir, output_dir, person_name):
    # Get all image files
    image_files = []
    for root, _, files in os.walk(person_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append(os.path.join(root, file))
    
    # Create output directory
    person_output_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_output_dir, exist_ok=True)
    
    # Process images
    successful_images = 0
    successful_image_paths = []

    for i, image_path in enumerate(tqdm(image_files, desc=f"Processing {person_name}")):
        success, result = process_image(image_path, person_output_dir, person_name)
        if success:
            successful_images += 1
            successful_image_paths.append(result)
    
    # This return statement was incorrectly indented in the original code
    return successful_images, successful_image_paths

def split_dataset(processed_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Create directories for train, val, and test sets
    train_dir = os.path.join(os.path.dirname(processed_dir), "train")
    val_dir = os.path.join(os.path.dirname(processed_dir), "val")
    test_dir = os.path.join(os.path.dirname(processed_dir), "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(processed_dir) 
                   if os.path.isdir(os.path.join(processed_dir, d))]
    
    # Initialize counts
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Process each person
    for person in person_dirs:
        person_dir = os.path.join(processed_dir, person)
        
        # Get all image files (excluding .npy files)
        image_files = [f for f in os.listdir(person_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        # Shuffle the image files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split the image files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Create person directories in the split directories
        os.makedirs(os.path.join(train_dir, person), exist_ok=True)
        os.makedirs(os.path.join(val_dir, person), exist_ok=True)
        os.makedirs(os.path.join(test_dir, person), exist_ok=True)
        
        # Copy images and their encoding files to the split directories
        for file in train_files:
            shutil.copy(os.path.join(person_dir, file), os.path.join(train_dir, person, file))
            npy_file = os.path.splitext(file)[0] + ".npy"
            if os.path.exists(os.path.join(person_dir, npy_file)):
                shutil.copy(os.path.join(person_dir, npy_file), os.path.join(train_dir, person, npy_file))
            train_count += 1
        
        for file in val_files:
            shutil.copy(os.path.join(person_dir, file), os.path.join(val_dir, person, file))
            npy_file = os.path.splitext(file)[0] + ".npy"
            if os.path.exists(os.path.join(person_dir, npy_file)):
                shutil.copy(os.path.join(person_dir, npy_file), os.path.join(val_dir, person, npy_file))
            val_count += 1
        
        for file in test_files:
            shutil.copy(os.path.join(person_dir, file), os.path.join(test_dir, person, file))
            npy_file = os.path.splitext(file)[0] + ".npy"
            if os.path.exists(os.path.join(person_dir, npy_file)):
                shutil.copy(os.path.join(person_dir, npy_file), os.path.join(test_dir, person, npy_file))
            test_count += 1
    
    print(f"Dataset split complete: {train_count} training, {val_count} validation, {test_count} test images")
    
    # Create class mapping
    class_mapping = {i: person for i, person in enumerate(sorted(person_dirs))}
    
    # Save class mapping
    mapping_file = os.path.join(os.path.dirname(processed_dir), "class_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    print(f"Class mapping saved to {mapping_file}")
    
    return train_dir, val_dir, test_dir, mapping_file

def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser(description="Process facial images for training")
    parser.add_argument("--input", type=str, default="data/raw", 
                        help="Input directory with raw images")
    parser.add_argument("--output", type=str, default="data/processed", 
                        help="Output directory for processed images")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio of images to use for training")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Ratio of images to use for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Ratio of images to use for testing")
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(args.input) 
                   if os.path.isdir(os.path.join(args.input, d))]
    
    # Process each person's images
    total_processed = 0
    
    for person in person_dirs:
        person_dir = os.path.join(args.input, person)
        person_name = person
        
        print(f"Processing images for {person_name}...")
        processed_count, _ = process_person_images(person_dir, args.output, person_name)
        total_processed += processed_count
        print(f"Processed {processed_count} images for {person_name}")
    
    print(f"Processing complete. Processed a total of {total_processed} images.")
    
    # Split the dataset
    print("Splitting dataset into train, validation, and test sets...")
    train_dir, val_dir, test_dir, mapping_file = split_dataset(
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    print("Dataset preparation complete.")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")
    print(f"Test data: {test_dir}")
    print(f"Class mapping: {mapping_file}")

if __name__ == "__main__":
    main()