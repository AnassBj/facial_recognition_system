## 1. Setting up Python Environment


# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate




## 2. Installing Required Libraries


# Install core dependencies
python -m pip install --upgrade pip   
pip install numpy==1.24.3  

# Install computer vision and machine learning libraries
pip install opencv-python==4.8.0.76
pip install matplotlib==3.7.3
pip install scikit-learn==1.3.2
pip install pillow==10.0.1
pip install tqdm==4.66.1

# Install web scraping libraries
pip install requests==2.31.0
pip install beautifulsoup4==4.12.2
pip install selenium==4.15.2
pip install webdriver_manager==4.0.1

# Install PyTorch with CUDA support 
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install face recognition libraries
pip install dlib
pip install dlib-binary
pip install face-recognition==1.3.0

# Install Flask for web interface
pip install flask==2.3.3
pip install werkzeug==2.3.7

# Install utilities
pip install pandas==2.0.3
```


## 3. Setting up Project Structure

Create the following directory structure for your project:

```
facial_recognition_system/
├── app.py                  # Main web application
├── face_detector.py        # Face detection module
├── face_recognizer.py      # Face recognition module
├── data_collector.py       # Dataset collection script
├── data_processor.py       # Dataset preparation script
├── model_trainer.py        # Model training script
├── utils.py                # Utility functions
├── templates/              # Web interface templates
│   ├── index.html          # Homepage
│   └── process.html        # Results page
├── static/                 # CSS, JS, and other static files
│   ├── css/
│   │   └── style.css       # CSS styles
│   └── js/
│       ├── main.js         # Main page JavaScript
│       └── result.js      # Process page JavaScript
├── uploads/                # Temporary folder for uploaded videos
├── output/                 # Processed videos
├── models/                 # Face detection model files
│   ├── opencv_face_detector_uint8.pb
│   └── opencv_face_detector.pbtxt
└── data/                   # Dataset
    ├── raw/                # Raw scraped images
    ├── processed/          # Processed images
    ├── train/              # Training data
    ├── val/                # Validation data
    ├── test/               # Test data
    └── model/              # Trained model files
```

Create all directories:

# Create all the necessary directories
mkdir -p facial_recognition_system/templates
mkdir -p facial_recognition_system/static/css
mkdir -p facial_recognition_system/static/js
mkdir -p facial_recognition_system/uploads
mkdir -p facial_recognition_system/output
mkdir -p facial_recognition_system/models
mkdir -p facial_recognition_system/data/raw
mkdir -p facial_recognition_system/data/processed
mkdir -p facial_recognition_system/data/train
mkdir -p facial_recognition_system/data/val
mkdir -p facial_recognition_system/data/test
mkdir -p facial_recognition_system/data/model


## 4. Setting up Browser for Web Scraping (for dataset collection)

1. Install Firefox or Chrome browser
   - Firefox: https://www.mozilla.org/firefox/
   - Chrome: https://www.google.com/chrome/

2. The script will automatically download the appropriate driver

## 5. Setting up the Model Files

1. Create a folder named `models` in your facial_recognition_system directory
2. Download these two files for the face detector:
   - OpenCV face detector model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb
   - Config file: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
3. Place them in the `models` folder


## 6. Collecting and Preparing Dataset (Optional)

If you want to use the deep learning approach with your own trained model:


# 1. Collect images for your dataset
python data_collector_getty.py

# 2. Process the dataset (detect faces and prepare for training)
python data_processor.py

# 3. Train the face recognition model
python model_trainer.py


## 7. Running the Application


# Run the Flask web application
python app.py
```

After running the application, open your web browser and navigate to:
http://127.0.0.1:5000

## 10. Using the System

1. Upload a video through the web interface
2. The system will detect faces in the video and show thumbnails
3. For each detected face, you can:
   - Identify the person by entering their name
   - Leave unknown faces unidentified
4. Click "Process Video" to generate the output with color-coded bounding boxes:
   - Green for identified people with their names
   - Red for unknown people
5. View or download the processed video

