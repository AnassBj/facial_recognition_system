// main.js - JavaScript for the index page

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadSpinner = document.getElementById('upload-spinner');
    const errorMessage = document.getElementById('error-message');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadProgressBar = document.getElementById('upload-progress-bar');
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const videoInput = document.getElementById('video-file');
        
        // Validate file selection
        if (!videoInput.files || videoInput.files.length === 0) {
            showError('Please select a video file to upload.');
            return;
        }
        
        const file = videoInput.files[0];
        
        // Validate file type
        const validTypes = ['.mp4', '.avi', '.mov', '.mkv', '.wmv'];
        const fileExt = file.name.substr(file.name.lastIndexOf('.')).toLowerCase();
        if (!validTypes.includes(fileExt)) {
            showError('Please select a valid video file (MP4, AVI, MOV, MKV, WMV).');
            return;
        }
        
        // Validate file size (max 500MB)
        const maxSize = 500 * 1024 * 1024; // 500MB in bytes
        if (file.size > maxSize) {
            showError('File size is too large. Maximum allowed size is 500MB.');
            return;
        }
        
        // Create FormData object
        const formData = new FormData();
        formData.append('video', file);
        
        // Show upload progress
        uploadBtn.disabled = true;
        uploadSpinner.classList.remove('d-none');
        errorMessage.classList.add('d-none');
        uploadProgress.classList.remove('d-none');
        
        // Send upload request with progress tracking
        const xhr = new XMLHttpRequest();
        
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                uploadProgressBar.style.width = percentComplete + '%';
                uploadProgressBar.textContent = percentComplete + '%';
            }
        });
        
        xhr.addEventListener('load', function() {
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.success && response.redirect) {
                        window.location.href = response.redirect;
                    } else {
                        showError(response.error || 'An unknown error occurred.');
                        resetUploadForm();
                    }
                } catch (e) {
                    showError('Failed to parse server response.');
                    resetUploadForm();
                }
            } else {
                showError('Upload failed: ' + xhr.statusText);
                resetUploadForm();
            }
        });
        
        xhr.addEventListener('error', function() {
            showError('A network error occurred during upload.');
            resetUploadForm();
        });
        
        xhr.addEventListener('abort', function() {
            showError('Upload was aborted.');
            resetUploadForm();
        });
        
        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    });
    
    // Helper function to show error messages
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
    }
    
    // Helper function to reset the upload form
    function resetUploadForm() {
        uploadBtn.disabled = false;
        uploadSpinner.classList.add('d-none');
        uploadProgress.classList.add('d-none');
        uploadProgressBar.style.width = '0%';
        uploadProgressBar.textContent = '';
    }
});