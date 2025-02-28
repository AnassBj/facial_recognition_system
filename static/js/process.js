// process.js - JavaScript for the process page

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const processBtn = document.getElementById('process-btn');
    const processSpinner = document.getElementById('process-spinner');
    const processingStatus = document.getElementById('processing-status');
    const errorMessage = document.getElementById('error-message');
    const resultsSection = document.getElementById('results-section');
    const statsTotal = document.getElementById('stats-total');
    const statsKnown = document.getElementById('stats-known');
    const statsUnknown = document.getElementById('stats-unknown');
    const viewVideoBtn = document.getElementById('view-video-btn');
    const downloadVideoBtn = document.getElementById('download-video-btn');
    const processedVideo = document.getElementById('processed-video');
    const videoModal = new bootstrap.Modal(document.getElementById('videoModal'));
    const faceCards = document.querySelectorAll('.face-card');
    const saveFaceBtns = document.querySelectorAll('.save-face-btn');
    
    // Initialize face cards
    faceCards.forEach(card => {
        const faceInput = card.querySelector('.face-name-input');
        const faceId = card.dataset.faceId;
        
        if (faceInput.value && !faceInput.value.startsWith('Unknown')) {
            card.classList.add('is-known');
            card.classList.remove('is-unknown');
        } else {
            card.classList.add('is-unknown');
            card.classList.remove('is-known');
        }
    });
    
    // Handle save face button clicks
    saveFaceBtns.forEach(btn => {
        btn.addEventListener('click', function(e) {
            const card = this.closest('.face-card');
            const faceId = card.dataset.faceId;
            const nameInput = card.querySelector('.face-name-input');
            const name = nameInput.value.trim();
            
            if (!name) {
                nameInput.value = `Unknown_${faceId}`;
                card.classList.add('is-unknown');
                card.classList.remove('is-known');
                return;
            }
            
            // Disable button and show loading state
            this.disabled = true;
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';
            
            // Send identification request
            fetch('/identify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    face_id: faceId,
                    name: name
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update card styling
                    card.classList.add('is-known');
                    card.classList.remove('is-unknown');
                    
                    // Show success feedback
                    this.innerHTML = 'Saved ✓';
                    setTimeout(() => {
                        this.innerHTML = 'Save';
                        this.disabled = false;
                    }, 2000);
                } else {
                    showError(data.error || 'Failed to save identification');
                    this.innerHTML = 'Save';
                    this.disabled = false;
                }
            })
            .catch(error => {
                showError('Network error: ' + error.message);
                this.innerHTML = 'Save';
                this.disabled = false;
            });
        });
    });
    
    // Handle process video button click
    processBtn.addEventListener('click', function() {
        // Show loading state
        processBtn.disabled = true;
        processSpinner.classList.remove('d-none');
        processingStatus.classList.remove('d-none');
        errorMessage.classList.add('d-none');
        
        // Send processing request
        fetch('/process-video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        })
        .then(response => response.json())
        .then(data => {
            processBtn.disabled = false;
            processSpinner.classList.add('d-none');
            processingStatus.classList.add('d-none');
            
            if (data.success) {
                // Update statistics
                if (data.stats) {
                    statsTotal.textContent = data.stats.recognized_faces || 0;
                    statsKnown.textContent = data.stats.known_faces || 0;
                    statsUnknown.textContent = data.stats.unknown_faces || 0;
                }
                
                // Configure video links
                if (data.output_path) {
                    processedVideo.src = data.output_path;
                    viewVideoBtn.href = '#videoModal';
                    viewVideoBtn.setAttribute('data-bs-toggle', 'modal');
                    downloadVideoBtn.href = data.output_path;
                }
                
                // Show results section
                resultsSection.classList.remove('d-none');
            } else {
                showError(data.error || 'An unknown error occurred during processing.');
            }
        })
        .catch(error => {
            processBtn.disabled = false;
            processSpinner.classList.add('d-none');
            processingStatus.classList.add('d-none');
            showError('Network error: ' + error.message);
        });
    });
    
    // Handle video modal events
    document.getElementById('videoModal').addEventListener('hidden.bs.modal', function () {
        // Pause video when modal is closed
        processedVideo.pause();
    });
    
    // Check if video was already processed
    checkSessionStatus();
    
    // Setup cleanup timer (remove old sessions every 5 minutes)
    setInterval(cleanupSessions, 5 * 60 * 1000);
    
    // Helper function to show error messages
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
    }
    
    // Function to check session status
    function checkSessionStatus() {
        fetch(`/session/${sessionId}/status`)
            .then(response => response.json())
            .then(data => {
                if (data.processed && data.output_path) {
                    // Video already processed, update UI
                    processedVideo.src = data.output_path;
                    viewVideoBtn.href = '#videoModal';
                    viewVideoBtn.setAttribute('data-bs-toggle', 'modal');
                    downloadVideoBtn.href = data.output_path;
                    
                    // Show results section
                    resultsSection.classList.remove('d-none');
                    
                    // Disable process button
                    processBtn.disabled = true;
                    processBtn.textContent = 'Already Processed ✓';
                }
            })
            .catch(error => {
                console.error('Error checking session status:', error);
            });
    }
    
    // Function to clean up old sessions
    function cleanupSessions() {
        fetch('/cleanup-sessions', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log(`Cleaned up ${data.removed_sessions} expired sessions`);
        })
        .catch(error => {
            console.error('Error cleaning up sessions:', error);
        });
    }
});