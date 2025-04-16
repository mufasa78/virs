document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const progressContainer = document.getElementById('progress-container');
    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const inputVideo = document.getElementById('input-video');
    const outputVideo = document.getElementById('output-video');
    const downloadBtn = document.getElementById('download-btn');
    const newUploadBtn = document.getElementById('new-upload-btn');
    const errorBackBtn = document.getElementById('error-back-btn');

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form
        const videoInput = document.getElementById('video');
        if (!videoInput.files || videoInput.files.length === 0) {
            showError('Please select a video file to upload.');
            return;
        }
        
        const file = videoInput.files[0];
        const fileSize = file.size / 1024 / 1024; // Convert to MB
        if (fileSize > 500) {
            showError('File size exceeds the maximum limit of 500MB.');
            return;
        }
        
        // Show progress
        uploadForm.style.display = 'none';
        progressContainer.style.display = 'block';
        resultsContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        
        // Create FormData and send request
        const formData = new FormData(uploadForm);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Show results
            progressContainer.style.display = 'none';
            resultsContainer.style.display = 'block';
            
            // Set video sources
            inputVideo.src = data.input_video;
            outputVideo.src = data.output_video;
            
            // Set download link
            downloadBtn.href = data.output_video;
            downloadBtn.download = data.output_video.split('/').pop();
            
            // Load videos
            inputVideo.load();
            outputVideo.load();
        })
        .catch(error => {
            console.error('Error:', error);
            showError('An error occurred while processing your video. Please try again.');
        });
    });
    
    // Handle "Process Another Video" button
    newUploadBtn.addEventListener('click', function() {
        resetForm();
    });
    
    // Handle "Try Again" button in error message
    errorBackBtn.addEventListener('click', function() {
        resetForm();
    });
    
    // Function to show error message
    function showError(message) {
        progressContainer.style.display = 'none';
        resultsContainer.style.display = 'none';
        errorContainer.style.display = 'block';
        errorMessage.textContent = message;
    }
    
    // Function to reset the form
    function resetForm() {
        uploadForm.reset();
        uploadForm.style.display = 'block';
        progressContainer.style.display = 'none';
        resultsContainer.style.display = 'none';
        errorContainer.style.display = 'none';
    }
});
