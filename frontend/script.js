document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');
    const predictButton = document.getElementById('predictButton');
    const predictionElement = document.getElementById('prediction');
    const confidenceElement = document.getElementById('confidence');
    
    // Set up canvas
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    
    // API endpoint (will be updated with actual deployment URL)
    const apiUrl = 'https://your-aci-instance.azurecontainer.io/predict/';
    
    let isDrawing = false;
    
    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Clear the canvas
    clearButton.addEventListener('click', clearCanvas);
    
    // Make prediction
    predictButton.addEventListener('click', predict);
    
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        let x, y;
        
        if (e.type === 'mousemove') {
            x = e.offsetX;
            y = e.offsetY;
        } else if (e.type === 'touchmove' || e.type === 'touchstart') {
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            x = touch.clientX - rect.left;
            y = touch.clientY - rect.top;
        }
        
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y);
        ctx.stroke();
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function handleTouch(e) {
        e.preventDefault();
        if (e.type === 'touchstart') {
            startDrawing(e);
        } else if (e.type === 'touchmove') {
            draw(e);
        }
    }
    
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictionElement.textContent = '-';
        confidenceElement.textContent = '-';
    }
    
    function predict() {
        // Preprocessing - resize to 28x28 internally
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        
        // Convert to blob
        tempCanvas.toBlob(function(blob) {
            const formData = new FormData();
            formData.append('file', blob);
            
            // Show loading state
            predictionElement.textContent = 'Loading...';
            confidenceElement.textContent = '-';
            
            // Send to API
            fetch(apiUrl, {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('API request failed');
                }
                return response.json();
            })
            .then(data => {
                predictionElement.textContent = data.predicted_digit;
                confidenceElement.textContent = (data.confidence * 100).toFixed(2);
            })
            .catch(error => {
                console.error('Error:', error);
                predictionElement.textContent = 'Error';
                confidenceElement.textContent = '-';
            });
        });
    }
});