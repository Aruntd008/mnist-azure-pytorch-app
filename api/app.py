from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MNIST Digit Classification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "MNIST Digit Classification API - Proxy to Azure ML"}

@app.get("/health")
def health_check():
    azure_ml_url = os.environ.get("AZURE_ML_ENDPOINT", "")
    return {
        "status": "healthy",
        "azure_ml_endpoint_configured": bool(azure_ml_url)
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Get Azure ML endpoint URL from environment variable
        azure_ml_url = os.environ.get("AZURE_ML_ENDPOINT", "")
        if not azure_ml_url:
            logger.error("Azure ML endpoint not configured")
            raise HTTPException(status_code=500, detail="Azure ML endpoint not configured")
        
        logger.info(f"Using Azure ML endpoint: {azure_ml_url}")
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')  # Convert to grayscale
        
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Convert tensor to list for JSON serialization
        # The AzureML endpoint expects a specific format with a "data" key
        input_data = {"data": image_tensor.reshape(-1).tolist()}
        
        logger.info("Sending request to Azure ML endpoint")
        
        # Forward to Azure ML endpoint
        response = requests.post(
            azure_ml_url,
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(f"Azure ML response error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Error from ML service: {response.text}")
        
        # Parse the response
        result = response.json()
        logger.info(f"Received response from Azure ML: {result}")
        
        # Extract the prediction
        # Assuming the Azure ML endpoint returns {"predictions": [digit]} format
        if "predictions" in result:
            predicted_class = result["predictions"][0]
            
            # We don't get confidence scores from basic predictions
            # So we'll return a placeholder or calculate softmax if available
            confidence = 0.95  # Placeholder confidence
            
            return {
                "predicted_digit": int(predicted_class),
                "confidence": float(confidence)
            }
        else:
            logger.error(f"Unexpected response format: {result}")
            raise HTTPException(status_code=500, detail=f"Unexpected response format: {result}")
            
    except Exception as e:
        logger.exception(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")