from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import json
import sys
import os

# Add model directory to path
sys.path.insert(0, os.path.abspath('../model'))
from model import MNISTModel

app = FastAPI(title="MNIST Digit Classification API with PyTorch")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = MNISTModel()
        model.load_state_dict(torch.load('/app/model/mnist_pytorch.pt', map_location=torch.device('cpu')))
        model.eval()
        print("PyTorch model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "MNIST Digit Classification API with PyTorch"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read and process the image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')  # Convert to grayscale
        
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        return {
            "predicted_digit": int(predicted_class),
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")