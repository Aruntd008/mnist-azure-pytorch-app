import json
import os
import logging
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import sys

# Set up enhanced logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print environment information for debugging
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"AZUREML_MODEL_DIR: {os.getenv('AZUREML_MODEL_DIR', 'Not set')}")
if os.getenv('AZUREML_MODEL_DIR'):
    logger.info(f"Model directory contents: {os.listdir(os.getenv('AZUREML_MODEL_DIR'))}")

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    logger.info(f"Added {current_dir} to sys.path")

# Import the model class
try:
    from model import MNISTModel
    logger.info("Successfully imported MNISTModel")
except ImportError as e:
    logger.error(f"Error importing MNISTModel: {e}")
    # If the model module can't be found in the current directory,
    # define the model class here as a fallback
    import torch.nn as nn
    
    class MNISTModel(nn.Module):
        def __init__(self):
            super(MNISTModel, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Pooling layer
            self.pool = nn.MaxPool2d(2, 2)
            
            # Fully connected layers
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            
            # Dropout layer to prevent overfitting
            self.dropout = nn.Dropout(0.25)
            
        def forward(self, x):
            # First convolution block
            x = self.pool(F.relu(self.conv1(x)))
            
            # Second convolution block
            x = self.pool(F.relu(self.conv2(x)))
            
            # Reshape for fully connected layers
            x = x.view(-1, 64 * 7 * 7)
            
            # Fully connected layers with dropout
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            
            return x
    
    logger.info("Created backup MNISTModel class")

# Initialize the model
def init():
    global model
    
    try:
        # Check for the model file in multiple possible locations
        possible_locations = [
            'mnist_pytorch.pt',
            'model/mnist_pytorch.pt',
            os.path.join(os.getenv('AZUREML_MODEL_DIR', ''), 'mnist_pytorch.pt'),
            os.path.join(os.getenv('AZUREML_MODEL_DIR', ''), 'model/mnist_pytorch.pt')
        ]
        
        # If AZUREML_MODEL_DIR is set, check for all files in that directory
        if os.getenv('AZUREML_MODEL_DIR'):
            model_dir = os.getenv('AZUREML_MODEL_DIR')
            logger.info(f"Listing all files in AZUREML_MODEL_DIR and subdirectories:")
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    logger.info(f"  {full_path}")
                    if file.endswith('.pt') or file.endswith('.pth') or 'model' in file:
                        possible_locations.append(full_path)
        
        # Log all possible locations and whether they exist
        for loc in possible_locations:
            logger.info(f"Checking model location: {loc}, exists: {os.path.exists(loc)}")
        
        model_path = None
        for loc in possible_locations:
            if os.path.exists(loc):
                model_path = loc
                logger.info(f"Found model at {model_path}")
                break
        
        if model_path is None:
            # If we can't find the state dict file, look for MLflow saved model
            mlflow_model_dir = os.getenv('AZUREML_MODEL_DIR', '')
            if os.path.exists(os.path.join(mlflow_model_dir, 'MLmodel')):
                import mlflow.pytorch
                logger.info(f"Attempting to load MLflow model from {mlflow_model_dir}")
                model = mlflow.pytorch.load_model(mlflow_model_dir)
                logger.info(f"Loaded MLflow model from {mlflow_model_dir}")
                return
            else:
                error_msg = f"Model file not found in any of these locations: {possible_locations}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Load the model
        logger.info(f"Initializing MNISTModel")
        model = MNISTModel()
        logger.info(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Run inference on input data
def run(raw_data):
    try:
        logger.info("Starting inference")
        # Parse the input data
        inputs = json.loads(raw_data)
        logger.info(f"Input keys: {inputs.keys()}")
        
        # Handle different input formats
        if 'data' in inputs:
            # Base64 encoded image
            img_bytes = base64.b64decode(inputs['data'])
            image = Image.open(io.BytesIO(img_bytes)).convert('L')
            logger.info("Successfully decoded base64 image")
        elif 'url' in inputs:
            # URL to an image (not implemented in this example)
            logger.error("URL-based inference not implemented")
            raise NotImplementedError("URL-based inference not implemented")
        else:
            logger.error(f"Invalid input format. Available keys: {list(inputs.keys())}")
            raise ValueError("Input must contain either 'data' (base64 encoded image) or 'url'")
        
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        logger.info("Image preprocessed successfully")
        
        # Get prediction
        with torch.no_grad():
            logger.info("Running model inference")
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        logger.info(f"Prediction complete: digit={predicted_class}, confidence={confidence:.4f}")
        
        # Return the result
        result = {
            'predicted_digit': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist()
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return json.dumps({"error": str(e)})