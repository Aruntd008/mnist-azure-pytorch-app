import os
import torch
import glob
import json
import mlflow.pyfunc
from model import MNISTModel  # Still importing the model class for reference


model = None

def init():
    global model
    
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir(os.getcwd())}")
    
    # Get the Azure ML model directory
    model_dir = os.getenv('AZUREML_MODEL_DIR', '')
    print(f"AZUREML_MODEL_DIR: {model_dir}")
    
    if model_dir:
        # List all contents recursively to help with debugging
        print("Listing all files in model directory:")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                print(f"  {os.path.join(root, file)}")
    
        try:
            # First try to load using MLflow (preferred method with registered models)
            print("Attempting to load model using MLflow...")
            
            # Find the MLmodel file (should be in the model directory)
            mlmodel_paths = glob.glob(os.path.join(model_dir, "**/MLmodel"), recursive=True)
            
            if mlmodel_paths:
                # Get the directory containing the MLmodel file
                mlflow_model_dir = os.path.dirname(mlmodel_paths[0])
                print(f"Found MLmodel at: {mlmodel_paths[0]}")
                
                # Load the model using MLflow's API
                model = mlflow.pytorch.load_model(mlflow_model_dir)
                print("Successfully loaded model with MLflow")
                
                # Set the model for inference
                if hasattr(model, 'eval'):
                    model.eval()
                    print("Model set to evaluation mode")
                
                return
            else:
                print("No MLmodel file found, will try alternative loading methods")
                
        except Exception as e:
            print(f"Error loading with MLflow: {e}, trying alternative methods")
    
        # Fallback: try to load PyTorch model directly
        try:
            # Try to find model.pth file which is commonly used in MLflow
            model_paths = glob.glob(os.path.join(model_dir, "**/model.pth"), recursive=True)
            if model_paths:
                model_path = model_paths[0]
                print(f"Found model file at: {model_path}")
                
                # Create a new model instance
                pytorch_model = MNISTModel()
                
                # Load the state dictionary
                pytorch_model.load_state_dict(torch.load(
                    model_path,
                    map_location=torch.device("cpu")
                ))
                
                # Set to evaluation mode
                pytorch_model.eval()
                
                # Set the global model
                model = pytorch_model
                print("Model loaded directly with PyTorch and set to eval mode")
                return
            
            # As a last resort, try the original file name
            original_model_file = "mnist_pytorch.pt"
            original_paths = glob.glob(os.path.join(model_dir, f"**/{original_model_file}"), recursive=True)
            
            if original_paths:
                model_path = original_paths[0]
                print(f"Found original model at {model_path}")
                
                # Create a new model instance
                pytorch_model = MNISTModel()
                
                # Load the state dictionary
                pytorch_model.load_state_dict(torch.load(
                    model_path,
                    map_location=torch.device("cpu")
                ))
                
                # Set to evaluation mode
                pytorch_model.eval()
                
                # Set the global model
                model = pytorch_model
                print("Model loaded directly with PyTorch and set to eval mode")
                return
                
        except Exception as e:
            print(f"Error in fallback loading: {e}")
    
    # If we get here, we couldn't load the model
    raise FileNotFoundError("Model could not be loaded from any known location or method.")


def run(input_data):
    global model
    
    if model is None:
        return {"error": "Model not loaded correctly"}
    
    try:
        # Parse input data
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
        
        # Assume input is a dict with key "data", shape (batch, 784)
        import numpy as np
        data = input_data.get("data")
        
        # Check if we're using an MLflow loaded model or PyTorch model
        if hasattr(model, 'predict'):
            # MLflow model - use predict method
            predictions = model.predict(data)
            return {"predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions}
        else:
            # PyTorch model - use forward
            inputs = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
            return {"predictions": predicted.tolist()}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}