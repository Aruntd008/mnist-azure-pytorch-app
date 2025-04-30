import os
import logging
import torch
import torch.serialization as serialization
from model import MNISTModel  # Make sure this import works

logger = logging.getLogger("score")
logging.basicConfig(level=logging.INFO)

model = None

def init():
    global model

    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir(os.getcwd())}")
    logger.info(f"AZUREML_MODEL_DIR: {os.getenv('AZUREML_MODEL_DIR', 'Not set')}")

    # Check possible model locations
    candidate_paths = [
        os.getenv("AZUREML_MODEL_DIR", ""),
        ".",
        "./model",
    ]
    model_file = "mnist_pytorch.pt"
    model_path = None

    for path in candidate_paths:
        full_path = os.path.join(path, model_file)
        logger.info(f"Checking model location: {full_path}, exists: {os.path.exists(full_path)}")
        if os.path.exists(full_path):
            model_path = full_path
            break

    if model_path is None:
        raise FileNotFoundError(f"Model file '{model_file}' not found in any known locations.")

    logger.info(f"Found model at {model_path}")
    logger.info("Unpickling full model...")

    try:
        with serialization.safe_globals([MNISTModel]):
            loaded = torch.load(
                model_path,
                map_location=torch.device("cpu"),
                weights_only=False
            )
        if isinstance(loaded, MNISTModel):
            global model
            model = loaded
            model.eval()
            logger.info("Model unpickled and set to eval mode.")
        else:
            raise RuntimeError(f"Expected MNISTModel, got {type(loaded)}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def run(input_data):
    global model
    try:
        # Assume input is a dict with key "data", shape (batch, 784)
        import numpy as np
        data = input_data.get("data")
        inputs = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
        return {"predictions": predicted.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}
