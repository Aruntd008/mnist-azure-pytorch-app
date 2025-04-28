import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
from model import MNISTModel

# Initialize the model
def init():
    global model
    
    # Load the model
    model = MNISTModel()
    model.load_state_dict(torch.load('model/mnist_pytorch.pt', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully")

# Run inference on input data
def run(raw_data):
    try:
        # Get the input data as a Pillow image
        img_data = json.loads(raw_data)['data']
        img_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(img_bytes)).convert('L')
        
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
            probabilities = F.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Return the result
        result = {
            'predicted_digit': predicted_class,
            'confidence': confidence
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})