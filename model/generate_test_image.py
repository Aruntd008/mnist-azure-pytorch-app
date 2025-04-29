"""
Generate a test MNIST digit image for testing the local endpoint.
This script creates a sample digit image that can be used with test_local_endpoint.py.
"""

import os
import argparse
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate a test MNIST digit image")
    parser.add_argument("--digit", type=int, default=None, help="Specific digit to generate (0-9)")
    parser.add_argument("--output-dir", default="./test_images", help="Directory to save the test image")
    parser.add_argument("--index", type=int, default=0, help="Index of the digit in the test dataset")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Find an image of the specified digit if requested
    if args.digit is not None:
        if not 0 <= args.digit <= 9:
            raise ValueError("Digit must be between 0 and 9")
        
        # Find the first occurrence of the requested digit
        for i, (image, label) in enumerate(test_dataset):
            if label == args.digit:
                args.index = i
                break
    
    # Get the image and label
    image, label = test_dataset[args.index]
    
    # Convert tensor to PIL Image
    image = transforms.ToPILImage()(image)
    
    # Save the image
    output_path = os.path.join(args.output_dir, f"mnist_digit_{label}_{args.index}.png")
    image.save(output_path)
    
    print(f"Generated test image for digit {label}")
    print(f"Saved to: {output_path}")
    print(f"You can use this image with test_local_endpoint.py by adding --test-image {output_path}")

if __name__ == "__main__":
    main()
