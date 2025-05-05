from torchvision import datasets, transforms
import json

# Load MNIST test set
mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
image, label = mnist[0]  # get the first test image

# Flatten the image and convert to a list
flat_image = image.view(-1).tolist()

# Save JSON payload
with open("mnist_test.json", "w") as f:
    json.dump({"data": flat_image}, f)

print("Label:", label)  # Optional: to know the ground truth
