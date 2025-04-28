import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        return F.log_softmax(x, dim=1)