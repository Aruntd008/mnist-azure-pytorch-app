import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTModel
import mlflow
import mlflow.pytorch

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training in Azure ML')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='directory to save the model to')
    parser.add_argument('--registered-model-name', type=str, default='mnist-pytorch',
                        help='name for the registered model')
    
    args = parser.parse_args()
    
    # Print arguments for logging
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # MNIST dataset transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Training dataset
    train_dataset = datasets.MNIST(root='./data', 
                                  train=True, 
                                  transform=transform,
                                  download=True)
    
    # Test dataset
    test_dataset = datasets.MNIST(root='./data', 
                                 train=False, 
                                 transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.test_batch_size, 
                            shuffle=False)
    
    # Initialize model
    model = MNISTModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    # Enable MLflow autologging
    mlflow.pytorch.autolog()
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("momentum", args.momentum)
        mlflow.log_param("model_type", "CNN")
        mlflow.log_metric("num_samples", len(train_dataset))
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Log training progress
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    
                    # Log metrics to MLflow
                    mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_loader) + batch_idx)
            
            # Test
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    # Sum up batch loss
                    test_loss += criterion(output, target).item()
                    
                    # Get the index of the max log-probability
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Average test loss and accuracy
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), accuracy))
            
            # Log metrics to MLflow
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
        
        # Define conda environment for model deployment
        conda_env = {
            'name': 'mnist-pytorch-env',
            'channels': ['conda-forge'],
            'dependencies': [
                'python=3.10',
                {
                    'pip': [
                        'mlflow==2.17.0',
                        'torch>=2.0.0',
                        'torchvision>=0.15.0',
                        'numpy>=1.24.0',
                        'pillow>=9.5.0',
                        'scikit-learn>=1.0.0',
                        'azureml-inference-server-http>=0.7.0',  # Add this package
                    ]
                }
            ],
        }
        
        # Save model checkpoint with PyTorch native format
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'mnist_pytorch.pt'))
        print(f"Model checkpoint saved to {os.path.join(args.output_dir, 'mnist_pytorch.pt')}")
        
        # Register model with MLflow
        print(f"Registering the model via MLflow as {args.registered_model_name}")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=args.registered_model_name,
            conda_env=conda_env,
            registered_model_name=args.registered_model_name
        )
        
        # Also save the model with MLflow format
        mlflow.pytorch.save_model(
            pytorch_model=model,
            path=os.path.join(args.output_dir, "mlflow_model")
        )
        print(f"MLflow model saved to {os.path.join(args.output_dir, 'mlflow_model')}")
    
    # End MLflow run
    mlflow.end_run()

if __name__ == '__main__':
    main()