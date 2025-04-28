# MNIST Classification with Azure ML Studio and PyTorch

This project demonstrates a complete machine learning workflow for MNIST digit classification using:

- PyTorch for model development
- Azure ML Studio for training and deployment
- JFrog Artifactory for container registry
- Azure Container Instances for model hosting
- Azure App Service for frontend hosting
- GitHub Actions for CI/CD automation

## Project Structure

- `model/`: PyTorch model and Azure ML configuration
- `api/`: FastAPI application for exposing the model
- `frontend/`: Web interface for interacting with the model
- `infrastructure/`: Terraform code for Azure resources
- `.github/workflows/`: CI/CD pipeline configuration

## Getting Started

### Prerequisites

- Azure subscription
- JFrog Artifactory account
- GitHub account
- Visual Studio Code with necessary extensions

### Development Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r model/requirements.txt
   pip install -r api/requirements.txt