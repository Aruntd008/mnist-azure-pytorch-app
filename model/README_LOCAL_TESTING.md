# Testing MNIST Model with Local Endpoints

This guide explains how to test your MNIST model using Azure ML local endpoints before deploying to the cloud.

## Prerequisites

- Azure ML SDK v2 installed
- A registered MNIST model in your Azure ML workspace
- Python 3.8+ with required packages (see `environment.yml`)

## Quick Start

The easiest way to test your model locally is to use the example script:

```bash
python test_local_endpoint_example.py --subscription-id <your-subscription-id> --resource-group <your-resource-group> --workspace-name <your-workspace-name>
```

This script will:
1. Generate a test image of the digit "5"
2. Deploy your model to a local endpoint
3. Test the endpoint with the generated image
4. Clean up the local endpoint

## Step-by-Step Guide

### 1. Generate a Test Image

First, generate a test image from the MNIST dataset:

```bash
python generate_test_image.py --digit 5
```

This will create a test image in the `test_images` directory.

### 2. Deploy and Test the Model

Deploy your model to a local endpoint and test it:

```bash
python test_local_endpoint.py --subscription-id <your-subscription-id> --resource-group <your-resource-group> --workspace-name <your-workspace-name> --test-image test_images/mnist_digit_5_0.png
```

### 3. Clean Up

When you're done testing, clean up the local endpoint:

```bash
python test_local_endpoint.py --subscription-id <your-subscription-id> --resource-group <your-resource-group> --workspace-name <your-workspace-name> --cleanup
```

## Advanced Options

The `test_local_endpoint.py` script supports several options:

- `--model-name`: Name of the registered model (default: "mnist-pytorch")
- `--endpoint-name`: Name for the local endpoint (default: "mnist-local-endpoint")
- `--debug`: Enable debug mode with more verbose logging
- `--generate-image`: Generate a test image if none is provided
- `--cleanup`: Clean up the local endpoint

## Troubleshooting

If you encounter issues:

1. Use the `--debug` flag for more detailed logging
2. Check that your model is registered in the Azure ML workspace
3. Ensure your environment has all required dependencies
4. Verify that Docker is running (required for local endpoints)

## Additional Resources

- [Azure ML SDK v2 Documentation](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)
- [Local Endpoints Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-managed-online-endpoints-visual-studio-code)
