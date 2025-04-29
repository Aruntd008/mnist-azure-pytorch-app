"""
Test deploying a model to a local endpoint using Azure ML SDK v2.
This script allows you to test your model locally before deploying to the cloud.

Usage:
    python test_local_endpoint.py --subscription-id <subscription_id> --resource-group <resource_group> --workspace-name <workspace_name> [--test-image <path_to_image>]

If no test image is provided, you can generate one using generate_test_image.py:
    python generate_test_image.py --digit 5
"""

import json
import base64
import argparse
import logging
import os
import shutil
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
    Model,
)
from azure.identity import DefaultAzureCredential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Deploy and test a model on a local endpoint")
parser.add_argument("--subscription-id", required=True, help="Azure subscription ID")
parser.add_argument("--resource-group", required=True, help="Azure resource group name")
parser.add_argument("--workspace-name", required=True, help="Azure ML workspace name")
parser.add_argument("--model-name", default="mnist-pytorch", help="Name of the registered model")
parser.add_argument("--endpoint-name", default="mnist-local-endpoint", help="Name for the local endpoint")
parser.add_argument("--test-image", help="Path to a test image file (optional)")
parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose logging")
parser.add_argument("--generate-image", action="store_true", help="Generate a test image if none is provided")
parser.add_argument("--cleanup", action="store_true", help="Clean up the local endpoint if it exists")
args = parser.parse_args()

# Set debug logging if requested
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Initialize the Azure ML client
logger.info("Initializing Azure ML client...")
try:
    # Try to use DefaultAzureCredential which will try different authentication methods
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )
    logger.info("Azure ML client initialized successfully using DefaultAzureCredential")
except Exception as e:
    logger.warning(f"DefaultAzureCredential failed: {e}")
    logger.info("Trying to authenticate using Azure CLI...")

    # Check if Azure CLI is logged in
    import subprocess
    try:
        # Run az account show to check if logged in
        subprocess.run(["az", "account", "show"], check=True, capture_output=True)

        # If we get here, we're logged in, so use AzureCliCredential
        from azure.identity import AzureCliCredential
        credential = AzureCliCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name
        )
        logger.info("Azure ML client initialized successfully using AzureCliCredential")
    except Exception as cli_error:
        logger.error(f"Azure CLI authentication failed: {cli_error}")
        logger.info("Please log in to Azure CLI using 'az login' and try again")
        raise

# Check if cleanup is requested
if args.cleanup:
    logger.info(f"Cleaning up local endpoint '{args.endpoint_name}'...")
    try:
        try:
            # Try the newer API first
            result = ml_client.online_endpoints.begin_delete(name=args.endpoint_name, local=True)
            if hasattr(result, 'result'):
                result.result()  # Wait for completion if it returns an operation
            logger.info(f"Successfully deleted local endpoint '{args.endpoint_name}' using begin_delete")
        except (AttributeError, TypeError):
            # Fall back to the older API
            ml_client.online_endpoints.delete(name=args.endpoint_name, local=True)
            logger.info(f"Successfully deleted local endpoint '{args.endpoint_name}' using delete")
        exit(0)
    except Exception as e:
        logger.error(f"Error deleting endpoint: {e}")
        exit(1)

# Get the latest version of the model
logger.info(f"Retrieving latest version of model '{args.model_name}'...")
model_versions = list(ml_client.models.list(name=args.model_name))
if not model_versions:
    raise Exception(f"No model found with name {args.model_name}")

# Sort by creation time to get the latest
latest_model = sorted(model_versions, key=lambda x: x.creation_context.created_at, reverse=True)[0]
logger.info(f"Using model {latest_model.name} version {latest_model.version}")

# Generate a test image if requested and no test image is provided
if args.generate_image and not args.test_image:
    logger.info("Generating a test image...")
    try:
        from pathlib import Path

        # Create output directory
        current_dir = Path(__file__).parent
        output_dir = current_dir / "test_images"
        output_dir.mkdir(exist_ok=True)

        # Check if we can use the generate_test_image.py script
        generate_script = current_dir / "generate_test_image.py"

        if not generate_script.exists():
            raise FileNotFoundError(f"Could not find generate_test_image.py at {generate_script}")

        # Try to create a simple test image directly without using torch
        # This is a fallback in case torch is not installed
        logger.info("Creating a simple test image directly...")

        # Create a simple 28x28 image with a digit-like pattern
        try:
            from PIL import Image, ImageDraw

            # Create a blank image
            img = Image.new('L', (28, 28), color=0)
            draw = ImageDraw.Draw(img)

            # Draw a simple digit "5" pattern
            draw.rectangle([(5, 5), (20, 7)], fill=255)  # Top horizontal
            draw.rectangle([(5, 5), (7, 14)], fill=255)  # Top-left vertical
            draw.rectangle([(5, 12), (20, 14)], fill=255)  # Middle horizontal
            draw.rectangle([(18, 14), (20, 22)], fill=255)  # Bottom-right vertical
            draw.rectangle([(5, 20), (20, 22)], fill=255)  # Bottom horizontal

            # Save the image
            test_image_path = output_dir / "simple_digit_5.png"
            img.save(test_image_path)

            args.test_image = str(test_image_path)
            logger.info(f"Created simple test image: {args.test_image}")

        except Exception as pil_error:
            logger.warning(f"Could not create simple image with PIL: {pil_error}")

            # If that fails too, try running the generate_test_image.py script
            # First check if torch is installed
            try:
                import subprocess

                # Try to install torch if it's not already installed
                logger.info("Checking if torch is installed...")
                try:
                    # Just try to import torch to see if it's installed
                    __import__('torch')
                    logger.info("torch is already installed")
                except ImportError:
                    logger.info("torch is not installed, trying to install it...")
                    subprocess.run([
                        "pip", "install", "torch", "torchvision", "--quiet"
                    ], check=True)
                    logger.info("torch installed successfully")

                # Now run the generate_test_image.py script
                logger.info("Running generate_test_image.py...")
                subprocess.run([
                    "python",
                    str(generate_script),
                    "--digit", "5",
                    "--output-dir", str(output_dir)
                ], check=True)

                # Set the test image path to the generated image
                for file in output_dir.glob("mnist_digit_*.png"):
                    args.test_image = str(file)
                    logger.info(f"Using generated test image: {args.test_image}")
                    break
            except Exception as script_error:
                logger.error(f"Error running generate_test_image.py: {script_error}")
                raise
    except Exception as e:
        logger.error(f"Error generating test image: {e}")
        logger.error("Continuing without a test image...")

# Create a local endpoint
logger.info(f"Creating local endpoint '{args.endpoint_name}'...")
endpoint = ManagedOnlineEndpoint(
    name=args.endpoint_name,
    description="Local MNIST PyTorch inference endpoint",
    auth_mode="key"
)

# Create or update the endpoint
try:
    # The API changed in newer versions of the SDK
    try:
        # Try the newer API first
        result = ml_client.online_endpoints.begin_create_or_update(endpoint, local=True)
        if hasattr(result, 'result'):
            result.result()  # Wait for completion if it returns an operation
        logger.info(f"Created local endpoint '{args.endpoint_name}' using begin_create_or_update")
    except (AttributeError, TypeError):
        # Fall back to the older API
        ml_client.online_endpoints.create_or_update(endpoint, local=True)
        logger.info(f"Created local endpoint '{args.endpoint_name}' using create_or_update")
except Exception as e:
    logger.error(f"Error creating endpoint: {e}")
    raise

# Create environment configuration
logger.info("Setting up environment for deployment...")
environment = Environment(
    name="pytorch-inference-env",
    description="Environment for MNIST PyTorch inference",
    conda_file="./environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

# Create a deployment
logger.info("Creating deployment...")

# For local endpoints, we need to use a local model file
# First, check if we have a local model file

local_model_path = Path("./mnist_pytorch.pt")
if not local_model_path.exists():
    # If the model doesn't exist locally, we need to download it
    logger.info(f"Local model file not found at {local_model_path}. Downloading from Azure ML...")

    # Download the model
    try:
        # Create a directory for the model if it doesn't exist
        os.makedirs("./downloaded_model", exist_ok=True)

        # Download the model
        ml_client.models.download(
            name=latest_model.name,
            version=latest_model.version,
            download_path="./downloaded_model"
        )

        # Find the downloaded model file
        for root, dirs, files in os.walk("./downloaded_model"):
            for file in files:
                if file.endswith(".pt") or file.endswith(".pth"):
                    downloaded_model_path = os.path.join(root, file)
                    # Copy the model file to the current directory
                    shutil.copy(downloaded_model_path, str(local_model_path))
                    logger.info(f"Downloaded model file to {local_model_path}")
                    break
            if local_model_path.exists():
                break

        if not local_model_path.exists():
            raise FileNotFoundError(f"Could not find downloaded model file with .pt or .pth extension")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

# Create a model reference with the local path
model_reference = Model(
    path=str(local_model_path)
)

deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=args.endpoint_name,
    model=model_reference,
    environment=environment,
    code_configuration=CodeConfiguration(
        code="./",  # Use current directory
        scoring_script="score.py"
    ),
    environment_variables={
        "AZUREML_ENTRY_SCRIPT": "score.py",
        "AZUREML_LOG_LEVEL": "DEBUG" if args.debug else "INFO"
    },
    instance_type="Standard_DS2_v2",
    instance_count=1
)

# Deploy the model to the local endpoint
try:
    logger.info("Deploying model to local endpoint...")
    try:
        # Try the newer API first
        result = ml_client.online_deployments.begin_create_or_update(
            deployment,
            local=True,
            vscode_debug=True  # Enable VS Code debugging
        )
        if hasattr(result, 'result'):
            result.result()  # Wait for completion if it returns an operation
        logger.info(f"Created deployment 'default' for local endpoint '{args.endpoint_name}' using begin_create_or_update")
    except (AttributeError, TypeError):
        # Fall back to the older API
        ml_client.online_deployments.create_or_update(
            deployment,
            local=True,
            vscode_debug=True  # Enable VS Code debugging
        )
        logger.info(f"Created deployment 'default' for local endpoint '{args.endpoint_name}' using create_or_update")
except Exception as e:
    logger.error(f"Error creating deployment: {e}")
    # Delete the endpoint if deployment fails
    logger.info("Cleaning up failed deployment...")
    try:
        # Try the newer API first
        result = ml_client.online_endpoints.begin_delete(name=args.endpoint_name, local=True)
        if hasattr(result, 'result'):
            result.result()
    except (AttributeError, TypeError):
        # Fall back to the older API
        ml_client.online_endpoints.delete(name=args.endpoint_name, local=True)
    raise

# Set the deployment as the default for the endpoint
logger.info("Setting deployment as default...")
try:
    # Try the newer API first
    result = ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(
            name=args.endpoint_name,
            traffic={"default": 100}
        ),
        local=True
    )
    if hasattr(result, 'result'):
        result.result()  # Wait for completion if it returns an operation
    logger.info("Set deployment as default using begin_create_or_update")
except (AttributeError, TypeError):
    # Fall back to the older API
    ml_client.online_endpoints.create_or_update(
        ManagedOnlineEndpoint(
            name=args.endpoint_name,
            traffic={"default": 100}
        ),
        local=True
    )
    logger.info("Set deployment as default using create_or_update")

# Get endpoint details
try:
    endpoint = ml_client.online_endpoints.get(name=args.endpoint_name, local=True)
    logger.info(f"Local endpoint '{args.endpoint_name}' deployed successfully!")
    logger.info(f"Endpoint details: {endpoint}")
except Exception as e:
    logger.error(f"Error getting endpoint details: {e}")
    logger.info(f"Local endpoint '{args.endpoint_name}' was deployed, but could not retrieve details.")

# Create a sample request if a test image is provided
if args.test_image:
    logger.info(f"Preparing test request using image: {args.test_image}")
    try:
        # Open and convert the image to base64
        with open(args.test_image, "rb") as image_file:
            image_bytes = image_file.read()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

        # Create the request JSON
        request_data = {
            "data": base64_encoded
        }

        # Save the request to a file
        request_file_path = "sample-request.json"
        with open(request_file_path, "w") as f:
            json.dump(request_data, f)

        logger.info(f"Sample request saved to {request_file_path}")

        # Test the endpoint
        logger.info("Testing the endpoint...")
        response = ml_client.online_endpoints.invoke(
            endpoint_name=args.endpoint_name,
            request_file=request_file_path,
            local=True
        )

        logger.info("Response from endpoint:")
        print(json.dumps(json.loads(response), indent=2))

        # Parse the response to display the prediction
        try:
            result = json.loads(response)
            if 'predicted_digit' in result:
                logger.info(f"Predicted digit: {result['predicted_digit']}")
                logger.info(f"Confidence: {result['confidence']:.4f}")
            else:
                logger.warning(f"Unexpected response format: {result}")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
    except Exception as e:
        logger.error(f"Error testing endpoint: {e}")
else:
    logger.info("\nNo test image provided. To test the endpoint, you can:")
    logger.info("1. Use the Azure ML SDK to invoke the endpoint:")
    logger.info("   response = ml_client.online_endpoints.invoke(")
    logger.info(f"       endpoint_name='{args.endpoint_name}',")
    logger.info("       request_file='sample-request.json',")
    logger.info("       local=True")
    logger.info("   )")
    logger.info("\n2. Or create a sample request JSON file with a base64-encoded image and run:")
    logger.info(f"   python test_local_endpoint.py --subscription-id {args.subscription_id} --resource-group {args.resource_group} --workspace-name {args.workspace_name} --test-image path/to/image.png")
    logger.info("\n3. Or generate a test image and use it:")
    logger.info(f"   python generate_test_image.py --digit 5")
    logger.info(f"   python test_local_endpoint.py --subscription-id {args.subscription_id} --resource-group {args.resource_group} --workspace-name {args.workspace_name} --test-image test_images/mnist_digit_5_*.png")

logger.info("\nTo clean up the local endpoint when done, run:")
logger.info(f"python test_local_endpoint.py --subscription-id {args.subscription_id} --resource-group {args.resource_group} --workspace-name {args.workspace_name} --endpoint-name {args.endpoint_name} --cleanup")
