import os
import json
import argparse
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# Parse arguments
parser = argparse.ArgumentParser(description='Deploy MNIST model to ACI')
parser.add_argument('--subscription-id', type=str, required=True, help='Azure subscription ID')
parser.add_argument('--resource-group', type=str, required=True, help='Azure resource group')
parser.add_argument('--workspace-name', type=str, required=True, help='Azure ML workspace name')
parser.add_argument('--model-name', type=str, required=True, help='Registered model name')
parser.add_argument('--service-name', type=str, required=True, help='Deployment service name')
parser.add_argument('--cpu-cores', type=float, default=1.0, help='Number of CPU cores')
parser.add_argument('--memory-gb', type=float, default=1.0, help='Memory in GB')

args = parser.parse_args()

# Connect to workspace
ws = Workspace(
    subscription_id=args.subscription_id,
    resource_group=args.resource_group,
    workspace_name=args.workspace_name
)
print(f"Connected to workspace {ws.name}")

# Get the registered model
model = Model(ws, name=args.model_name)
print(f"Model {model.name} (version {model.version}) loaded")

# Create inference environment
env = Environment(name="pytorch-inference-env")
env.docker.enabled = True

# Specify conda dependencies
conda_deps = CondaDependencies()
conda_deps.add_pip_package("torch>=2.0.0")
conda_deps.add_pip_package("torchvision>=0.15.0")
conda_deps.add_pip_package("fastapi>=0.95.0")
conda_deps.add_pip_package("uvicorn>=0.22.0")
conda_deps.add_pip_package("python-multipart>=0.0.18")
conda_deps.add_pip_package("pillow>=9.5.0")
conda_deps.add_pip_package("numpy>=1.24.0")

env.python.conda_dependencies = conda_deps

# Create inference configuration
inference_config = InferenceConfig(
    entry_script="score.py",
    source_directory=os.path.dirname(os.path.realpath(__file__)),
    environment=env
)

# Define deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=args.cpu_cores,
    memory_gb=args.memory_gb,
    tags={"data": "MNIST", "framework": "PyTorch"},
    description="MNIST digit classifier"
)

# Deploy the web service
service = Model.deploy(
    workspace=ws,
    name=args.service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

# Wait for deployment to complete
service.wait_for_deployment(show_output=True)

# Get the scoring endpoint
print(f"Service deployed successfully. Scoring URI: {service.scoring_uri}")

# Save endpoint information to a file
endpoint_info = {
    "scoring_uri": service.scoring_uri,
    "service_name": service.name,
    "model_name": model.name,
    "model_version": model.version
}

with open("endpoint_info.json", "w") as f:
    json.dump(endpoint_info, f)

print(f"Endpoint information saved to endpoint_info.json")