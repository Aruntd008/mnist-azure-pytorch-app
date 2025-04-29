import json
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

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

# Connect to workspace using Azure ML SDK v2
ml_client = MLClient(
    DefaultAzureCredential(),
    args.subscription_id,
    args.resource_group,
    args.workspace_name
)
print(f"Connected to workspace {ml_client.workspace_name}")

# Get the latest version of the registered model
model_versions = list(ml_client.models.list(name=args.model_name))
if not model_versions:
    raise Exception(f"No model with name {args.model_name} found in workspace")

latest_model = model_versions[0]
print(f"Model {latest_model.name} (version {latest_model.version}) loaded")

# Create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=args.service_name,
    description="MNIST digit classifier endpoint",
    auth_mode="key"
)

# Check if the endpoint already exists and delete it if necessary
try:
    # Check if endpoint exists
    existing_endpoint = ml_client.online_endpoints.get(name=args.service_name)
    if existing_endpoint:
        print(f"Found existing endpoint {args.service_name}. Deleting it first...")
        ml_client.online_endpoints.begin_delete(name=args.service_name).wait()
        print(f"Endpoint {args.service_name} deleted successfully.")
except Exception as e:
    # If the endpoint doesn't exist or can't be accessed, we can continue
    print(f"No existing endpoint found or unable to access it: {e}")

# Create or update the endpoint
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Endpoint {endpoint.name} created or updated")

# Create an environment for inference
environment = Environment(
    name="pytorch-inference-env",
    description="Environment for PyTorch inference",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

# Create the deployment
deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=args.service_name,
    model=latest_model.id,
    environment=environment,
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1
)

# Create or update the deployment
deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
print(f"Deployment {deployment.name} created or updated")

# Set the deployment as the default for the endpoint
endpoint.traffic = {"default": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Get the scoring endpoint
scoring_uri = endpoint.scoring_uri
print(f"Service deployed successfully. Scoring URI: {scoring_uri}")

# Save endpoint information to a file
endpoint_info = {
    "scoring_uri": scoring_uri,
    "service_name": endpoint.name,
    "model_name": latest_model.name,
    "model_version": latest_model.version
}

with open("endpoint_info.json", "w") as f:
    json.dump(endpoint_info, f)

print(f"Endpoint information saved to endpoint_info.json")