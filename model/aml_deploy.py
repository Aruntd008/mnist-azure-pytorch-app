from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential, ClientSecretCredential
import os
import argparse
import json

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--subscription-id", required=True)
parser.add_argument("--resource-group", required=True)
parser.add_argument("--workspace-name", required=True)
parser.add_argument("--model-name", default="mnist-pytorch")
parser.add_argument("--service-name", default="mnist-inference-service")
args = parser.parse_args()

# Authenticate using DefaultAzureCredential (which works with Managed Identity or az login)
try:
    credential = DefaultAzureCredential()
    
    # Initialize MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )
    print("Successfully authenticated using DefaultAzureCredential")
except Exception as e:
    print(f"DefaultAzureCredential failed. Error: {e}")
    
    # Fall back to ClientSecretCredential if environment variables are available
    if all(var in os.environ for var in ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"]):
        print("Trying ClientSecretCredential...")
        credential = ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"]
        )
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name
        )
        print("Successfully authenticated using ClientSecretCredential")
    else:
        raise Exception("Could not authenticate to Azure. Please check your credentials.")

# Get the latest version of the model
model_versions = list(ml_client.models.list(name=args.model_name))
if not model_versions:
    raise Exception(f"No model found with name {args.model_name}")

# Sort by creation time to get the latest
latest_model = sorted(model_versions, key=lambda x: x.creation_context.created_at, reverse=True)[0]
print(f"Using model {latest_model.name} version {latest_model.version}")

# Create a custom environment instead of using a curated one
custom_env = Environment(
    name="pytorch-inference-env",
    description="Custom environment for MNIST inference",
    conda_file="./environment.yml",  # Fixed path - was "../model/environment.yml"
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

try:
    # Check if the environment already exists
    env = ml_client.environments.get(name="pytorch-inference-env")
    print("Using existing environment")
except Exception:
    # Create the environment if it doesn't exist
    print("Creating new environment...")
    env = ml_client.environments.create_or_update(custom_env)
    print(f"Created environment {env.name} version {env.version}")

# Create a unique endpoint name (endpoint names must be unique within a region)
import uuid
unique_suffix = str(uuid.uuid4())[:8]
endpoint_name = f"{args.service_name}-{unique_suffix}"

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="MNIST PyTorch inference endpoint",
    auth_mode="key"
)

try:
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Created endpoint {endpoint_name}")
except Exception as e:
    print(f"Error creating endpoint: {e}")
    raise
print('model name',latest_model.id)
# Create deployment with increased instance size and added debugging env variables
deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint_name,
    model=latest_model.id,
    environment=env.id,
    instance_type="Standard_E4s_v3",  # Upgraded from DS2_v2 as recommended
    instance_count=1,
    code_configuration=CodeConfiguration(
        code="./",  # Use current directory
        scoring_script="score.py"
    ),
    environment_variables={
        "AZUREML_ENTRY_SCRIPT": "score.py",
        "AZUREML_LOG_LEVEL": "DEBUG"  # Add debug logging
    }
)

try:
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Created deployment 'default' for endpoint {endpoint_name}")
except Exception as e:
    print(f"Error creating deployment: {e}")
    # Delete the endpoint if deployment fails   
    ml_client.online_endpoints.begin_delete(name=endpoint_name)
    raise

# Set the deployment as the default for the endpoint
ml_client.online_endpoints.begin_create_or_update(
    ManagedOnlineEndpoint(
        name=endpoint_name,
        traffic={"default": 100}
    )
).result()

# Get endpoint details
endpoint = ml_client.online_endpoints.get(name=endpoint_name)
scoring_uri = endpoint.scoring_uri

print(f"Endpoint '{endpoint_name}' deployed successfully!")
print(f"Scoring URI: {scoring_uri}")

# Save the endpoint information for later use
endpoint_info = {
    "endpoint_name": endpoint_name,
    "scoring_uri": scoring_uri
}

with open("endpoint_info.json", "w") as f:
    json.dump(endpoint_info, f)

print("Endpoint information saved to endpoint_info.json")