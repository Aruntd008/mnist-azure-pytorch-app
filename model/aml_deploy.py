# Simplified deployment script
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from dotenv import load_dotenv
import os
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

# Load variables from .env
load_dotenv()

# Access the environment variables
credential = ClientSecretCredential(
    tenant_id=os.getenv("TENANT_ID"),
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
)

ml_client = MLClient(
    credential,
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RESOURCE_GROUP_NAME"),
    workspace_name=os.getenv("WORKSPACE_NAME"),
)

# Get the model
model = ml_client.models.get(name="mnist-pytorch", version="7")

# Create a new endpoint
endpoint_name = "mnist-inference-s"  # Use a different name
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key",
    description="MNIST inference endpoint"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment using curated environment
deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint_name,
    model=model.id,
    environment="AzureML-PyTorch-1.10-CPU",  # Curated environment
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment).result()