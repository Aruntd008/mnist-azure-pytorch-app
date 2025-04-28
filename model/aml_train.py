from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subscription-id", required=True)
parser.add_argument("--resource-group", required=True)
parser.add_argument("--workspace-name", required=True)
parser.add_argument("--compute-name", required=True)
parser.add_argument("--experiment-name", default="mnist-pytorch-experiment")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--registered-model-name", default="mnist-pytorch")
args = parser.parse_args()

# 1) Authenticate & get MLClient
ml_client = MLClient(
    DefaultAzureCredential(),
    args.subscription_id,
    args.resource_group,
    args.workspace_name
)

# 2) Define or reuse a curated environment 
# This environment should include MLflow and other dependencies
cpu_env = Environment(
    name="pytorch-mlflow-env",
    description="PyTorch training with MLflow model registration",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)
cpu_env = ml_client.environments.create_or_update(cpu_env)

# 3) Submit a command job
job = command(
    code="./",  
    command=(
        "python train.py "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        f"--registered-model-name {args.registered_model_name} "
        "--output-dir outputs"
    ),
    environment=cpu_env,
    compute=args.compute_name,
    experiment_name=args.experiment_name,
    # Enable MLflow tracking in the job
    environment_variables={
        "MLFLOW_TRACKING_URI": "azureml"
    },
    # Set outputs to capture both regular outputs and MLflow artifacts
    outputs={
        "outputs": Output(type="uri_folder")
    }
)

# Submit the job and stream logs
returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)

# Get the status of the job
returned_job = ml_client.jobs.get(returned_job.name)
print(f"Job status: {returned_job.status}")

# Note: With this approach, the model registration is handled directly by MLflow
# inside the training script (train.py), so we don't need to manually register
# the model here. MLflow will register the model with the AzureML workspace during 
# the training job.

# We can still output the job information
if returned_job.status == "Completed":
    output_uri = returned_job.outputs["outputs"].path
    print(f"Job completed successfully. Output files are in: {output_uri}")
    
    # The model is already registered via MLflow in the train.py script
    print(f"Model has been registered as {args.registered_model_name}")
    print(f"Check the Azure ML studio to see all model versions")
else:
    print(f"Job failed with status: {returned_job.status}")