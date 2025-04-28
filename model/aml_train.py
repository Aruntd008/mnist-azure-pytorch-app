from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subscription-id", required=True)
parser.add_argument("--resource-group", required=True)
parser.add_argument("--workspace-name", required=True)
parser.add_argument("--compute-name", required=True)
parser.add_argument("--experiment-name", default="mnist-pytorch-experiment")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

# 1) Authenticate & get MLClient
ml_client = MLClient(
    DefaultAzureCredential(),
    args.subscription_id,
    args.resource_group,
    args.workspace_name
)

# 2) Ensure compute exists (or use existing)
# (you can omit if already created via CLI / portal)

# 3) Define or reuse a curated environment
cpu_env = Environment(
    name="custom-pytorch-env",              # choose your own name
    description="PyTorch training with custom deps",
    conda_file="environment.yml",           # your Conda YAML
    # you can also specify a base image if needed:
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)
cpu_env = ml_client.environments.create_or_update(cpu_env)

# 4) Submit a command job (unchanged)
job = command(
    code="./",  
    command=(
        "python train.py "
        f"--epochs {args.epochs} "
        f"--batch-size {args.batch_size} "
        "--output-dir outputs"
    ),
    environment=cpu_env,
    compute=args.compute_name,
    experiment_name=args.experiment_name,
    outputs={
        "outputs": Output(type="uri_folder")
    }
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)

# 5) Once the job is finished, pull its output URI
returned_job = ml_client.jobs.get(returned_job.name)
if returned_job.status == "Completed":
    # use the 'default' channel, which has the real path
    output_uri = returned_job.outputs["default"].path
    print(f"Model files are in: {output_uri}")

    model_path = f"azureml://jobs/{returned_job.name}/outputs/outputs/mnist_pytorch.pt"
    model = Model(
        name="mnist-pytorch",
        path=model_path,
        description="MNIST model trained with PyTorch",
        tags={"framework": "pytorch", "dataset": "mnist"}
    )
    registered_model = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered_model.name} v{registered_model.version}")
else:
    print(f"Job failed: {returned_job.status}")
