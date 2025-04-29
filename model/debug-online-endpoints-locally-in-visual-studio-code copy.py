#Launch development container

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential


subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace_name = "<AML_WORKSPACE_NAME>"

endpoint_name = "<ENDPOINT_NAME>"

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)


deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=Model(path="../model-1/model/sklearn_regression_model.pkl"),
    code_configuration=CodeConfiguration(
        code="../model-1/onlinescoring", scoring_script="score.py"
    ),
    environment=Environment(
        conda_file="../model-1/environment/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

deployment = ml_client.online_deployments.begin_create_or_update(
    deployment, local=True, vscode_debug=True
).result()

#2. Debug your endpoint

endpoint = ml_client.online_endpoints.get(name=endpoint_name, local=True)

request_file_path = "../model-1/sample-request.json"

ml_client.online_endpoints.invoke(endpoint_name, request_file_path, local=True)


print(endpoint)


#3. Edit your endpoint


new_deployment = ManagedOnlineDeployment(
    name="green",
    endpoint_name=endpoint_name,
    model=Model(path="../model-2/model/sklearn_regression_model.pkl"),
    code_configuration=CodeConfiguration(
        code="../model-2/onlinescoring", scoring_script="score.py"
    ),
    environment=Environment(
        conda_file="../model-2/environment/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    ),
    instance_type="Standard_DS3_v2",
    instance_count=2,
)

deployment = ml_client.online_deployments.begin_create_or_update(
    new_deployment, local=True, vscode_debug=True
).result()