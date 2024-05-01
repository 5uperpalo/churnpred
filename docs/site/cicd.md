# CI/CD

This section includes some ideas for model developmnet and deployment within Azure services - i.e. EcoVadis cloud services vendor.

* MLFlow for experiment tracking and endpoint deployment
  * https://mlflow.org/docs/1.25.1/python_api/mlflow.azureml.html
  * https://mlflow.org/docs/latest/deployment/index.html
  * https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints?view=azureml-api-2&tabs=mlflow
* DVC for dataset versioning 
  * python api: https://dvc.org/doc/api-reference
  * Dataset storage in Azure Blob Storage
    * https://dvc.org/doc/user-guide/data-management/remote-storage/azure-blob-storage#microsoft-azure-blob-storage 
  * Dataset and predictions(dev) visualization from Azure Blob Storage
    * https://learn.microsoft.com/en-us/azure/data-explorer/azure-data-explorer-dashboards
    * https://learn.microsoft.com/en-us/azure/data-explorer/create-event-grid-connection?tabs=portal-adx%2Cazure-blob-storage
* Data handling in PROD
  * databricks: https://learn.microsoft.com/en-us/azure/databricks/getting-started/data-pipeline-get-started
  * create and share dashboards within organization: https://learn.microsoft.com/en-us/azure/databricks/sql/get-started/sample-dashboards
