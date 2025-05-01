provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "mnist" {
  name     = "mnist-classification-rg"
  location = "East US"
}

# Create App Service Plan
resource "azurerm_app_service_plan" "mnist" {
  name                = "mnist-app-service-plan"
  location            = azurerm_resource_group.mnist.location
  resource_group_name = azurerm_resource_group.mnist.name
  kind                = "Linux"
  reserved            = true

  sku {
    tier = "Basic"
    size = "B1"
  }
}

# Create App Service for API
resource "azurerm_app_service" "mnist_api" {
  name                = "mnist-pytorch-api"
  location            = azurerm_resource_group.mnist.location
  resource_group_name = azurerm_resource_group.mnist.name
  app_service_plan_id = azurerm_app_service_plan.mnist.id

  site_config {
    linux_fx_version = "DOCKER|${var.jfrog_url}/${var.jfrog_repo}/mnist-api:latest"
    always_on        = true
  }

  app_settings = {
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE" = "false"
    "DOCKER_REGISTRY_SERVER_URL"          = "https://${var.jfrog_url}"
    "DOCKER_REGISTRY_SERVER_USERNAME"     = var.jfrog_username
    "DOCKER_REGISTRY_SERVER_PASSWORD"     = var.jfrog_password
    "AZURE_ML_ENDPOINT"                   = var.api_url
  }
}

# Create App Service for frontend
resource "azurerm_app_service" "mnist_frontend" {
  name                = "mnist-pytorch-frontend"
  location            = azurerm_resource_group.mnist.location
  resource_group_name = azurerm_resource_group.mnist.name
  app_service_plan_id = azurerm_app_service_plan.mnist.id

  site_config {
    linux_fx_version = "DOCKER|${var.jfrog_url}/${var.jfrog_repo}/mnist-frontend:latest"
    always_on        = true
  }

  app_settings = {
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE" = "false"
    "DOCKER_REGISTRY_SERVER_URL"          = "https://${var.jfrog_url}"
    "DOCKER_REGISTRY_SERVER_USERNAME"     = var.jfrog_username
    "DOCKER_REGISTRY_SERVER_PASSWORD"     = var.jfrog_password
    "API_URL"                             = "https://${azurerm_app_service.mnist_api.default_site_hostname}"
  }
}

output "api_url" {
  value = "https://${azurerm_app_service.mnist_api.default_site_hostname}"
}

output "frontend_url" {
  value = "https://${azurerm_app_service.mnist_frontend.default_site_hostname}"
}