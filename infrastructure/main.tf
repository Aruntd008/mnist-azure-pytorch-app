provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "mnist" {
  name     = "mnist-classification-terraform-rg"
  location = "Canada Central"
}

# Use the newer service_plan resource, now on Basic B1
resource "azurerm_service_plan" "mnist" {
  name                = "mnist-service-plan"
  location            = azurerm_resource_group.mnist.location
  resource_group_name = azurerm_resource_group.mnist.name
  os_type             = "Linux"
  sku_name            = "B1"  # Basic tier
}

# Linux Web App for API
resource "azurerm_linux_web_app" "mnist_api" {
  name                = "mnist-pytorch-api"
  location            = azurerm_resource_group.mnist.location
  resource_group_name = azurerm_resource_group.mnist.name
  service_plan_id     = azurerm_service_plan.mnist.id

  site_config {
    application_stack {
      docker_image_name       = "${var.jfrog_url}/${var.jfrog_repo}/mnist-api:latest"
      docker_registry_url     = "https://${var.jfrog_url}"
      docker_registry_username = var.jfrog_username
      docker_registry_password = var.jfrog_password
    }
    # Basic B1 supports always_on
    always_on = true
  }

  app_settings = {
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE" = "false"
    "AZURE_ML_ENDPOINT"                   = var.api_url
  }
}

# Linux Web App for frontend
resource "azurerm_linux_web_app" "mnist_frontend" {
  name                = "mnist-pytorch-frontend"
  location            = azurerm_resource_group.mnist.location
  resource_group_name = azurerm_resource_group.mnist.name
  service_plan_id     = azurerm_service_plan.mnist.id

  site_config {
    application_stack {
      docker_image_name       = "${var.jfrog_url}/${var.jfrog_repo}/mnist-frontend:latest"
      docker_registry_url     = "https://${var.jfrog_url}"
      docker_registry_username = var.jfrog_username
      docker_registry_password = var.jfrog_password
    }
    always_on = true
  }

  app_settings = {
    "WEBSITES_ENABLE_APP_SERVICE_STORAGE" = "false"
    "API_URL"                             = "https://${azurerm_linux_web_app.mnist_api.default_hostname}"
  }
}

output "api_url" {
  value = "https://${azurerm_linux_web_app.mnist_api.default_hostname}"
}

output "frontend_url" {
  value = "https://${azurerm_linux_web_app.mnist_frontend.default_hostname}"
}
