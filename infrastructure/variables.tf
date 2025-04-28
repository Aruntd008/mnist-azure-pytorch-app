variable "jfrog_url" {
  description = "JFrog Artifactory URL"
  type        = string
}

variable "jfrog_username" {
  description = "JFrog Artifactory username"
  type        = string
}

variable "jfrog_password" {
  description = "JFrog Artifactory password"
  type        = string
  sensitive   = true
}

variable "jfrog_repo" {
  description = "JFrog Artifactory repository name"
  type        = string
}

variable "api_url" {
  description = "URL of the deployed ACI instance from Azure ML"
  type        = string
}