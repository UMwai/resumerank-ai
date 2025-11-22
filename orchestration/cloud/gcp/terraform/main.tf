# Investment Signals Orchestration - GCP Infrastructure
#
# This Terraform configuration creates:
# - Cloud Run services for Airflow components
# - Cloud SQL PostgreSQL
# - Cloud Memorystore Redis
# - Cloud Storage bucket
# - Secret Manager for credentials
# - Cloud Monitoring
#
# Estimated monthly cost (minimal usage):
# - Cloud Run: $20-40 (with minimum instances)
# - Cloud SQL: $15-25 (db-f1-micro)
# - Memorystore: $10-15 (basic tier)
# - Cloud Storage: $1-5
# - Secret Manager: $1-2
# - Cloud Monitoring: $5-10
# Total: ~$50-100/month

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  backend "gcs" {
    bucket = "investment-signals-terraform-state"
    prefix = "orchestration"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev/staging/production)"
  type        = string
  default     = "dev"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "airflow_admin_password" {
  description = "Airflow admin password"
  type        = string
  sensitive   = true
}

# Enable required APIs
resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "vpcaccess.googleapis.com",
    "servicenetworking.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "investment-signals-${var.environment}"
  auto_create_subnetworks = false

  depends_on = [google_project_service.services]
}

resource "google_compute_subnetwork" "main" {
  name          = "investment-signals-${var.environment}"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.main.id

  private_ip_google_access = true
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "main" {
  name          = "investment-signals-${var.environment}"
  region        = var.region
  network       = google_compute_network.main.name
  ip_cidr_range = "10.8.0.0/28"

  min_instances = 2
  max_instances = var.environment == "production" ? 10 : 3

  depends_on = [google_project_service.services]
}

# Private Service Connection for Cloud SQL
resource "google_compute_global_address" "private_ip" {
  name          = "investment-signals-private-ip-${var.environment}"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}

resource "google_service_networking_connection" "private" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip.name]
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "main" {
  name             = "investment-signals-${var.environment}"
  database_version = "POSTGRES_14"
  region           = var.region

  depends_on = [google_service_networking_connection.private]

  settings {
    tier              = var.environment == "production" ? "db-custom-1-3840" : "db-f1-micro"
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"

    disk_size         = 20
    disk_autoresize   = true
    disk_autoresize_limit = 100

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
    }

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = var.environment == "production"
      backup_retention_settings {
        retained_backups = var.environment == "production" ? 7 : 1
      }
    }

    maintenance_window {
      day  = 7  # Sunday
      hour = 4
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  deletion_protection = var.environment == "production"
}

resource "google_sql_database" "main" {
  name     = "investment_signals"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "main" {
  name     = "signals"
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Cloud Memorystore Redis
resource "google_redis_instance" "main" {
  name               = "investment-signals-${var.environment}"
  tier               = "BASIC"
  memory_size_gb     = var.environment == "production" ? 2 : 1
  region             = var.region
  redis_version      = "REDIS_7_0"

  authorized_network = google_compute_network.main.id

  depends_on = [google_project_service.services]
}

# Cloud Storage Bucket
resource "google_storage_bucket" "main" {
  name     = "investment-signals-${var.environment}-${var.project_id}"
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    project     = "investment-signals"
  }
}

# Secret Manager
resource "google_secret_manager_secret" "db_credentials" {
  secret_id = "investment-signals-db-credentials-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_version" "db_credentials" {
  secret = google_secret_manager_secret.db_credentials.id

  secret_data = jsonencode({
    username = google_sql_user.main.name
    password = var.db_password
    host     = google_sql_database_instance.main.private_ip_address
    port     = 5432
    database = google_sql_database.main.name
  })
}

resource "google_secret_manager_secret" "airflow_credentials" {
  secret_id = "investment-signals-airflow-credentials-${var.environment}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret_version" "airflow_credentials" {
  secret = google_secret_manager_secret.airflow_credentials.id

  secret_data = jsonencode({
    admin_username = "admin"
    admin_password = var.airflow_admin_password
    fernet_key     = base64encode(random_string.fernet_key.result)
  })
}

resource "random_string" "fernet_key" {
  length  = 32
  special = false
}

# Service Account for Cloud Run
resource "google_service_account" "cloud_run" {
  account_id   = "investment-signals-run-${var.environment}"
  display_name = "Investment Signals Cloud Run Service Account"
}

resource "google_project_iam_member" "cloud_run_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_project_iam_member" "cloud_run_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_secret_manager_secret_iam_member" "db_credentials" {
  secret_id = google_secret_manager_secret.db_credentials.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

resource "google_secret_manager_secret_iam_member" "airflow_credentials" {
  secret_id = google_secret_manager_secret.airflow_credentials.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Cloud Run Service - Airflow Webserver
resource "google_cloud_run_v2_service" "airflow_webserver" {
  name     = "airflow-webserver-${var.environment}"
  location = var.region

  template {
    service_account = google_service_account.cloud_run.email

    scaling {
      min_instance_count = var.environment == "production" ? 1 : 0
      max_instance_count = var.environment == "production" ? 4 : 2
    }

    vpc_access {
      connector = google_vpc_access_connector.main.id
      egress    = "ALL_TRAFFIC"
    }

    containers {
      image = "apache/airflow:2.7.3-python3.11"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.environment == "production" ? "2" : "1"
          memory = var.environment == "production" ? "2Gi" : "1Gi"
        }
      }

      env {
        name  = "AIRFLOW__CORE__EXECUTOR"
        value = "LocalExecutor"
      }

      env {
        name = "AIRFLOW__CORE__SQL_ALCHEMY_CONN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_credentials.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "AIRFLOW__WEBSERVER__EXPOSE_CONFIG"
        value = "true"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 30
        period_seconds        = 10
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_secret_manager_secret_version.db_credentials,
    google_secret_manager_secret_version.airflow_credentials,
  ]
}

# Allow unauthenticated access to webserver (or configure IAP)
resource "google_cloud_run_v2_service_iam_member" "webserver_public" {
  count    = var.environment == "dev" ? 1 : 0
  location = google_cloud_run_v2_service.airflow_webserver.location
  name     = google_cloud_run_v2_service.airflow_webserver.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Outputs
output "network_name" {
  value = google_compute_network.main.name
}

output "database_private_ip" {
  value = google_sql_database_instance.main.private_ip_address
}

output "database_connection_name" {
  value = google_sql_database_instance.main.connection_name
}

output "redis_host" {
  value = google_redis_instance.main.host
}

output "storage_bucket" {
  value = google_storage_bucket.main.name
}

output "airflow_webserver_url" {
  value = google_cloud_run_v2_service.airflow_webserver.uri
}

output "service_account_email" {
  value = google_service_account.cloud_run.email
}
