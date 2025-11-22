# Investment Signals Orchestration - AWS Infrastructure
#
# This Terraform configuration creates:
# - ECS Fargate cluster for Airflow
# - RDS PostgreSQL database
# - ElastiCache Redis
# - S3 bucket for logs and artifacts
# - Secrets Manager for credentials
# - CloudWatch for logging and monitoring
#
# Estimated monthly cost (minimal usage):
# - ECS Fargate: $30-50 (spot instances)
# - RDS t3.micro: $15-20
# - ElastiCache t3.micro: $10-15
# - S3: $1-5
# - Secrets Manager: $1-2
# - CloudWatch: $5-10
# Total: ~$60-100/month

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "investment-signals-terraform-state"
    key    = "orchestration/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "investment-signals"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
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

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "investment-signals-${var.environment}"
  cidr = "10.0.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_dns_hostnames   = true
  enable_dns_support     = true

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

# Security Groups
resource "aws_security_group" "ecs" {
  name_prefix = "investment-signals-ecs-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "investment-signals-ecs-${var.environment}"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "investment-signals-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = {
    Name = "investment-signals-rds-${var.environment}"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "investment-signals-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  tags = {
    Name = "investment-signals-redis-${var.environment}"
  }
}

# RDS PostgreSQL
resource "aws_db_subnet_group" "main" {
  name       = "investment-signals-${var.environment}"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

resource "aws_db_instance" "main" {
  identifier = "investment-signals-${var.environment}"

  engine               = "postgres"
  engine_version       = "14.9"
  instance_class       = var.environment == "production" ? "db.t3.small" : "db.t3.micro"
  allocated_storage    = 20
  max_allocated_storage = 100

  db_name  = "investment_signals"
  username = "signals"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = var.environment == "production" ? 7 : 1
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  skip_final_snapshot       = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "investment-signals-final-${var.environment}" : null

  performance_insights_enabled = false
  storage_encrypted           = true

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "investment-signals-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "investment-signals-${var.environment}"
  engine               = "redis"
  node_type            = var.environment == "production" ? "cache.t3.small" : "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

# S3 Bucket for logs and artifacts
resource "aws_s3_bucket" "main" {
  bucket = "investment-signals-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "main" {
  bucket = aws_s3_bucket.main.id

  rule {
    id     = "logs-cleanup"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = 30
    }
  }
}

# Secrets Manager
resource "aws_secretsmanager_secret" "db_credentials" {
  name = "investment-signals/${var.environment}/db-credentials"

  tags = {
    Name = "investment-signals-db-${var.environment}"
  }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_db_instance.main.username
    password = var.db_password
    host     = aws_db_instance.main.address
    port     = aws_db_instance.main.port
    database = aws_db_instance.main.db_name
  })
}

resource "aws_secretsmanager_secret" "airflow_credentials" {
  name = "investment-signals/${var.environment}/airflow-credentials"

  tags = {
    Name = "investment-signals-airflow-${var.environment}"
  }
}

resource "aws_secretsmanager_secret_version" "airflow_credentials" {
  secret_id = aws_secretsmanager_secret.airflow_credentials.id
  secret_string = jsonencode({
    admin_username = "admin"
    admin_password = var.airflow_admin_password
    fernet_key     = base64encode(random_string.fernet_key.result)
  })
}

resource "random_string" "fernet_key" {
  length  = 32
  special = false
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "investment-signals-${var.environment}"

  setting {
    name  = "containerInsights"
    value = var.environment == "production" ? "enabled" : "disabled"
  }

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = var.environment == "production" ? "FARGATE" : "FARGATE_SPOT"
  }
}

# IAM Role for ECS Tasks
resource "aws_iam_role" "ecs_task_execution" {
  name = "investment-signals-ecs-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_secrets" {
  name = "secrets-access"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.db_credentials.arn,
          aws_secretsmanager_secret.airflow_credentials.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "investment-signals-ecs-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.main.arn,
          "${aws_s3_bucket.main.arn}/*"
        ]
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "main" {
  name              = "/ecs/investment-signals-${var.environment}"
  retention_in_days = var.environment == "production" ? 30 : 7

  tags = {
    Name = "investment-signals-${var.environment}"
  }
}

# Outputs
output "vpc_id" {
  value = module.vpc.vpc_id
}

output "database_endpoint" {
  value = aws_db_instance.main.address
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "s3_bucket" {
  value = aws_s3_bucket.main.id
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "secrets_db_credentials_arn" {
  value = aws_secretsmanager_secret.db_credentials.arn
}

output "secrets_airflow_credentials_arn" {
  value = aws_secretsmanager_secret.airflow_credentials.arn
}
