# Production Deployment Guide

## Overview

This guide covers the production deployment of the Investment Signals Orchestration platform. It includes instructions for deploying to AWS and GCP, monitoring setup, and operational procedures.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Local Development](#local-development)
4. [AWS Deployment](#aws-deployment)
5. [GCP Deployment](#gcp-deployment)
6. [Monitoring Stack](#monitoring-stack)
7. [Alert Configuration](#alert-configuration)
8. [Logging](#logging)
9. [Rollback Procedures](#rollback-procedures)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

- Docker & Docker Compose (v2.0+)
- AWS CLI v2 (for AWS deployment)
- gcloud CLI (for GCP deployment)
- Terraform >= 1.5
- jq (for JSON parsing)

### Required Credentials

#### AWS
- AWS Access Key and Secret Key with permissions for:
  - ECS (Elastic Container Service)
  - ECR (Elastic Container Registry)
  - ALB (Application Load Balancer)
  - RDS (for PostgreSQL)
  - ElastiCache (for Redis)
  - CloudWatch

#### GCP
- Service account with permissions for:
  - Cloud Run
  - Artifact Registry
  - Cloud SQL
  - Cloud Monitoring
  - Cloud Logging

---

## Architecture Overview

```
                                    +------------------+
                                    |   Load Balancer  |
                                    +--------+---------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
    +---------v--------+         +-----------v----------+        +----------v---------+
    |  Airflow Web     |         |  Airflow Scheduler   |        |  Airflow Worker    |
    |  (Port 8080)     |         |                      |        |  (Celery)          |
    +--------+---------+         +-----------+----------+        +----------+---------+
              |                              |                              |
              +------------------------------+------------------------------+
                                             |
                               +-------------v-------------+
                               |      PostgreSQL           |
                               |   (Airflow + Signals DB)  |
                               +---------------------------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
    +---------v--------+         +-----------v----------+        +----------v---------+
    |   Prometheus     |         |    Alertmanager      |        |      Grafana       |
    |   (Port 9090)    |         |    (Port 9093)       |        |    (Port 3000)     |
    +------------------+         +----------------------+        +--------------------+
              |
    +---------v--------+
    |      Loki        |
    |   (Port 3100)    |
    +------------------+
```

---

## Local Development

### Quick Start

```bash
# Navigate to orchestration directory
cd orchestration/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access services
# Airflow:      http://localhost:8080 (admin/admin)
# Grafana:      http://localhost:3000 (admin/admin)
# Prometheus:   http://localhost:9090
# Alertmanager: http://localhost:9093
```

### Development Mode

```bash
# Use development compose file for hot-reloading
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Environment Variables

Create a `.env` file in the `docker/` directory:

```bash
# Airflow
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=secure_password

# Database
DB_USER=signals
DB_PASSWORD=secure_password

# Grafana
GF_ADMIN_USER=admin
GF_ADMIN_PASSWORD=secure_password

# Alerting
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=app-specific-password
SMTP_FROM=alerts@your-domain.com
ALERT_EMAIL_RECIPIENTS=team@your-domain.com

# Slack (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx
SLACK_CRITICAL_CHANNEL=#alerts-critical
SLACK_WARNING_CHANNEL=#alerts-warning

# PagerDuty (optional)
PAGERDUTY_SERVICE_KEY=your-service-key
```

---

## AWS Deployment

### One-Command Deployment

```bash
# Deploy to production
./scripts/deploy_aws.sh prod deploy

# Deploy to staging
./scripts/deploy_aws.sh staging deploy

# Deploy to development
./scripts/deploy_aws.sh dev deploy
```

### Manual Steps

#### 1. Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
```

#### 2. Set Environment Variables

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPOSITORY=investment-signals
export ECS_CLUSTER=investment-signals-prod
export ECS_SERVICE=orchestration-prod
```

#### 3. Build and Push Images

```bash
# Build images
docker build -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:airflow-latest \
    -f docker/Dockerfile.airflow docker/

docker build -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:healthcheck-latest \
    -f docker/Dockerfile.healthcheck docker/

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Push images
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:airflow-latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:healthcheck-latest
```

#### 4. Deploy Infrastructure with Terraform

```bash
cd cloud/aws
terraform init
terraform plan -var="environment=prod"
terraform apply -var="environment=prod"
```

### AWS Cost Optimization

| Service | Dev | Staging | Prod |
|---------|-----|---------|------|
| ECS (Fargate) | 0.5 vCPU, 1GB | 1 vCPU, 2GB | 2 vCPU, 4GB |
| RDS | db.t3.micro | db.t3.small | db.t3.medium |
| ElastiCache | cache.t3.micro | cache.t3.micro | cache.t3.small |
| **Estimated Monthly** | ~$60 | ~$100 | ~$200 |

---

## GCP Deployment

### One-Command Deployment

```bash
# Deploy to production
./scripts/deploy_gcp.sh prod deploy

# Deploy to staging
./scripts/deploy_gcp.sh staging deploy

# Deploy to development
./scripts/deploy_gcp.sh dev deploy
```

### Manual Steps

#### 1. Configure GCP Credentials

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### 2. Set Environment Variables

```bash
export GCP_PROJECT=your-project-id
export GCP_REGION=us-central1
export GCR_REPOSITORY=investment-signals
export CLOUD_RUN_SERVICE=orchestration-prod
```

#### 3. Deploy

```bash
./scripts/deploy_gcp.sh prod deploy
```

### GCP Cost Optimization

| Service | Dev | Staging | Prod |
|---------|-----|---------|------|
| Cloud Run | min=0, max=2 | min=1, max=5 | min=2, max=10 |
| Cloud SQL | db-f1-micro | db-g1-small | db-custom-2-4096 |
| Memorystore | 1GB | 1GB | 2GB |
| **Estimated Monthly** | ~$50 | ~$100 | ~$200 |

---

## Monitoring Stack

### Prometheus Metrics

The following custom metrics are exposed:

| Metric | Type | Description |
|--------|------|-------------|
| `investment_signals_api_cost_dollars_total` | Counter | Cumulative API costs by provider |
| `investment_signals_api_requests_total` | Counter | Total API requests by provider |
| `investment_signals_pipeline_duration_seconds` | Histogram | Pipeline execution duration |
| `investment_signals_signals_detected_total` | Counter | Signals detected by pipeline |
| `investment_signals_data_quality_errors_total` | Counter | Data quality errors |

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (or your deployment URL).

Available dashboards:

1. **Pipeline Overview** - Overall pipeline health and performance
2. **SLA Compliance** - SLA metrics and violations
3. **Cost Monitoring** - API costs and budget tracking
4. **Alert History** - Alert trends and resolution times
5. **Data Quality** - Data quality metrics and issues

### Accessing Dashboards

```
Dashboard URL patterns:
- Pipeline Overview: /d/pipeline-overview/pipeline-overview
- Cost Monitoring: /d/cost-monitoring/cost-monitoring
- Alert History: /d/alert-history/alert-history
- SLA Compliance: /d/sla-compliance/sla-compliance
- Data Quality: /d/data-quality/data-quality
```

---

## Alert Configuration

### Alert Severity Levels

| Severity | Response Time | Notification Channels | Example |
|----------|---------------|----------------------|---------|
| Critical | < 15 min | PagerDuty + Slack + Email | System down, SLA breach |
| Warning | < 4 hours | Slack + Email | High error rate, slow performance |
| Info | Best effort | Email (batched) | Informational events |

### Configuring Notifications

#### Slack

1. Create a Slack app at https://api.slack.com/apps
2. Enable Incoming Webhooks
3. Create webhooks for each channel
4. Set environment variables:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx
export SLACK_CRITICAL_CHANNEL=#alerts-critical
export SLACK_WARNING_CHANNEL=#alerts-warning
```

#### PagerDuty

1. Create a service in PagerDuty
2. Get the Integration Key
3. Set environment variable:

```bash
export PAGERDUTY_SERVICE_KEY=your-integration-key
```

#### Email

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER=your-email@gmail.com
export SMTP_PASSWORD=app-specific-password
export ALERT_EMAIL_RECIPIENTS=team@your-domain.com
```

### Alert Rules

Key alerts configured:

| Alert | Severity | Threshold | Description |
|-------|----------|-----------|-------------|
| DAGExecutionFailed | Critical | Any failure | DAG execution failed |
| HighDailyAPICost | Warning | >$50/day | API costs exceed daily budget |
| WeeklyBudgetExceeded | Critical | >$300/week | Weekly budget exceeded |
| SLABreach | Critical | Per pipeline | Pipeline SLA violated |
| HighTaskFailureRate | Warning | >20% | High task failure rate |

---

## Logging

### Log Architecture

```
Application Logs --> Promtail --> Loki --> Grafana
```

### Log Retention

- **Development**: 7 days
- **Staging**: 14 days
- **Production**: 30 days

### Querying Logs in Grafana

```logql
# All errors from clinical trial pipeline
{pipeline="clinical_trial"} |= "ERROR"

# API cost logs
{job="application"} | json | api_cost > 0

# Recent DAG failures
{job="dag_logs"} | json | status="failed"
```

### Log Labels

| Label | Description | Example Values |
|-------|-------------|----------------|
| `job` | Log source | airflow, healthcheck, dag_logs |
| `level` | Log level | INFO, WARNING, ERROR |
| `pipeline` | Pipeline name | clinical_trial, patent_ip |
| `dag_id` | Airflow DAG ID | clinical_trial_signal_detection |
| `task_id` | Airflow task ID | fetch_trials, calculate_scores |

---

## Rollback Procedures

### Automatic Rollback

The deployment scripts automatically roll back if health checks fail:

```bash
# The deploy script will automatically rollback on failure
./scripts/deploy_aws.sh prod deploy
```

### Manual Rollback

#### AWS

```bash
# Rollback to previous version
./scripts/deploy_aws.sh prod rollback

# Or manually
aws ecs update-service \
    --cluster investment-signals-prod \
    --service orchestration-prod \
    --task-definition <previous-task-definition-arn> \
    --force-new-deployment
```

#### GCP

```bash
# Rollback to previous version
./scripts/deploy_gcp.sh prod rollback

# Or manually switch traffic
gcloud run services update-traffic orchestration-prod \
    --region us-central1 \
    --to-revisions <previous-revision>=100
```

---

## Troubleshooting

### Common Issues

#### 1. DAG Not Running

**Symptoms**: DAG shows as "paused" or no runs scheduled

**Resolution**:
```bash
# Unpause the DAG
docker-compose exec airflow-webserver airflow dags unpause clinical_trial_signal_detection

# Trigger a run manually
docker-compose exec airflow-webserver airflow dags trigger clinical_trial_signal_detection
```

#### 2. Database Connection Errors

**Symptoms**: "Connection refused" or "timeout" errors

**Resolution**:
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection
docker-compose exec postgres pg_isready -U airflow

# View PostgreSQL logs
docker-compose logs postgres
```

#### 3. High Memory Usage

**Symptoms**: Container OOM killed or slow performance

**Resolution**:
```bash
# Check memory usage
docker stats

# Increase memory limits in docker-compose.yml
# Or scale horizontally by adding more workers
```

#### 4. Alertmanager Not Sending Alerts

**Symptoms**: Alerts firing but no notifications

**Resolution**:
1. Check Alertmanager config: `http://localhost:9093/#/status`
2. Verify environment variables are set
3. Check logs: `docker-compose logs alertmanager`
4. Test notification channel directly

### Health Check Endpoints

| Service | Endpoint | Expected Response |
|---------|----------|-------------------|
| Airflow | `/health` | `{"status": "healthy"}` |
| Health Check | `/health` | JSON with component status |
| Prometheus | `/-/healthy` | `Prometheus is Healthy.` |
| Alertmanager | `/-/healthy` | `OK` |
| Grafana | `/api/health` | `{"database": "ok"}` |
| Loki | `/ready` | `ready` |

### Getting Support

1. Check the logs: `docker-compose logs -f <service>`
2. Review Grafana dashboards for metrics
3. Check Alertmanager for firing alerts
4. Query Loki for detailed logs
5. Open an issue on GitHub

---

## Production Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Secrets stored securely (AWS Secrets Manager / GCP Secret Manager)
- [ ] Database backups enabled
- [ ] SSL certificates configured
- [ ] DNS records configured
- [ ] Alert channels tested
- [ ] Runbooks reviewed

### Post-Deployment

- [ ] Health checks passing
- [ ] Dashboards loading correctly
- [ ] Sample DAG run successful
- [ ] Alerts firing to correct channels
- [ ] Logs flowing to Loki
- [ ] Cost monitoring active
- [ ] Documentation updated

---

## Contact

For questions or issues:
- GitHub Issues: https://github.com/UMwai/investment-orchestration/issues
- Documentation: https://github.com/UMwai/investment-orchestration/wiki
