# Investment Signals Orchestration System

Automated scheduling and orchestration system for investment intelligence pipelines using Apache Airflow.

## Overview

This orchestration system manages three investment signal detection pipelines:

| Pipeline | Schedule | Description |
|----------|----------|-------------|
| **Clinical Trial Signals** | Daily at 6:00 PM ET | Monitors ClinicalTrials.gov for Phase 3 trial updates |
| **Patent/IP Intelligence** | Weekly on Monday 8:00 AM ET | Analyzes FDA Orange Book for patent cliff opportunities |
| **Insider Activity + Hiring** | Multiple schedules | Form 4, 13F, and job posting analysis |

### Sub-schedules for Insider/Hiring Pipeline

| Signal Type | Schedule | Description |
|-------------|----------|-------------|
| Form 4 | Every 30 min (9:30 AM - 4:00 PM ET, Mon-Fri) | Real-time insider trading alerts |
| 13F | Quarterly (45 days after quarter end) | Institutional holdings analysis |
| Job Postings | Daily at 9:00 AM ET | Company hiring signal detection |

## Directory Structure

```
orchestration/
├── dags/                          # Airflow DAG definitions
│   ├── clinical_trial_dag.py      # Clinical trial signal DAG
│   ├── patent_ip_dag.py           # Patent/IP intelligence DAG
│   └── insider_hiring_dag.py      # Insider/hiring signals DAGs
├── config/                        # Configuration files
│   ├── settings.py                # Central configuration
│   ├── env.example                # Environment template
│   └── __init__.py
├── alerts/                        # Alerting system
│   ├── alert_manager.py           # Multi-channel alert manager
│   └── __init__.py
├── monitoring/                    # Health checks and metrics
│   ├── health_checks.py           # Pipeline health monitoring
│   ├── metrics.py                 # Prometheus metrics collection
│   └── __init__.py
├── docker/                        # Docker deployment
│   ├── docker-compose.yml         # Local orchestration stack
│   ├── Dockerfile.airflow         # Custom Airflow image
│   ├── Dockerfile.healthcheck     # Health check service
│   ├── requirements.txt           # Python dependencies
│   ├── init-db.sql                # Database initialization
│   ├── prometheus.yml             # Prometheus configuration
│   └── grafana/                   # Grafana dashboards
├── cloud/                         # Cloud deployments
│   ├── aws/terraform/             # AWS infrastructure
│   └── gcp/terraform/             # GCP infrastructure
├── scripts/                       # Utility scripts
│   └── setup.sh                   # Local setup script
└── README.md                      # This file
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- 4GB+ RAM available for Docker
- Ports 3000, 5432, 8080, 9090, 9091 available

### Local Setup

1. **Clone and navigate to orchestration directory:**
   ```bash
   cd orchestration
   ```

2. **Run the setup script:**
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure environment:**
   ```bash
   cp config/env.example config/.env
   # Edit config/.env with your settings
   ```

4. **Access services:**
   - Airflow: http://localhost:8080 (admin/admin)
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Health Check: http://localhost:9091/health

### Manual Docker Setup

```bash
cd docker

# Set Airflow user ID
echo "AIRFLOW_UID=$(id -u)" > .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Configuration

### Environment Variables

Copy `config/env.example` to `config/.env` and configure:

```bash
# Required
DB_HOST=localhost
DB_PORT=5432
DB_NAME=investment_signals
DB_USER=signals
DB_PASSWORD=your_secure_password

# Alerting (optional but recommended)
EMAIL_ALERTS_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_RECIPIENTS=analyst@company.com

SLACK_ALERTS_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#investment-signals
```

### Alert Configuration

#### Email Alerts
- Sent for high-confidence signals (score >= 7.0)
- Daily digest summaries
- Pipeline failure notifications

#### Slack Alerts
- Real-time Form 4 notifications
- High-priority signal alerts
- System health warnings

#### SMS Alerts (Twilio)
- Critical signals only (score >= 9.0)
- System failures
- Requires Twilio account

## Airflow DAGs

### 1. Clinical Trial Signal Detection

**DAG ID:** `clinical_trial_signal_detection`
**Schedule:** Daily at 6:00 PM ET

**Tasks:**
1. `health_check` - Verify system health
2. `data_fetching` - Parallel fetch from ClinicalTrials.gov and SEC
3. `run_change_detection` - Detect trial changes
4. `calculate_scores` - Score signals
5. `alerting` - Send alerts and daily digest
6. `record_metrics` - Log execution metrics

### 2. Patent/IP Intelligence

**DAG ID:** `patent_ip_intelligence`
**Schedule:** Weekly on Monday at 8:00 AM ET

**Tasks:**
1. `extraction` - Extract FDA Orange Book data
2. `enrichment` - Enrich with USPTO and ANDA data
3. `generate_calendar` - Create patent cliff calendar
4. `score_opportunities` - Score investment opportunities
5. `load_to_database` - Store results
6. `output` - Send alerts and export reports

### 3. Insider/Hiring Signals

**DAG IDs:**
- `insider_form4_signals` - Every 30 min during market hours
- `institutional_13f_signals` - Quarterly
- `hiring_signals` - Daily at 9:00 AM ET
- `insider_hiring_master` - Manual trigger for all

## Monitoring

### Health Checks

Access health endpoints:
```bash
# Overall health
curl http://localhost:9091/health

# Pipeline-specific
curl http://localhost:9091/health/clinical_trials
curl http://localhost:9091/health/patent_ip
curl http://localhost:9091/health/insider_hiring

# Kubernetes probes
curl http://localhost:9091/ready
curl http://localhost:9091/live
```

### Prometheus Metrics

Metrics available at `http://localhost:9091/metrics`:

| Metric | Description |
|--------|-------------|
| `pipeline_executions_total` | Total pipeline executions by status |
| `pipeline_duration_seconds` | Pipeline execution duration |
| `signals_detected_total` | Total signals detected by type |
| `alerts_sent_total` | Total alerts sent by channel |
| `health_check_status` | Health check status (1=healthy) |
| `health_check_duration_seconds` | Health check duration |

### Grafana Dashboards

Pre-configured dashboards:
- **Pipeline Overview** - Execution status, signals detected, alerts sent
- **Health Status** - System health, data freshness
- **Cost Tracking** - API usage, cloud resource costs

## Cloud Deployment

### AWS (Terraform)

```bash
cd cloud/aws/terraform

# Initialize
terraform init

# Plan
terraform plan -var="db_password=your_password" -var="airflow_admin_password=your_password"

# Apply
terraform apply -var="db_password=your_password" -var="airflow_admin_password=your_password"
```

**Resources created:**
- ECS Fargate cluster (using spot instances)
- RDS PostgreSQL (t3.micro)
- ElastiCache Redis (t3.micro)
- S3 bucket for artifacts
- Secrets Manager for credentials
- CloudWatch for logging

### GCP (Terraform)

```bash
cd cloud/gcp/terraform

# Initialize
terraform init

# Plan
terraform plan -var="project_id=your-project" -var="db_password=your_password" -var="airflow_admin_password=your_password"

# Apply
terraform apply -var="project_id=your-project" -var="db_password=your_password" -var="airflow_admin_password=your_password"
```

**Resources created:**
- Cloud Run services for Airflow
- Cloud SQL PostgreSQL (db-f1-micro)
- Cloud Memorystore Redis (basic tier)
- Cloud Storage bucket
- Secret Manager for credentials

## Cost Estimates

### Local Development
- Free (Docker on local machine)

### AWS (Minimal Production)
| Resource | Monthly Cost |
|----------|-------------|
| ECS Fargate (spot) | $30-50 |
| RDS t3.micro | $15-20 |
| ElastiCache t3.micro | $10-15 |
| S3 | $1-5 |
| Secrets Manager | $1-2 |
| CloudWatch | $5-10 |
| **Total** | **$60-100** |

### GCP (Minimal Production)
| Resource | Monthly Cost |
|----------|-------------|
| Cloud Run | $20-40 |
| Cloud SQL db-f1-micro | $15-25 |
| Memorystore basic | $10-15 |
| Cloud Storage | $1-5 |
| Secret Manager | $1-2 |
| **Total** | **$50-100** |

## Alerting System

### Alert Types

| Type | Channels | Trigger |
|------|----------|---------|
| High-Confidence Signal | Email, Slack | Score >= 7.0 |
| Critical Signal | Email, Slack, SMS | Score >= 9.0 |
| Pipeline Failure | Email | Any task failure |
| Daily Digest | Email | End of pipeline run |
| Real-time Form 4 | Slack | Significant insider activity |

### Alert Deduplication

- Same signal won't alert twice within 24 hours
- Configurable via `ALERT_DEDUP_WINDOW_HOURS`
- Prevents alert fatigue

## Troubleshooting

### Common Issues

**Airflow not starting:**
```bash
# Check logs
docker-compose logs airflow-scheduler

# Verify database connection
docker-compose exec postgres pg_isready -U airflow
```

**DAGs not appearing:**
```bash
# Check DAG folder mounting
docker-compose exec airflow-scheduler ls /opt/airflow/dags

# Check for import errors
docker-compose exec airflow-scheduler airflow dags list-import-errors
```

**Database connection failed:**
```bash
# Check database status
docker-compose ps postgres

# Verify credentials
docker-compose exec postgres psql -U signals -d investment_signals -c "SELECT 1"
```

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f airflow-scheduler
docker-compose logs -f airflow-worker

# Tail last 100 lines
docker-compose logs --tail=100 airflow-scheduler
```

### Reset Environment

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Start fresh
docker-compose up -d
```

## Development

### Adding a New DAG

1. Create DAG file in `dags/`:
   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator

   with DAG(
       dag_id="my_new_dag",
       schedule_interval="0 8 * * *",
       ...
   ) as dag:
       # Define tasks
   ```

2. Test locally:
   ```bash
   # Validate DAG
   docker-compose exec airflow-scheduler airflow dags test my_new_dag 2024-01-01
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_health_checks.py

# With coverage
pytest --cov=orchestration
```

## Security

### Best Practices

1. **Secrets Management:**
   - Never commit `.env` files
   - Use AWS Secrets Manager or GCP Secret Manager in production
   - Rotate credentials regularly

2. **Network Security:**
   - Use VPC for cloud deployments
   - Restrict database access to private subnets
   - Enable TLS for all connections

3. **Access Control:**
   - Change default Airflow passwords
   - Use IAM roles for cloud resources
   - Implement RBAC in Airflow (Enterprise)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review Airflow logs
3. Check health endpoints
4. Review Grafana dashboards for anomalies

## License

Internal use only. All rights reserved.
