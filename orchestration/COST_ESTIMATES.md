# Investment Signals Orchestration - Cost Estimates

This document provides detailed cost estimates for running the orchestration system across different deployment options.

## Summary

| Deployment Option | Monthly Cost | Best For |
|------------------|--------------|----------|
| Local (Docker) | $0 | Development, testing |
| AWS Minimal | $60-100 | Startups, small teams |
| AWS Production | $200-400 | Production workloads |
| GCP Minimal | $50-100 | Startups, small teams |
| GCP Production | $150-350 | Production workloads |

## Local Development (Docker Compose)

**Cost: $0/month**

Resources used:
- Local Docker containers
- SQLite or local PostgreSQL
- No cloud services

Requirements:
- Machine with 4GB+ RAM
- Docker Desktop

---

## AWS Deployment

### Minimal Configuration (Dev/Staging)

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| **ECS Fargate (Spot)** | | |
| - Airflow Webserver | 0.5 vCPU, 1GB RAM | $8-12 |
| - Airflow Scheduler | 0.5 vCPU, 1GB RAM | $8-12 |
| - Airflow Worker | 1 vCPU, 2GB RAM | $15-25 |
| **RDS PostgreSQL** | db.t3.micro, 20GB | $15-20 |
| **ElastiCache Redis** | cache.t3.micro | $10-15 |
| **S3** | ~10GB storage | $1-3 |
| **Secrets Manager** | 3-5 secrets | $1-2 |
| **CloudWatch** | Logs + basic monitoring | $5-10 |
| **Data Transfer** | ~10GB/month | $1-2 |
| **Total** | | **$60-100** |

### Production Configuration

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| **ECS Fargate** | | |
| - Airflow Webserver | 1 vCPU, 2GB RAM, 2 tasks | $40-60 |
| - Airflow Scheduler | 1 vCPU, 2GB RAM | $20-30 |
| - Airflow Workers | 2 vCPU, 4GB RAM, 2 tasks | $80-120 |
| - Health Check Service | 0.25 vCPU, 0.5GB RAM | $5-8 |
| **RDS PostgreSQL** | db.t3.small, Multi-AZ | $40-60 |
| **ElastiCache Redis** | cache.t3.small, cluster | $25-35 |
| **S3** | ~50GB storage + versioning | $5-10 |
| **Secrets Manager** | 5-10 secrets | $3-5 |
| **CloudWatch** | Enhanced monitoring | $15-25 |
| **ALB** | Application Load Balancer | $20-30 |
| **Data Transfer** | ~50GB/month | $5-10 |
| **Total** | | **$250-400** |

### Cost Optimization Tips (AWS)

1. **Use Spot Instances for Fargate:**
   - Save up to 70% on compute
   - Configure `FARGATE_SPOT` capacity provider
   - Good for non-critical workloads

2. **Reserved Instances for RDS:**
   - Save up to 40% with 1-year commitment
   - Good for production databases

3. **S3 Lifecycle Policies:**
   - Move old logs to Glacier
   - Delete logs after 30 days

4. **Right-size Resources:**
   - Start with t3.micro, scale as needed
   - Monitor CloudWatch metrics

---

## GCP Deployment

### Minimal Configuration (Dev/Staging)

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| **Cloud Run** | | |
| - Airflow Webserver | 1 vCPU, 1GB RAM, min 0 | $10-20 |
| - Airflow Scheduler | 1 vCPU, 1GB RAM, min 0 | $10-20 |
| - Health Check Service | 0.5 vCPU, 0.5GB RAM | $5-10 |
| **Cloud SQL** | db-f1-micro, 20GB | $15-25 |
| **Memorystore Redis** | Basic, 1GB | $10-15 |
| **Cloud Storage** | ~10GB | $1-2 |
| **Secret Manager** | 3-5 secrets | $1-2 |
| **Cloud Monitoring** | Basic | $0 (free tier) |
| **VPC Connector** | f1-micro instances | $5-10 |
| **Total** | | **$50-100** |

### Production Configuration

| Resource | Specification | Monthly Cost |
|----------|--------------|--------------|
| **Cloud Run** | | |
| - Airflow Webserver | 2 vCPU, 2GB RAM, min 1 | $40-60 |
| - Airflow Scheduler | 2 vCPU, 2GB RAM, min 1 | $40-60 |
| - Workers (Cloud Run Jobs) | 2 vCPU, 4GB RAM | $30-50 |
| - Health Check Service | 1 vCPU, 1GB RAM | $15-25 |
| **Cloud SQL** | db-custom-1-3840, HA | $60-80 |
| **Memorystore Redis** | Standard, 2GB | $30-40 |
| **Cloud Storage** | ~50GB | $2-5 |
| **Secret Manager** | 5-10 secrets | $2-3 |
| **Cloud Monitoring** | Enhanced | $10-20 |
| **Cloud Armor** | WAF rules | $5-10 |
| **Total** | | **$230-350** |

### Cost Optimization Tips (GCP)

1. **Use Committed Use Discounts:**
   - Save up to 57% on Cloud SQL
   - 1 or 3-year commitments

2. **Cloud Run Scale to Zero:**
   - Set `min_instance_count = 0` for dev
   - Only pay when services are active

3. **Use Preemptible VMs:**
   - For batch processing jobs
   - Save up to 80%

4. **Cloud Storage Lifecycle:**
   - Archive old data to Coldline
   - Delete after retention period

---

## API Costs

External API costs (not included above):

| API | Cost | Notes |
|-----|------|-------|
| SEC EDGAR | Free | Rate limited (10 req/sec) |
| ClinicalTrials.gov | Free | Rate limited |
| FDA Orange Book | Free | |
| USPTO | Free | API key required |
| LinkedIn Jobs | Variable | Depends on plan |
| Twilio SMS | $0.0075/msg | Only for critical alerts |

---

## Monthly Cost by Pipeline Frequency

| Pipeline | Frequency | Est. Compute/month |
|----------|-----------|-------------------|
| Clinical Trials | Daily (6 PM) | 30 runs x 15 min = 7.5 hrs |
| Patent/IP | Weekly (Mon) | 4 runs x 60 min = 4 hrs |
| Form 4 | Every 30 min (market hours) | ~260 runs x 5 min = 21.6 hrs |
| 13F | Quarterly | 4 runs x 120 min = 8 hrs |
| Job Postings | Daily | 30 runs x 30 min = 15 hrs |
| **Total** | | **~56 hrs/month** |

---

## Recommendations by Team Size

### Solo Developer / Startup
- **Deployment:** Local Docker or GCP minimal
- **Monthly Budget:** $0-50
- **Configuration:**
  - Cloud Run scale-to-zero
  - db-f1-micro database
  - Basic Redis tier

### Small Team (2-5 people)
- **Deployment:** AWS/GCP minimal
- **Monthly Budget:** $60-150
- **Configuration:**
  - Spot/preemptible instances
  - t3.micro/db-f1-micro resources
  - Email alerts only

### Growth Team (5-15 people)
- **Deployment:** AWS/GCP standard
- **Monthly Budget:** $150-300
- **Configuration:**
  - Reserved instances for DB
  - t3.small resources
  - Full alerting (Email + Slack)

### Enterprise (15+ people)
- **Deployment:** AWS/GCP production
- **Monthly Budget:** $300-500+
- **Configuration:**
  - Multi-AZ databases
  - Auto-scaling workers
  - Full observability stack
  - SLA monitoring

---

## Break-Even Analysis

Assuming the system generates 1 actionable signal per week with average ROI:

| Signal Value | Annual Value | Break-Even Cost |
|-------------|--------------|-----------------|
| $1,000/signal | $52,000 | $4,333/month |
| $5,000/signal | $260,000 | $21,666/month |
| $10,000/signal | $520,000 | $43,333/month |

Even at minimal deployment costs ($60-100/month), the system pays for itself with just one valuable signal per year.

---

## Monitoring Costs

To track actual costs:

1. **AWS Cost Explorer:**
   - Enable cost allocation tags
   - Set up budget alerts
   - Review weekly

2. **GCP Billing:**
   - Export to BigQuery
   - Create cost dashboards
   - Set budget alerts

3. **In-App Tracking:**
   - Use `CostTracker` class in monitoring module
   - Review cost breakdown endpoint

```python
from orchestration.monitoring import CostTracker

tracker = CostTracker()
breakdown = tracker.get_cost_breakdown()
print(f"Daily: ${breakdown['daily']['total']:.2f}")
print(f"Monthly: ${breakdown['monthly']['total']:.2f}")
```
