#!/bin/bash
# =============================================================================
# GCP Deployment Script for Investment Signals Orchestration
# =============================================================================
#
# This script deploys the Investment Signals Orchestration platform to GCP.
# It supports:
# - Blue-green deployment using Cloud Run revisions
# - Health checks before traffic switching
# - Automatic rollback on failure
# - Environment-specific configurations
#
# Prerequisites:
# - gcloud CLI configured with appropriate credentials
# - Docker installed and running
# - Terraform >= 1.5 (for infrastructure provisioning)
# - jq for JSON parsing
#
# Usage:
#   ./deploy_gcp.sh [environment] [action]
#
# Arguments:
#   environment: dev, staging, prod (default: dev)
#   action: deploy, rollback, status, health (default: deploy)
#
# Examples:
#   ./deploy_gcp.sh prod deploy      # Deploy to production
#   ./deploy_gcp.sh staging rollback # Rollback staging
#   ./deploy_gcp.sh prod health      # Check production health
#
# Environment Variables:
#   GCP_PROJECT         - GCP project ID (required)
#   GCP_REGION          - GCP region (default: us-central1)
#   GCR_REPOSITORY      - Artifact Registry repository (default: investment-signals)
#   CLOUD_RUN_SERVICE   - Cloud Run service name (default: orchestration-{env})
#   DEPLOYMENT_TIMEOUT  - Deployment timeout in seconds (default: 600)
#   HEALTH_CHECK_RETRIES - Number of health check retries (default: 30)
#
# =============================================================================

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="${1:-dev}"
ACTION="${2:-deploy}"
GCP_REGION="${GCP_REGION:-us-central1}"
GCR_REPOSITORY="${GCR_REPOSITORY:-investment-signals}"
DEPLOYMENT_TIMEOUT="${DEPLOYMENT_TIMEOUT:-600}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-30}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-10}"

# Derived values
CLOUD_RUN_SERVICE="${CLOUD_RUN_SERVICE:-orchestration-${ENVIRONMENT}}"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
DEPLOYMENT_ID="deploy-${TIMESTAMP}"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed. Please install it first."
        exit 1
    fi

    # Check GCP authentication
    if ! gcloud auth print-identity-token &> /dev/null; then
        log_error "GCP credentials are not configured. Please run 'gcloud auth login'."
        exit 1
    fi

    # Get GCP Project ID if not set
    if [ -z "${GCP_PROJECT:-}" ]; then
        GCP_PROJECT=$(gcloud config get-value project 2>/dev/null)
        if [ -z "$GCP_PROJECT" ]; then
            log_error "GCP_PROJECT is not set. Please set it or run 'gcloud config set project PROJECT_ID'."
            exit 1
        fi
    fi

    # Set Artifact Registry path
    GAR_REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCR_REPOSITORY}"

    log_success "Prerequisites check passed"
    log_info "Project: ${GCP_PROJECT}"
    log_info "Region: ${GCP_REGION}"
}

validate_environment() {
    log_info "Validating environment: ${ENVIRONMENT}"

    case $ENVIRONMENT in
        dev|staging|prod)
            log_success "Valid environment: ${ENVIRONMENT}"
            ;;
        *)
            log_error "Invalid environment: ${ENVIRONMENT}. Must be one of: dev, staging, prod"
            exit 1
            ;;
    esac
}

# =============================================================================
# Docker Build and Push Functions
# =============================================================================

build_docker_images() {
    log_info "Building Docker images..."

    cd "${PROJECT_ROOT}/docker"

    # Build Airflow image
    log_info "Building Airflow image..."
    docker build -t "${GAR_REGISTRY}/airflow:${DEPLOYMENT_ID}" \
        -f Dockerfile.airflow \
        --build-arg ENVIRONMENT="${ENVIRONMENT}" \
        .

    # Build healthcheck image
    log_info "Building healthcheck image..."
    docker build -t "${GAR_REGISTRY}/healthcheck:${DEPLOYMENT_ID}" \
        -f Dockerfile.healthcheck \
        .

    log_success "Docker images built successfully"
}

push_docker_images() {
    log_info "Pushing Docker images to Artifact Registry..."

    # Configure Docker for Artifact Registry
    gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

    # Create repository if it doesn't exist
    gcloud artifacts repositories describe "${GCR_REPOSITORY}" \
        --location="${GCP_REGION}" \
        --project="${GCP_PROJECT}" 2>/dev/null || \
    gcloud artifacts repositories create "${GCR_REPOSITORY}" \
        --repository-format=docker \
        --location="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --description="Investment Signals container images"

    # Push images
    docker push "${GAR_REGISTRY}/airflow:${DEPLOYMENT_ID}"
    docker push "${GAR_REGISTRY}/healthcheck:${DEPLOYMENT_ID}"

    # Tag as latest for environment
    docker tag "${GAR_REGISTRY}/airflow:${DEPLOYMENT_ID}" "${GAR_REGISTRY}/airflow:${ENVIRONMENT}-latest"
    docker tag "${GAR_REGISTRY}/healthcheck:${DEPLOYMENT_ID}" "${GAR_REGISTRY}/healthcheck:${ENVIRONMENT}-latest"
    docker push "${GAR_REGISTRY}/airflow:${ENVIRONMENT}-latest"
    docker push "${GAR_REGISTRY}/healthcheck:${ENVIRONMENT}-latest"

    log_success "Docker images pushed to Artifact Registry"
}

# =============================================================================
# Cloud Run Deployment Functions
# =============================================================================

get_current_revision() {
    log_info "Getting current revision..."

    CURRENT_REVISION=$(gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='value(status.latestReadyRevisionName)' 2>/dev/null || echo "")

    if [ -z "$CURRENT_REVISION" ]; then
        log_warning "No existing revision found. This may be a new deployment."
    else
        log_info "Current revision: ${CURRENT_REVISION}"
    fi
}

deploy_cloud_run() {
    log_info "Deploying to Cloud Run..."

    # Load environment-specific config
    ENV_FILE="${PROJECT_ROOT}/config/${ENVIRONMENT}.env"
    ENV_VARS=""
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment variables from ${ENV_FILE}"
        ENV_VARS=$(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
    fi

    # Deploy main service (Airflow + Orchestration)
    log_info "Deploying orchestration service..."
    gcloud run deploy "${CLOUD_RUN_SERVICE}" \
        --image="${GAR_REGISTRY}/airflow:${DEPLOYMENT_ID}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --platform=managed \
        --port=8080 \
        --cpu=2 \
        --memory=4Gi \
        --min-instances=1 \
        --max-instances=10 \
        --timeout=3600 \
        --concurrency=80 \
        --set-env-vars="ENVIRONMENT=${ENVIRONMENT},DEPLOYMENT_ID=${DEPLOYMENT_ID}${ENV_VARS:+,$ENV_VARS}" \
        --allow-unauthenticated \
        --no-traffic \
        --tag="${DEPLOYMENT_ID}" \
        --labels="environment=${ENVIRONMENT},deployment-id=${DEPLOYMENT_ID}"

    # Deploy healthcheck service
    log_info "Deploying healthcheck service..."
    gcloud run deploy "${CLOUD_RUN_SERVICE}-healthcheck" \
        --image="${GAR_REGISTRY}/healthcheck:${DEPLOYMENT_ID}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --platform=managed \
        --port=9091 \
        --cpu=1 \
        --memory=512Mi \
        --min-instances=1 \
        --max-instances=3 \
        --timeout=60 \
        --concurrency=100 \
        --set-env-vars="ENVIRONMENT=${ENVIRONMENT}" \
        --allow-unauthenticated \
        --labels="environment=${ENVIRONMENT},deployment-id=${DEPLOYMENT_ID}"

    log_success "Services deployed to Cloud Run"
}

# =============================================================================
# Traffic Management Functions
# =============================================================================

switch_traffic() {
    log_info "Switching traffic to new revision..."

    # Get new revision name
    NEW_REVISION=$(gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='value(status.latestCreatedRevisionName)')

    log_info "Switching 100% traffic to: ${NEW_REVISION}"

    gcloud run services update-traffic "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --to-revisions="${NEW_REVISION}=100"

    log_success "Traffic switched to new revision"
}

gradual_traffic_shift() {
    log_info "Starting gradual traffic shift..."

    NEW_REVISION=$(gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='value(status.latestCreatedRevisionName)')

    # Traffic percentages for gradual rollout
    TRAFFIC_STEPS=(10 25 50 75 100)

    for percentage in "${TRAFFIC_STEPS[@]}"; do
        log_info "Shifting ${percentage}% traffic to ${NEW_REVISION}..."

        gcloud run services update-traffic "${CLOUD_RUN_SERVICE}" \
            --region="${GCP_REGION}" \
            --project="${GCP_PROJECT}" \
            --to-revisions="${NEW_REVISION}=${percentage}"

        # Wait and check health
        sleep 30

        if ! check_health_quick; then
            log_error "Health check failed at ${percentage}% traffic. Rolling back..."
            rollback
            exit 1
        fi

        log_success "Health check passed at ${percentage}% traffic"
    done

    log_success "Gradual traffic shift completed"
}

# =============================================================================
# Health Check Functions
# =============================================================================

get_service_url() {
    log_info "Getting service URL..."

    SERVICE_URL=$(gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='value(status.url)' 2>/dev/null || echo "")

    HEALTHCHECK_URL=$(gcloud run services describe "${CLOUD_RUN_SERVICE}-healthcheck" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='value(status.url)' 2>/dev/null || echo "")

    if [ -z "$SERVICE_URL" ]; then
        log_warning "Could not determine service URL"
        return 1
    fi

    log_info "Service URL: ${SERVICE_URL}"
    log_info "Healthcheck URL: ${HEALTHCHECK_URL}"
}

check_health_quick() {
    # Quick health check for gradual rollout
    local timeout=60
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        HC_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
            "${HEALTHCHECK_URL}/health" 2>/dev/null || echo "000")

        if [ "$HC_RESPONSE" == "200" ]; then
            return 0
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    return 1
}

check_health() {
    log_info "Running health checks..."

    get_service_url || return 1

    local retries=$HEALTH_CHECK_RETRIES
    local count=0

    while [ $count -lt $retries ]; do
        # Check main service health
        MAIN_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
            "${SERVICE_URL}/health" 2>/dev/null || echo "000")

        # Check healthcheck service
        HC_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
            "${HEALTHCHECK_URL}/health" 2>/dev/null || echo "000")

        if [ "$MAIN_HEALTH" == "200" ] && [ "$HC_HEALTH" == "200" ]; then
            log_success "All health checks passed!"

            # Get detailed health status
            log_info "Detailed health status:"
            curl -s "${HEALTHCHECK_URL}/health" | jq '.' 2>/dev/null || true

            return 0
        fi

        log_info "Health check attempt $((count + 1))/${retries}: Main=${MAIN_HEALTH}, HealthCheck=${HC_HEALTH}"
        sleep $HEALTH_CHECK_INTERVAL
        count=$((count + 1))
    done

    log_error "Health checks failed after ${retries} attempts"
    return 1
}

# =============================================================================
# Rollback Functions
# =============================================================================

rollback() {
    log_warning "Initiating rollback..."

    # Get previous revision
    REVISIONS=$(gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='value(status.traffic.revisionName)' 2>/dev/null | head -2)

    PREVIOUS_REVISION=$(echo "$REVISIONS" | tail -1)

    if [ -z "$PREVIOUS_REVISION" ]; then
        log_error "No previous revision found for rollback"
        exit 1
    fi

    log_info "Rolling back to: ${PREVIOUS_REVISION}"

    # Switch all traffic to previous revision
    gcloud run services update-traffic "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --to-revisions="${PREVIOUS_REVISION}=100"

    log_success "Rollback completed. Traffic now serving from: ${PREVIOUS_REVISION}"
}

# =============================================================================
# Status Functions
# =============================================================================

show_status() {
    log_info "Getting deployment status for ${ENVIRONMENT}..."

    echo ""
    echo "=== Cloud Run Service Status ==="
    gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='table(
            status.url,
            status.latestReadyRevisionName,
            status.conditions[0].status,
            metadata.labels.environment,
            metadata.labels.deployment-id
        )' 2>/dev/null || echo "Service not found"

    echo ""
    echo "=== Traffic Distribution ==="
    gcloud run services describe "${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --format='table(
            status.traffic.revisionName,
            status.traffic.percent,
            status.traffic.latestRevision
        )' 2>/dev/null || echo "No traffic information available"

    echo ""
    echo "=== Recent Revisions ==="
    gcloud run revisions list \
        --service="${CLOUD_RUN_SERVICE}" \
        --region="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --limit=5 \
        --format='table(
            metadata.name,
            status.conditions[0].status,
            metadata.creationTimestamp
        )' 2>/dev/null || echo "No revisions found"

    echo ""
}

# =============================================================================
# Cloud SQL Setup (for production database)
# =============================================================================

setup_cloud_sql() {
    log_info "Setting up Cloud SQL instance..."

    INSTANCE_NAME="investment-signals-${ENVIRONMENT}"

    # Check if instance exists
    if gcloud sql instances describe "$INSTANCE_NAME" --project="${GCP_PROJECT}" &>/dev/null; then
        log_info "Cloud SQL instance already exists"
        return 0
    fi

    log_info "Creating Cloud SQL instance..."
    gcloud sql instances create "$INSTANCE_NAME" \
        --project="${GCP_PROJECT}" \
        --region="${GCP_REGION}" \
        --database-version=POSTGRES_14 \
        --tier=db-f1-micro \
        --storage-size=10GB \
        --storage-type=SSD \
        --backup-start-time=03:00 \
        --availability-type=zonal \
        --root-password="${DB_PASSWORD:-changeme}"

    # Create databases
    gcloud sql databases create airflow --instance="$INSTANCE_NAME" --project="${GCP_PROJECT}"
    gcloud sql databases create investment_signals --instance="$INSTANCE_NAME" --project="${GCP_PROJECT}"

    log_success "Cloud SQL setup completed"
}

# =============================================================================
# Main Deployment Function
# =============================================================================

deploy() {
    log_info "Starting deployment to ${ENVIRONMENT}..."
    log_info "Deployment ID: ${DEPLOYMENT_ID}"

    # Build and push images
    build_docker_images
    push_docker_images

    # Get current state
    get_current_revision

    # Deploy to Cloud Run
    deploy_cloud_run

    # Verify health before switching traffic
    log_info "Waiting for services to be ready..."
    sleep 30

    if check_health; then
        # For production, use gradual traffic shift
        if [ "$ENVIRONMENT" == "prod" ]; then
            gradual_traffic_shift
        else
            switch_traffic
        fi

        log_success "Deployment to ${ENVIRONMENT} completed successfully!"
        log_info "Deployment ID: ${DEPLOYMENT_ID}"

        # Output service URLs
        echo ""
        echo "=== Service URLs ==="
        echo "Main Service:     ${SERVICE_URL}"
        echo "Health Check:     ${HEALTHCHECK_URL}"
        echo ""
        echo "Access Airflow at: ${SERVICE_URL}"
    else
        log_error "Health checks failed. Keeping traffic on previous revision."
        log_info "New revision deployed but not receiving traffic."
        log_info "Run './deploy_gcp.sh ${ENVIRONMENT} rollback' to clean up or investigate."
        exit 1
    fi
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    echo "=============================================="
    echo " Investment Signals - GCP Deployment Script"
    echo "=============================================="
    echo ""

    validate_environment
    check_prerequisites

    case $ACTION in
        deploy)
            deploy
            ;;
        rollback)
            rollback
            ;;
        status)
            show_status
            ;;
        health)
            check_health
            ;;
        setup-db)
            setup_cloud_sql
            ;;
        *)
            log_error "Invalid action: ${ACTION}. Must be one of: deploy, rollback, status, health, setup-db"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
