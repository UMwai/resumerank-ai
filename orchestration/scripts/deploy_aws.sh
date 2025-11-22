#!/bin/bash
# =============================================================================
# AWS Deployment Script for Investment Signals Orchestration
# =============================================================================
#
# This script deploys the Investment Signals Orchestration platform to AWS.
# It supports:
# - Blue-green deployment for zero-downtime updates
# - Health checks before traffic switching
# - Automatic rollback on failure
# - Environment-specific configurations
#
# Prerequisites:
# - AWS CLI v2 configured with appropriate credentials
# - Docker installed and running
# - Terraform >= 1.5 (for infrastructure provisioning)
# - jq for JSON parsing
#
# Usage:
#   ./deploy_aws.sh [environment] [action]
#
# Arguments:
#   environment: dev, staging, prod (default: dev)
#   action: deploy, rollback, status, health (default: deploy)
#
# Examples:
#   ./deploy_aws.sh prod deploy      # Deploy to production
#   ./deploy_aws.sh staging rollback # Rollback staging
#   ./deploy_aws.sh prod health      # Check production health
#
# Environment Variables:
#   AWS_REGION          - AWS region (default: us-east-1)
#   AWS_ACCOUNT_ID      - AWS account ID (required)
#   ECR_REPOSITORY      - ECR repository name (default: investment-signals)
#   ECS_CLUSTER         - ECS cluster name (default: investment-signals-{env})
#   ECS_SERVICE         - ECS service name (default: orchestration-{env})
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
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-investment-signals}"
DEPLOYMENT_TIMEOUT="${DEPLOYMENT_TIMEOUT:-600}"
HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-30}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-10}"

# Derived values
ECS_CLUSTER="${ECS_CLUSTER:-investment-signals-${ENVIRONMENT}}"
ECS_SERVICE="${ECS_SERVICE:-orchestration-${ENVIRONMENT}}"
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

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
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

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured. Please run 'aws configure'."
        exit 1
    fi

    # Get AWS Account ID if not set
    if [ -z "${AWS_ACCOUNT_ID:-}" ]; then
        AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    fi

    ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY}"

    log_success "Prerequisites check passed"
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
    docker build -t "${ECR_IMAGE}:airflow-${DEPLOYMENT_ID}" \
        -f Dockerfile.airflow \
        --build-arg ENVIRONMENT="${ENVIRONMENT}" \
        .

    # Build healthcheck image
    log_info "Building healthcheck image..."
    docker build -t "${ECR_IMAGE}:healthcheck-${DEPLOYMENT_ID}" \
        -f Dockerfile.healthcheck \
        .

    log_success "Docker images built successfully"
}

push_docker_images() {
    log_info "Pushing Docker images to ECR..."

    # Login to ECR
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_REGISTRY}"

    # Create repository if it doesn't exist
    aws ecr describe-repositories --repository-names "${ECR_REPOSITORY}" --region "${AWS_REGION}" 2>/dev/null || \
        aws ecr create-repository --repository-name "${ECR_REPOSITORY}" --region "${AWS_REGION}"

    # Push images
    docker push "${ECR_IMAGE}:airflow-${DEPLOYMENT_ID}"
    docker push "${ECR_IMAGE}:healthcheck-${DEPLOYMENT_ID}"

    # Tag as latest for environment
    docker tag "${ECR_IMAGE}:airflow-${DEPLOYMENT_ID}" "${ECR_IMAGE}:airflow-${ENVIRONMENT}-latest"
    docker tag "${ECR_IMAGE}:healthcheck-${DEPLOYMENT_ID}" "${ECR_IMAGE}:healthcheck-${ENVIRONMENT}-latest"
    docker push "${ECR_IMAGE}:airflow-${ENVIRONMENT}-latest"
    docker push "${ECR_IMAGE}:healthcheck-${ENVIRONMENT}-latest"

    log_success "Docker images pushed to ECR"
}

# =============================================================================
# ECS Deployment Functions
# =============================================================================

get_current_task_definition() {
    log_info "Getting current task definition..."

    CURRENT_TASK_DEF=$(aws ecs describe-services \
        --cluster "${ECS_CLUSTER}" \
        --services "${ECS_SERVICE}" \
        --region "${AWS_REGION}" \
        --query 'services[0].taskDefinition' \
        --output text 2>/dev/null || echo "")

    if [ -z "$CURRENT_TASK_DEF" ] || [ "$CURRENT_TASK_DEF" == "None" ]; then
        log_warning "No existing task definition found. This may be a new deployment."
        CURRENT_TASK_DEF=""
    else
        log_info "Current task definition: ${CURRENT_TASK_DEF}"
    fi
}

create_task_definition() {
    log_info "Creating new task definition..."

    # Generate task definition from template
    TASK_DEF_FILE="${PROJECT_ROOT}/cloud/aws/task-definition-${ENVIRONMENT}.json"

    if [ ! -f "$TASK_DEF_FILE" ]; then
        log_info "Task definition template not found, creating from default..."
        TASK_DEF_FILE="${PROJECT_ROOT}/cloud/aws/task-definition.json"
    fi

    # Update task definition with new image tags
    NEW_TASK_DEF=$(cat "$TASK_DEF_FILE" | \
        jq --arg img "${ECR_IMAGE}:airflow-${DEPLOYMENT_ID}" \
           --arg hc_img "${ECR_IMAGE}:healthcheck-${DEPLOYMENT_ID}" \
           --arg env "${ENVIRONMENT}" \
           '.containerDefinitions[0].image = $img |
            .containerDefinitions[] |= if .name == "healthcheck" then .image = $hc_img else . end |
            .tags += [{"key": "Environment", "value": $env}, {"key": "DeploymentId", "value": "'"${DEPLOYMENT_ID}"'"}]')

    # Register new task definition
    REGISTERED_TASK_DEF=$(aws ecs register-task-definition \
        --cli-input-json "$NEW_TASK_DEF" \
        --region "${AWS_REGION}" \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)

    log_success "New task definition registered: ${REGISTERED_TASK_DEF}"
}

deploy_blue_green() {
    log_info "Starting blue-green deployment..."

    # Get current deployment
    CURRENT_DEPLOYMENT=$(aws ecs describe-services \
        --cluster "${ECS_CLUSTER}" \
        --services "${ECS_SERVICE}" \
        --region "${AWS_REGION}" \
        --query 'services[0].deployments[?status==`PRIMARY`].id' \
        --output text 2>/dev/null || echo "")

    log_info "Current deployment: ${CURRENT_DEPLOYMENT:-none}"

    # Update service with new task definition
    log_info "Updating ECS service with new task definition..."
    aws ecs update-service \
        --cluster "${ECS_CLUSTER}" \
        --service "${ECS_SERVICE}" \
        --task-definition "${REGISTERED_TASK_DEF}" \
        --region "${AWS_REGION}" \
        --deployment-configuration "maximumPercent=200,minimumHealthyPercent=100" \
        --force-new-deployment \
        > /dev/null

    log_info "Waiting for deployment to stabilize..."
    wait_for_deployment_stable
}

wait_for_deployment_stable() {
    local timeout=$DEPLOYMENT_TIMEOUT
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        DEPLOYMENT_STATUS=$(aws ecs describe-services \
            --cluster "${ECS_CLUSTER}" \
            --services "${ECS_SERVICE}" \
            --region "${AWS_REGION}" \
            --query 'services[0].deployments[0].rolloutState' \
            --output text 2>/dev/null || echo "UNKNOWN")

        case $DEPLOYMENT_STATUS in
            "COMPLETED")
                log_success "Deployment completed successfully!"
                return 0
                ;;
            "FAILED")
                log_error "Deployment failed!"
                return 1
                ;;
            "IN_PROGRESS")
                log_info "Deployment in progress... (${elapsed}s elapsed)"
                ;;
            *)
                log_warning "Unknown deployment status: ${DEPLOYMENT_STATUS}"
                ;;
        esac

        sleep 10
        elapsed=$((elapsed + 10))
    done

    log_error "Deployment timed out after ${timeout} seconds"
    return 1
}

# =============================================================================
# Health Check Functions
# =============================================================================

get_service_url() {
    log_info "Getting service URL..."

    # Get load balancer DNS
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --names "investment-signals-${ENVIRONMENT}" \
        --region "${AWS_REGION}" \
        --query 'LoadBalancers[0].DNSName' \
        --output text 2>/dev/null || echo "")

    if [ -z "$ALB_DNS" ] || [ "$ALB_DNS" == "None" ]; then
        # Try to get from ECS service
        ALB_DNS=$(aws ecs describe-services \
            --cluster "${ECS_CLUSTER}" \
            --services "${ECS_SERVICE}" \
            --region "${AWS_REGION}" \
            --query 'services[0].loadBalancers[0].targetGroupArn' \
            --output text 2>/dev/null || echo "")

        if [ -z "$ALB_DNS" ] || [ "$ALB_DNS" == "None" ]; then
            log_warning "Could not determine service URL"
            return 1
        fi
    fi

    SERVICE_URL="http://${ALB_DNS}"
    log_info "Service URL: ${SERVICE_URL}"
}

check_health() {
    log_info "Running health checks..."

    get_service_url || return 1

    local retries=$HEALTH_CHECK_RETRIES
    local count=0

    while [ $count -lt $retries ]; do
        # Check Airflow health
        AIRFLOW_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
            "${SERVICE_URL}:8080/health" 2>/dev/null || echo "000")

        # Check healthcheck service
        HC_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
            "${SERVICE_URL}:9091/health" 2>/dev/null || echo "000")

        if [ "$AIRFLOW_HEALTH" == "200" ] && [ "$HC_HEALTH" == "200" ]; then
            log_success "All health checks passed!"

            # Get detailed health status
            log_info "Detailed health status:"
            curl -s "${SERVICE_URL}:9091/health" | jq '.' 2>/dev/null || true

            return 0
        fi

        log_info "Health check attempt $((count + 1))/${retries}: Airflow=${AIRFLOW_HEALTH}, HealthCheck=${HC_HEALTH}"
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

    # Get previous task definition
    PREVIOUS_TASK_DEF=$(aws ecs list-task-definitions \
        --family-prefix "${ECS_SERVICE}" \
        --region "${AWS_REGION}" \
        --sort DESC \
        --max-items 2 \
        --query 'taskDefinitionArns[1]' \
        --output text 2>/dev/null || echo "")

    if [ -z "$PREVIOUS_TASK_DEF" ] || [ "$PREVIOUS_TASK_DEF" == "None" ]; then
        log_error "No previous task definition found for rollback"
        exit 1
    fi

    log_info "Rolling back to: ${PREVIOUS_TASK_DEF}"

    # Update service with previous task definition
    aws ecs update-service \
        --cluster "${ECS_CLUSTER}" \
        --service "${ECS_SERVICE}" \
        --task-definition "${PREVIOUS_TASK_DEF}" \
        --region "${AWS_REGION}" \
        --force-new-deployment \
        > /dev/null

    log_info "Waiting for rollback to complete..."
    if wait_for_deployment_stable; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed"
        exit 1
    fi
}

# =============================================================================
# Status Functions
# =============================================================================

show_status() {
    log_info "Getting deployment status for ${ENVIRONMENT}..."

    # Get service status
    SERVICE_STATUS=$(aws ecs describe-services \
        --cluster "${ECS_CLUSTER}" \
        --services "${ECS_SERVICE}" \
        --region "${AWS_REGION}" \
        2>/dev/null || echo "{}")

    echo ""
    echo "=== ECS Service Status ==="
    echo "$SERVICE_STATUS" | jq '{
        serviceName: .services[0].serviceName,
        status: .services[0].status,
        runningCount: .services[0].runningCount,
        desiredCount: .services[0].desiredCount,
        taskDefinition: .services[0].taskDefinition,
        deployments: [.services[0].deployments[] | {
            id: .id,
            status: .status,
            rolloutState: .rolloutState,
            runningCount: .runningCount,
            desiredCount: .desiredCount
        }]
    }' 2>/dev/null || echo "Could not retrieve service status"

    echo ""
    echo "=== Recent Events ==="
    echo "$SERVICE_STATUS" | jq '.services[0].events[:5][] | {
        createdAt: .createdAt,
        message: .message
    }' 2>/dev/null || echo "Could not retrieve events"

    echo ""
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
    get_current_task_definition

    # Create new task definition
    create_task_definition

    # Deploy using blue-green strategy
    deploy_blue_green

    # Verify health
    if check_health; then
        log_success "Deployment to ${ENVIRONMENT} completed successfully!"
        log_info "Deployment ID: ${DEPLOYMENT_ID}"

        # Output service URLs
        echo ""
        echo "=== Service URLs ==="
        echo "Airflow:      ${SERVICE_URL}:8080"
        echo "Grafana:      ${SERVICE_URL}:3000"
        echo "Prometheus:   ${SERVICE_URL}:9090"
        echo "Alertmanager: ${SERVICE_URL}:9093"
        echo "Health Check: ${SERVICE_URL}:9091"
    else
        log_error "Health checks failed. Initiating automatic rollback..."
        rollback
        exit 1
    fi
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    echo "=============================================="
    echo " Investment Signals - AWS Deployment Script"
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
        *)
            log_error "Invalid action: ${ACTION}. Must be one of: deploy, rollback, status, health"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
