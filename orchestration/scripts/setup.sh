#!/bin/bash
#
# Investment Signals Orchestration - Setup Script
#
# This script sets up the local development environment:
# 1. Creates necessary directories
# 2. Copies environment template
# 3. Initializes the database
# 4. Starts Docker services
#
# Usage:
#   ./setup.sh              # Full setup
#   ./setup.sh --docker     # Start Docker only
#   ./setup.sh --db-init    # Initialize database only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DOCKER_DIR="$PROJECT_ROOT/docker"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Investment Signals Orchestration - Setup${NC}"
echo -e "${GREEN}==================================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"

    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command_exists docker-compose; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker daemon is not running${NC}"
        echo "Please start Docker and try again"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
}

# Create directories
create_directories() {
    echo -e "\n${YELLOW}Creating directories...${NC}"

    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/plugins"
    mkdir -p "$PROJECT_ROOT/data"

    echo -e "${GREEN}Directories created${NC}"
}

# Setup environment file
setup_env() {
    echo -e "\n${YELLOW}Setting up environment...${NC}"

    if [ ! -f "$PROJECT_ROOT/config/.env" ]; then
        cp "$PROJECT_ROOT/config/env.example" "$PROJECT_ROOT/config/.env"
        echo -e "${YELLOW}Created .env file from template${NC}"
        echo -e "${YELLOW}Please edit $PROJECT_ROOT/config/.env with your settings${NC}"
    else
        echo -e "${GREEN}.env file already exists${NC}"
    fi

    # Create symlink for Docker Compose
    if [ ! -f "$DOCKER_DIR/.env" ]; then
        ln -sf "$PROJECT_ROOT/config/.env" "$DOCKER_DIR/.env"
        echo -e "${GREEN}Created .env symlink for Docker${NC}"
    fi
}

# Set Airflow UID
set_airflow_uid() {
    echo -e "\n${YELLOW}Setting Airflow UID...${NC}"

    # Get current user UID
    CURRENT_UID=$(id -u)

    # Update .env file
    if grep -q "AIRFLOW_UID=" "$PROJECT_ROOT/config/.env"; then
        sed -i.bak "s/AIRFLOW_UID=.*/AIRFLOW_UID=$CURRENT_UID/" "$PROJECT_ROOT/config/.env"
    else
        echo "AIRFLOW_UID=$CURRENT_UID" >> "$PROJECT_ROOT/config/.env"
    fi

    echo -e "${GREEN}Set AIRFLOW_UID to $CURRENT_UID${NC}"
}

# Start Docker services
start_docker() {
    echo -e "\n${YELLOW}Starting Docker services...${NC}"

    cd "$DOCKER_DIR"

    # Pull images
    echo "Pulling Docker images..."
    docker-compose pull

    # Start services
    echo "Starting services..."
    docker-compose up -d

    echo -e "${GREEN}Docker services started${NC}"
}

# Initialize database
init_database() {
    echo -e "\n${YELLOW}Initializing database...${NC}"

    cd "$DOCKER_DIR"

    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL..."
    sleep 10

    # Check if database is ready
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U airflow >/dev/null 2>&1; then
            echo "PostgreSQL is ready"
            break
        fi
        echo "Waiting for PostgreSQL... ($i/30)"
        sleep 2
    done

    echo -e "${GREEN}Database initialized${NC}"
}

# Wait for Airflow
wait_for_airflow() {
    echo -e "\n${YELLOW}Waiting for Airflow...${NC}"

    for i in {1..60}; do
        if curl -s http://localhost:8080/health | grep -q "healthy"; then
            echo -e "${GREEN}Airflow is ready!${NC}"
            return 0
        fi
        echo "Waiting for Airflow to start... ($i/60)"
        sleep 5
    done

    echo -e "${YELLOW}Airflow may still be starting. Check logs with: docker-compose logs -f${NC}"
}

# Print status
print_status() {
    echo -e "\n${GREEN}==================================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}==================================================${NC}"

    echo -e "\n${YELLOW}Service URLs:${NC}"
    echo "  Airflow:     http://localhost:8080 (admin/admin)"
    echo "  Grafana:     http://localhost:3000 (admin/admin)"
    echo "  Prometheus:  http://localhost:9090"
    echo "  Health:      http://localhost:9091/health"

    echo -e "\n${YELLOW}Useful Commands:${NC}"
    echo "  View logs:           docker-compose logs -f"
    echo "  Stop services:       docker-compose down"
    echo "  Restart services:    docker-compose restart"
    echo "  View Airflow logs:   docker-compose logs -f airflow-scheduler"

    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo "  1. Edit config/.env with your settings"
    echo "  2. Enable DAGs in Airflow UI"
    echo "  3. Configure alerting (Slack, Email, SMS)"
}

# Main function
main() {
    case "${1:-full}" in
        --docker)
            check_prerequisites
            start_docker
            wait_for_airflow
            ;;
        --db-init)
            init_database
            ;;
        --status)
            cd "$DOCKER_DIR"
            docker-compose ps
            ;;
        --logs)
            cd "$DOCKER_DIR"
            docker-compose logs -f ${2:-}
            ;;
        --stop)
            cd "$DOCKER_DIR"
            docker-compose down
            echo -e "${GREEN}Services stopped${NC}"
            ;;
        full|*)
            check_prerequisites
            create_directories
            setup_env
            set_airflow_uid
            start_docker
            init_database
            wait_for_airflow
            print_status
            ;;
    esac
}

# Run main function
main "$@"
