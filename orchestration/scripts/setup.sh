#!/bin/bash
# =============================================================================
# Investment Signals Orchestration - Cross-Platform Setup Script
# =============================================================================
#
# This script sets up the local development environment for the Investment
# Signals Orchestration system on Mac, Linux, and Windows (Git Bash/WSL).
#
# Usage:
#   ./setup.sh              # Full setup
#   ./setup.sh --dev        # Development mode (lighter weight)
#   ./setup.sh --docker     # Start Docker only
#   ./setup.sh --db-init    # Initialize database only
#   ./setup.sh --check      # Check prerequisites only
#   ./setup.sh --clean      # Clean up and reset
#   ./setup.sh --status     # Show service status
#   ./setup.sh --logs       # View logs
#   ./setup.sh --stop       # Stop all services
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Default settings
DEV_MODE=false
SKIP_BUILD=false

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            OS="macos"
            ;;
        Linux*)
            if grep -q Microsoft /proc/version 2>/dev/null; then
                OS="wsl"
            else
                OS="linux"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS="windows"
            ;;
        *)
            OS="unknown"
            ;;
    esac
    echo "$OS"
}

# Get Docker Compose command (v1 or v2)
get_compose_cmd() {
    if command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    else
        echo ""
    fi
}

DETECTED_OS=$(detect_os)
COMPOSE_CMD=$(get_compose_cmd)

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Investment Signals Orchestration - Setup${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "${BLUE}Operating System: $DETECTED_OS${NC}"
echo -e "${BLUE}Project Root: $PROJECT_ROOT${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    local all_ok=true

    # Check Docker
    if command_exists docker; then
        docker_version=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
        echo -e "${GREEN}[OK]${NC} Docker: $docker_version"
    else
        echo -e "${RED}[ERROR]${NC} Docker is not installed"
        echo "       Install from: https://docs.docker.com/get-docker/"
        all_ok=false
    fi

    # Check Docker Compose
    if [ -n "$COMPOSE_CMD" ]; then
        compose_version=$($COMPOSE_CMD version 2>/dev/null | head -1 | awk '{print $NF}')
        echo -e "${GREEN}[OK]${NC} Docker Compose: $compose_version"
    else
        echo -e "${RED}[ERROR]${NC} Docker Compose is not installed"
        echo "       Install from: https://docs.docker.com/compose/install/"
        all_ok=false
    fi

    # Check if Docker daemon is running
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Docker daemon is running"
    else
        echo -e "${RED}[ERROR]${NC} Docker daemon is not running"
        echo "       Please start Docker Desktop or the Docker daemon"
        all_ok=false
    fi

    # Check available memory
    case "$DETECTED_OS" in
        macos)
            total_mem=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
            total_mem_gb=$((total_mem / 1024 / 1024 / 1024))
            ;;
        linux|wsl)
            total_mem=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
            total_mem_gb=$((total_mem / 1024 / 1024))
            ;;
        *)
            total_mem_gb=0
            ;;
    esac

    if [ "$total_mem_gb" -ge 4 ]; then
        echo -e "${GREEN}[OK]${NC} Memory: ${total_mem_gb}GB available"
    elif [ "$total_mem_gb" -gt 0 ]; then
        echo -e "${YELLOW}[WARN]${NC} Memory: ${total_mem_gb}GB (4GB+ recommended)"
    fi

    # Check required ports
    local ports_in_use=""
    for port in 8080 3000 9090 9091 5432; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            ports_in_use="$ports_in_use $port"
        fi
    done

    if [ -z "$ports_in_use" ]; then
        echo -e "${GREEN}[OK]${NC} Required ports are available"
    else
        echo -e "${YELLOW}[WARN]${NC} Ports in use:$ports_in_use"
    fi

    if [ "$all_ok" = true ]; then
        echo -e "\n${GREEN}All prerequisites met!${NC}"
        return 0
    else
        echo -e "\n${RED}Some prerequisites are missing. Please fix and try again.${NC}"
        return 1
    fi
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

# Clean up environment
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    cd "$DOCKER_DIR"

    echo -e "${YELLOW}This will stop all services and optionally remove data.${NC}"
    read -p "Continue? (y/N) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping services..."
        $COMPOSE_CMD down 2>/dev/null || true

        read -p "Remove volumes (all data will be lost)? (y/N) " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing volumes..."
            $COMPOSE_CMD down -v 2>/dev/null || true
            echo -e "${GREEN}Volumes removed${NC}"
        fi

        rm -f .env 2>/dev/null || true
        echo -e "${GREEN}Cleanup complete${NC}"
    else
        echo "Cleanup cancelled"
    fi
}

# Start in development mode
start_dev() {
    echo -e "\n${YELLOW}Starting in development mode...${NC}"
    cd "$DOCKER_DIR"

    # Use development compose file
    if [ -f "docker-compose.dev.yml" ]; then
        $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml up -d
    else
        $COMPOSE_CMD up -d
    fi

    echo -e "${GREEN}Development environment started${NC}"
}

# Main function
main() {
    case "${1:-full}" in
        --dev)
            check_prerequisites || exit 1
            create_directories
            setup_env
            set_airflow_uid
            DEV_MODE=true
            start_dev
            wait_for_airflow
            print_status
            ;;
        --docker)
            check_prerequisites || exit 1
            start_docker
            wait_for_airflow
            ;;
        --db-init)
            init_database
            ;;
        --check)
            check_prerequisites
            ;;
        --clean)
            cleanup
            ;;
        --status)
            cd "$DOCKER_DIR"
            $COMPOSE_CMD ps
            ;;
        --logs)
            cd "$DOCKER_DIR"
            $COMPOSE_CMD logs -f ${2:-}
            ;;
        --stop)
            cd "$DOCKER_DIR"
            $COMPOSE_CMD down
            echo -e "${GREEN}Services stopped${NC}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  (no option)   Full setup"
            echo "  --dev         Development mode (lighter weight)"
            echo "  --docker      Start Docker only"
            echo "  --db-init     Initialize database only"
            echo "  --check       Check prerequisites only"
            echo "  --clean       Clean up and reset"
            echo "  --status      Show service status"
            echo "  --logs        View logs"
            echo "  --stop        Stop all services"
            echo "  --help, -h    Show this help"
            ;;
        full|*)
            check_prerequisites || exit 1
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
