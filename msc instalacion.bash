#!/bin/bash

# MSC Framework v4.0 - Installation Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    echo -e "${GREEN}[MSC Framework]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ASCII Art Banner
echo "
███╗   ███╗███████╗ ██████╗    ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
████╗ ████║██╔════╝██╔════╝    ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
██╔████╔██║███████╗██║         █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ 
██║╚██╔╝██║╚════██║██║         ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██╔═██╗ 
██║ ╚═╝ ██║███████║╚██████╗    ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
╚═╝     ╚═╝╚══════╝ ╚═════╝    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
                                                    v4.0 with Claude AI
"

print_message "Starting MSC Framework installation..."

# Check Python version
print_message "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
print_message "Python $PYTHON_VERSION found ✓"

# Create virtual environment
print_message "Creating virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    print_message "Virtual environment created ✓"
fi

# Activate virtual environment
print_message "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_message "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_message "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_message "Creating project directories..."
mkdir -p data/checkpoints data/logs static monitoring/grafana/dashboards monitoring/grafana/datasources

# Copy environment template
if [ ! -f ".env" ]; then
    print_message "Creating .env file..."
    cp .env.example .env
    print_warning "Please edit .env file and add your CLAUDE_API_KEY"
else
    print_warning ".env file already exists. Skipping..."
fi

# Check for optional services
print_message "Checking optional services..."

# Check Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        print_message "Redis is running ✓"
    else
        print_warning "Redis is installed but not running"
    fi
else
    print_warning "Redis not found. Install Redis for distributed caching (optional)"
fi

# Check PostgreSQL
if command -v psql &> /dev/null; then
    print_message "PostgreSQL is installed ✓"
else
    print_warning "PostgreSQL not found. Install PostgreSQL for persistence (optional)"
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_message "Docker is installed ✓"
    
    # Ask if user wants to use Docker
    read -p "Do you want to set up Docker containers? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "Building Docker containers..."
        docker-compose build
        print_message "Docker setup complete ✓"
    fi
else
    print_warning "Docker not found. Install Docker for containerized deployment (optional)"
fi

# Generate secret keys if not set
print_message "Checking secret keys..."
if grep -q "your-secret-key-here" .env; then
    print_message "Generating secret keys..."
    MSC_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    
    # Update .env file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/your-secret-key-here/$MSC_SECRET/g" .env
        sed -i '' "s/your-jwt-secret-key-here/$JWT_SECRET/g" .env
    else
        # Linux
        sed -i "s/your-secret-key-here/$MSC_SECRET/g" .env
        sed -i "s/your-jwt-secret-key-here/$JWT_SECRET/g" .env
    fi
    print_message "Secret keys generated ✓"
fi

# Create sample configuration if not exists
if [ ! -f "config.yaml" ]; then
    print_message "Creating sample configuration..."
    cat > config.yaml << 'EOF'
# MSC Framework Configuration
simulation:
  steps: 10000
  step_delay: 0.1

agents:
  claude_taec: 3

api:
  enable: true
  port: 5000
EOF
    print_message "Sample configuration created ✓"
fi

# Installation complete
echo
print_message "Installation complete! 🎉"
echo
echo "Next steps:"
echo "1. Edit .env file and add your CLAUDE_API_KEY"
echo "2. (Optional) Configure config.yaml for your needs"
echo "3. Run the framework:"
echo "   source venv/bin/activate"
echo "   python msc_framework_enhanced.py"
echo
echo "For more information, see README.md"
echo
print_message "Happy synthesizing! 