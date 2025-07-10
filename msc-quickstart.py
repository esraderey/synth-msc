#!/usr/bin/env python3
"""
MSC Framework v4.0 - Quick Start Script

This script provides an interactive setup wizard for the MSC Framework.
"""

import os
import sys
import subprocess
import platform
import shutil
import secrets
import json
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message, color=Colors.OKGREEN):
    """Print colored message"""
    print(f"{color}{message}{Colors.ENDC}")

def print_banner():
    """Print MSC Framework banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   MSC Framework v4.0                          ║
    ║          Meta-cognitive Collective Synthesis                  ║
    ║                  with Claude AI                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Colors.HEADER)

def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored("Error: Python 3.8+ is required", Colors.FAIL)
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print_colored(f"✓ Python {version.major}.{version.minor}.{version.micro} detected", Colors.OKGREEN)
    return True

def check_command_exists(command):
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

def create_virtual_environment():
    """Create and activate virtual environment"""
    if os.path.exists("venv"):
        print_colored("Virtual environment already exists", Colors.WARNING)
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            shutil.rmtree("venv")
        else:
            return True
    
    print_colored("Creating virtual environment...", Colors.OKBLUE)
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print_colored("✓ Virtual environment created", Colors.OKGREEN)
    
    # Get activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate.bat"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print_colored(f"\nTo activate the virtual environment, run:", Colors.OKCYAN)
    print(f"  {activate_cmd}")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_colored("\nInstalling dependencies...", Colors.OKBLUE)
    
    # Determine pip command
    if platform.system() == "Windows":
        pip_cmd = os.path.join("venv", "Scripts", "pip")
    else:
        pip_cmd = os.path.join("venv", "bin", "pip")
    
    # Upgrade pip first
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    # Install requirements
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
    
    print_colored("✓ Dependencies installed successfully", Colors.OKGREEN)

def setup_environment():
    """Set up environment variables"""
    print_colored("\nSetting up environment...", Colors.OKBLUE)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_colored(".env file already exists", Colors.WARNING)
        response = input("Do you want to update it? (y/N): ").lower()
        if response != 'y':
            return
    
    # Read example file
    if env_example.exists():
        with open(env_example, 'r') as f:
            env_content = f.read()
    else:
        # Create basic env content
        env_content = """# MSC Framework Environment Variables
CLAUDE_API_KEY=
MSC_SECRET_KEY=
JWT_SECRET_KEY=
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://localhost/msc_framework
"""
    
    # Generate secret keys
    msc_secret = secrets.token_hex(32)
    jwt_secret = secrets.token_hex(32)
    
    env_content = env_content.replace("your-secret-key-here", msc_secret)
    env_content = env_content.replace("your-jwt-secret-key-here", jwt_secret)
    
    # Get Claude API key
    print_colored("\nClaude API Configuration", Colors.OKCYAN)
    print("To use Claude AI features, you need an API key from Anthropic.")
    print("Get your key at: https://console.anthropic.com/account/keys")
    
    claude_key = input("\nEnter your Claude API key (or press Enter to skip): ").strip()
    if claude_key:
        env_content = env_content.replace("CLAUDE_API_KEY=", f"CLAUDE_API_KEY={claude_key}")
        print_colored("✓ Claude API key configured", Colors.OKGREEN)
    else:
        print_colored("⚠ Claude API key not set - some features will be disabled", Colors.WARNING)
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print_colored("✓ Environment file created", Colors.OKGREEN)

def setup_directories():
    """Create necessary directories"""
    print_colored("\nCreating project directories...", Colors.OKBLUE)
    
    directories = [
        "data",
        "data/checkpoints",
        "data/logs",
        "static",
        "monitoring/grafana/dashboards",
        "monitoring/grafana/datasources"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_colored("✓ Directories created", Colors.OKGREEN)

def check_optional_services():
    """Check and configure optional services"""
    print_colored("\nChecking optional services...", Colors.OKBLUE)
    
    services = {
        "redis-cli": {
            "name": "Redis",
            "install": {
                "Darwin": "brew install redis",
                "Linux": "sudo apt-get install redis-server",
                "Windows": "Download from https://github.com/microsoftarchive/redis/releases"
            }
        },
        "psql": {
            "name": "PostgreSQL",
            "install": {
                "Darwin": "brew install postgresql",
                "Linux": "sudo apt-get install postgresql",
                "Windows": "Download from https://www.postgresql.org/download/windows/"
            }
        },
        "docker": {
            "name": "Docker",
            "install": {
                "Darwin": "Download Docker Desktop from https://www.docker.com/products/docker-desktop",
                "Linux": "curl -fsSL https://get.docker.com | sh",
                "Windows": "Download Docker Desktop from https://www.docker.com/products/docker-desktop"
            }
        }
    }
    
    system = platform.system()
    
    for cmd, info in services.items():
        if check_command_exists(cmd):
            print_colored(f"✓ {info['name']} is installed", Colors.OKGREEN)
        else:
            print_colored(f"✗ {info['name']} not found", Colors.WARNING)
            install_cmd = info['install'].get(system, "Check official documentation")
            print(f"  To install: {install_cmd}")

def create_sample_config():
    """Create sample configuration file"""
    config_file = Path("config.yaml")
    
    if config_file.exists():
        print_colored("\nconfig.yaml already exists", Colors.WARNING)
        return
    
    print_colored("\nCreating sample configuration...", Colors.OKBLUE)
    
    config_content = """# MSC Framework Configuration
simulation:
  steps: 10000
  step_delay: 0.1
  checkpoint_interval: 1000

agents:
  claude_taec: 3

claude:
  model: "claude-3-sonnet-20240229"
  temperature: 0.7

api:
  enable: true
  host: "0.0.0.0"
  port: 5000

# See config.yaml.example for all options
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print_colored("✓ Sample configuration created", Colors.OKGREEN)

def setup_docker():
    """Set up Docker environment"""
    if not check_command_exists("docker"):
        print_colored("\nDocker not installed, skipping Docker setup", Colors.WARNING)
        return
    
    response = input("\nDo you want to set up Docker environment? (y/N): ").lower()
    if response != 'y':
        return
    
    print_colored("\nSetting up Docker environment...", Colors.OKBLUE)
    
    # Check if docker-compose.yml exists
    if not os.path.exists("docker-compose.yml"):
        print_colored("docker-compose.yml not found", Colors.FAIL)
        return
    
    # Build images
    print_colored("Building Docker images...", Colors.OKBLUE)
    subprocess.run(["docker-compose", "build"], check=True)
    
    print_colored("✓ Docker environment ready", Colors.OKGREEN)
    print_colored("\nTo start with Docker:", Colors.OKCYAN)
    print("  docker-compose up")

def print_next_steps():
    """Print next steps for the user"""
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("Setup Complete! 🎉", Colors.HEADER)
    print_colored("="*60 + "\n", Colors.HEADER)
    
    print_colored("Next steps:", Colors.OKCYAN)
    print("\n1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Run the framework:")
    print("   python msc_framework_enhanced.py")
    
    print("\n3. Access the web interface:")
    print("   http://localhost:5000")
    
    print("\n4. View API documentation:")
    print("   http://localhost:5000/api/docs")
    
    print_colored("\nFor more information:", Colors.OKCYAN)
    print("- Configuration: edit config.yaml")
    print("- Environment: edit .env")
    print("- Documentation: see README.md")
    
    print_colored("\nHappy synthesizing! 🚀", Colors.OKGREEN)

def main():
    """Main setup wizard"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists("msc_framework_enhanced.py"):
        print_colored("Error: msc_framework_enhanced.py not found", Colors.FAIL)
        print("Please run this script from the MSC Framework directory")
        sys.exit(1)
    
    try:
        # Run setup steps
        create_virtual_environment()
        install_dependencies()
        setup_environment()
        setup_directories()
        create_sample_config()
        check_optional_services()
        setup_docker()
        print_next_steps()
        
    except subprocess.CalledProcessError as e:
        print_colored(f"\nError during setup: {e}", Colors.FAIL)
        sys.exit(1)
    except KeyboardInterrupt:
        print_colored("\n\nSetup cancelled by user", Colors.WARNING)
        sys.exit(0)
    except Exception as e:
        print_colored(f"\nUnexpected error: {e}", Colors.FAIL)
        sys.exit(1)

if __name__ == "__main__":
    main()