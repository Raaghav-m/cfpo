#!/bin/bash
#
# CFPO - Content Format Prompt Optimization
# ==========================================
#
# Setup and run script for CFPO optimization.
#
# Usage:
#   ./run.sh              # Run with HuggingFace (default)
#   ./run.sh --ollama     # Run with Ollama (local)
#   ./run.sh --hf         # Run with HuggingFace API
#   ./run.sh --web        # Run web interface (Gradio)
#   ./run.sh --setup      # Just setup, don't run
#   ./run.sh --help       # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Default settings
MODEL="huggingface"
MODEL_NAME=""
ROUNDS=3
BEAM_SIZE=2
RUN_WEB=false

# ============================================================
# Helper Functions
# ============================================================

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  CFPO - Content Format Prompt Optimization${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✖ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

show_help() {
    echo "CFPO - Content Format Prompt Optimization"
    echo ""
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --ollama          Use Ollama for local inference"
    echo "  --hf              Use HuggingFace API (default)"
    echo "  --model NAME      Specify model name"
    echo "  --rounds N        Number of optimization rounds (default: 3)"
    echo "  --setup           Only setup, don't run"
    echo "  --help            Show this help"
    echo "  --web             Run web interface (Gradio)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                      # Run with HuggingFace + phi model"
    echo "  ./run.sh --ollama --model mistral"
    echo "  ./run.sh --hf                 # Run with HuggingFace API"
    echo "  ./run.sh --web                # Run web interface"
    echo ""
    echo "Prerequisites:"
    echo "  For Ollama:"
    echo "    curl -fsSL https://ollama.com/install.sh | sh"
    echo "    ollama pull phi"
    echo "    ollama serve"
    echo ""
    echo "  For HuggingFace:"
    echo "    export HF_API_TOKEN=your_token_here"
    echo "    (Get token at: https://huggingface.co/settings/tokens)"
}

# ============================================================
# Setup Functions
# ============================================================

check_python() {
    print_step "Checking Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Found Python $PYTHON_VERSION"
}

setup_venv() {
    print_step "Setting up virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Created virtual environment"
    else
        print_success "Virtual environment exists"
    fi
    
    # Activate
    source "$VENV_DIR/bin/activate"
    
    # Install requirements
    print_step "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r "$SCRIPT_DIR/requirements.txt"
    print_success "Dependencies installed"
}

check_ollama() {
    print_step "Checking Ollama..."
    
    # Check if ollama is installed
    if ! command -v ollama &> /dev/null; then
        print_warning "Ollama not installed."
        echo ""
        echo "Install with:"
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
        echo ""
        read -p "Install Ollama now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            curl -fsSL https://ollama.com/install.sh | sh
        else
            print_error "Ollama required for local inference"
            exit 1
        fi
    fi
    
    # Check if ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_warning "Ollama not running. Starting it..."
        ollama serve &
        sleep 3
    fi
    
    MODEL_TO_CHECK=${MODEL_NAME:-phi}
    if ! ollama list | grep -q "$MODEL_TO_CHECK"; then
        print_step "Pulling model: $MODEL_TO_CHECK"
        ollama pull "$MODEL_TO_CHECK"
    fi
    
    print_success "Ollama ready with model: $MODEL_TO_CHECK"
}

check_huggingface() {
    print_step "Checking HuggingFace API..."
    
    if [ -z "$HF_API_TOKEN" ]; then
        print_error "HF_API_TOKEN not set!"
        echo ""
        echo "Get your free token at: https://huggingface.co/settings/tokens"
        echo "Then run: export HF_API_TOKEN=your_token_here"
        echo ""
        exit 1
    fi
    
    print_success "HuggingFace API token found"
}

# ============================================================
# Parse Arguments
# ============================================================

SETUP_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ollama)
            MODEL="ollama"
            shift
            ;;
        --hf|--huggingface)
            MODEL="huggingface"
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --beam-size)
            BEAM_SIZE="$2"
            shift 2
            ;;
        --setup)
            SETUP_ONLY=true
            shift
            ;;
        --web)
            RUN_WEB=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================
# Main
# ============================================================

print_header

# Setup
check_python
setup_venv

# Check model backend
if [ "$MODEL" = "ollama" ]; then
    check_ollama
else
    check_huggingface
fi

# Exit if setup only
if [ "$SETUP_ONLY" = true ]; then
    print_success "Setup complete!"
    exit 0
fi

# Build command
if [ "$RUN_WEB" = true ]; then
    CMD="$PYTHON_CMD $SCRIPT_DIR/app.py"
else
    CMD="$PYTHON_CMD $SCRIPT_DIR/main.py --model $MODEL --rounds $ROUNDS --beam-size $BEAM_SIZE"
    
    if [ -n "$MODEL_NAME" ]; then
        CMD="$CMD --model-name $MODEL_NAME"
    fi
fi

# Run
echo ""
if [ "$RUN_WEB" = true ]; then
    print_step "Starting CFPO Web Interface..."
else
    print_step "Starting CFPO Optimization..."
fi
echo -e "${YELLOW}Command: $CMD${NC}"
echo ""

$CMD

print_success "Done!"
