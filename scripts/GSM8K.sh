#!/bin/bash
# Script to run CFPO on GSM8K math reasoning task

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default settings
MODEL="${1:-huggingface}"
MODEL_NAME="${2:-meta-llama/Llama-3.1-8B-Instruct}"

echo "=========================================="
echo "CFPO - GSM8K Math Reasoning Optimization"
echo "=========================================="
echo "Model: $MODEL ($MODEL_NAME)"
echo "=========================================="

python main.py \
    --task GSM8K \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --rounds 3 \
    --beam-size 2 \
    --train-size 10 \
    --valid-size 5 \
    --test-size 5 \
    --num-feedbacks 1 \
    --num-random 2 \
    --num-format 2 \
    --use-uct \
    --uct-exploration 1.414

echo ""
echo "Done!"