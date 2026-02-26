#!/bin/bash
# Script to run CFPO on Multiple Choice tasks (BBH, ARC, MMLU)

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default settings
TASK="${1:-BBH}"
MODEL="${2:-huggingface}"
MODEL_NAME="${3:-meta-llama/Llama-3.1-8B-Instruct}"

echo "=========================================="
echo "CFPO - Multiple Choice Task Optimization"
echo "=========================================="
echo "Task: $TASK"
echo "Model: $MODEL ($MODEL_NAME)"
echo "=========================================="

python main.py \
    --task "$TASK" \
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