#!/usr/bin/env python3
"""
CFPO - Content Format Prompt Optimization
HuggingFace Spaces Application

This script can be run directly on HuggingFace Spaces or locally.
It provides a web interface using Gradio for prompt optimization.

Usage:
    # Local execution
    python app.py
    
    # Environment variables
    export HF_API_TOKEN=your_token_here
    python app.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, Tuple, Generator

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for Gradio (required for HuggingFace Spaces)
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Running in CLI mode.")
    print("Install with: pip install gradio")

from models import HuggingFaceModel
from tasks import GSM8KTask
from prompts import Prompt, PromptHistory
from mutators import MonteCarloMutator, FormatMutator
from optimizer import Optimizer, DetailedLogger


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('cfpo')


def create_initial_prompt(task_type: str = "gsm8k") -> Prompt:
    """Create initial prompt based on task type."""
    if task_type == "gsm8k":
        return Prompt(
            task_instruction="Solve the following math word problem step by step.",
            task_detail="Read the problem carefully. Identify the key numbers and operations needed. Show your work clearly.",
            output_format="End your response with 'The answer is: [NUMBER]' where [NUMBER] is the final numerical answer.",
            example_hinter="Here is an example:",
            examples="""Q: John has 5 apples. He buys 3 more. How many does he have?
A: John starts with 5 apples. He buys 3 more. Total = 5 + 3 = 8 apples.
The answer is: 8""",
            cot_hinter="Let's solve this step by step:",
        )
    else:
        # Generic task
        return Prompt(
            task_instruction="Complete the following task step by step.",
            task_detail="Analyze the problem carefully and provide a clear solution.",
            output_format="Provide a clear and concise answer.",
            example_hinter="Here is an example:",
            examples="",
            cot_hinter="Let's think step by step:",
        )


def run_optimization(
    model_name: str,
    num_rounds: int,
    beam_size: int,
    task_type: str,
    num_examples: int,
    api_token: str,
    progress=None
) -> Generator[str, None, Tuple[str, str]]:
    """
    Run CFPO optimization with progress updates.
    
    Yields status updates and returns (best_prompt, detailed_log).
    """
    output_lines = []
    
    def log_output(line: str):
        output_lines.append(line)
        return "\n".join(output_lines)
    
    try:
        yield log_output("=" * 60)
        yield log_output("CFPO OPTIMIZATION STARTED")
        yield log_output("=" * 60)
        yield log_output(f"Configuration:")
        yield log_output(f"   - Model: {model_name}")
        yield log_output(f"   - Rounds: {num_rounds}")
        yield log_output(f"   - Beam Size: {beam_size}")
        yield log_output(f"   - Task: {task_type}")
        yield log_output(f"   - Validation Examples: {num_examples}")
        yield log_output("=" * 60)
        
        # Initialize model
        yield log_output("\n[1/4] Initializing model...")
        
        token = api_token or os.environ.get('HF_API_TOKEN', '')
        if not token:
            yield log_output("ERROR: No HuggingFace API token provided!")
            yield log_output("Please set HF_API_TOKEN or provide token in the interface.")
            return ("", "\n".join(output_lines))
        
        model = HuggingFaceModel(
            model_name=model_name,
            api_token=token,
            mode="providers",
            max_tokens=1024,
            temperature=0.7,
            logger=logger
        )
        yield log_output(f"   ✓ Model initialized: {model_name}")
        
        # Initialize task
        yield log_output("\n[2/4] Loading task data...")
        task = GSM8KTask(
            valid_size=num_examples,
            test_size=num_examples,
        )
        yield log_output(f"   ✓ Task loaded: {task_type}")
        yield log_output(f"   ✓ Validation examples: {len(task.get_valid_data())}")
        
        # Initialize mutators
        yield log_output("\n[3/4] Initializing mutators...")
        mutators = [
            MonteCarloMutator(llm=model, task=task, logger=logger),
            FormatMutator(llm=model, task=task, logger=logger),
        ]
        yield log_output(f"   ✓ MonteCarloMutator ready")
        yield log_output(f"   ✓ FormatMutator ready")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/run_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer
        yield log_output("\n[4/4] Starting optimization...")
        optimizer = Optimizer(
            task=task,
            eval_llm=model,
            mutators=mutators,
            beam_size=beam_size,
            num_rounds=num_rounds,
            output_dir=output_dir,
            logger=logger,
        )
        
        # Create initial prompt
        init_prompt = create_initial_prompt(task_type)
        
        yield log_output("\n" + "=" * 60)
        yield log_output("INITIAL PROMPT")
        yield log_output("=" * 60)
        yield log_output(f"TASK_INSTRUCTION: {init_prompt.task_instruction}")
        yield log_output(f"TASK_DETAIL: {init_prompt.task_detail}")
        yield log_output(f"OUTPUT_FORMAT: {init_prompt.output_format}")
        yield log_output(f"EXAMPLES: {init_prompt.examples[:100]}...")
        yield log_output("=" * 60)
        
        # Run optimization
        best_prompt, best_score, history = optimizer.run(init_prompt)
        
        # Format results
        yield log_output("\n" + "=" * 60)
        yield log_output("OPTIMIZATION COMPLETE")
        yield log_output("=" * 60)
        yield log_output(f"Best score: {best_score:.2%}")
        yield log_output(f"Results saved to: {output_dir}")
        
        # Read detailed log
        detailed_log_path = os.path.join(output_dir, 'detailed_log.txt')
        detailed_log = ""
        if os.path.exists(detailed_log_path):
            with open(detailed_log_path, 'r') as f:
                detailed_log = f.read()
        
        # Return final results
        best_prompt_text = best_prompt.render() if best_prompt else "No prompt generated"
        return (best_prompt_text, detailed_log)
        
    except Exception as e:
        yield log_output(f"\nERROR: {str(e)}")
        import traceback
        yield log_output(traceback.format_exc())
        return ("", "\n".join(output_lines))


def create_gradio_interface():
    """Create the Gradio web interface."""
    
    with gr.Blocks(
        title="CFPO - Content Format Prompt Optimization",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🚀 CFPO - Content Format Prompt Optimization
        
        Automatically optimize your prompts using beam search and intelligent mutations.
        
        Based on the paper: **"Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization"**
        
        [📄 Paper](https://arxiv.org/abs/2502.04295) | [💻 GitHub](https://github.com/HenryLau7/CFPO)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                api_token = gr.Textbox(
                    label="HuggingFace API Token",
                    placeholder="hf_...",
                    type="password",
                    info="Get your free token at huggingface.co/settings/tokens"
                )
                
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "Qwen/Qwen2.5-7B-Instruct",
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                    ],
                    value="meta-llama/Llama-3.1-8B-Instruct",
                    info="Choose a model from HuggingFace"
                )
                
                task_type = gr.Dropdown(
                    label="Task Type",
                    choices=["gsm8k", "general"],
                    value="gsm8k",
                    info="GSM8K for math word problems"
                )
                
                num_rounds = gr.Slider(
                    label="Optimization Rounds",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    info="More rounds = better prompts (but slower)"
                )
                
                beam_size = gr.Slider(
                    label="Beam Size",
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    info="Number of prompts to keep each round"
                )
                
                num_examples = gr.Slider(
                    label="Validation Examples",
                    minimum=1,
                    maximum=20,
                    value=3,
                    step=1,
                    info="Number of examples for evaluation"
                )
                
                run_button = gr.Button("🚀 Start Optimization", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Progress")
                progress_output = gr.Textbox(
                    label="Optimization Log",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✨ Best Prompt")
                best_prompt_output = gr.Textbox(
                    label="Optimized Prompt",
                    lines=15,
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("### 📋 Detailed Log")
                detailed_log_output = gr.Textbox(
                    label="Detailed Optimization Log",
                    lines=15,
                    interactive=False
                )
        
        # Example configurations
        gr.Examples(
            examples=[
                ["meta-llama/Llama-3.1-8B-Instruct", "gsm8k", 3, 2, 3],
                ["Qwen/Qwen2.5-7B-Instruct", "gsm8k", 5, 4, 5],
            ],
            inputs=[model_name, task_type, num_rounds, beam_size, num_examples],
            label="Example Configurations"
        )
        
        # Connect button to function
        def run_wrapper(model, rounds, beam, task, examples, token):
            """Wrapper to handle generator output."""
            result = ("", "")
            for output in run_optimization(model, rounds, beam, task, examples, token):
                if isinstance(output, tuple):
                    result = output
                else:
                    yield output, "", ""
            yield result[1] if len(result) > 1 else "", result[0], result[1] if len(result) > 1 else ""
        
        run_button.click(
            fn=run_wrapper,
            inputs=[model_name, num_rounds, beam_size, task_type, num_examples, api_token],
            outputs=[progress_output, best_prompt_output, detailed_log_output]
        )
    
    return demo


def run_cli():
    """Run optimization in CLI mode."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CFPO - Content Format Prompt Optimization")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    
    # Run optimization
    for output in run_optimization(
        args.model,
        args.rounds,
        args.beam_size,
        "gsm8k",
        args.examples,
        args.token
    ):
        if isinstance(output, str):
            print(output.split('\n')[-1])  # Print latest line
        else:
            print("\n" + "=" * 60)
            print("BEST PROMPT:")
            print("=" * 60)
            print(output[0])


if __name__ == "__main__":
    if GRADIO_AVAILABLE:
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    else:
        run_cli()
