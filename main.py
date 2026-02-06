#!/usr/bin/env python3
"""
CFPO - Content Format Prompt Optimization

Main entry point for running prompt optimization.

Usage:
    # Using Ollama (free, local)
    python main.py --model ollama --model-name phi
    
    # Using HuggingFace (cloud API)
    export HF_API_TOKEN=your_token
    python main.py --model huggingface
    
    # Run on HuggingFace Spaces
    python app.py

Based on the paper: "Beyond Prompt Content: Enhancing LLM Performance via 
Content-Format Integrated Prompt Optimization" (https://arxiv.org/abs/2502.04295)
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OllamaModel, HuggingFaceModel
from tasks import GSM8KTask
from prompts import Prompt, PromptHistory
from mutators import CaseDiagnosisMutator, MonteCarloMutator, FormatMutator
from optimizer import Optimizer


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    logger = logging.getLogger('cfpo')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_initial_prompt() -> Prompt:
    """Create the initial prompt for GSM8K."""
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


def create_model(model_type: str, model_name: str, logger: logging.Logger, **kwargs):
    """Create the appropriate LLM model."""
    if model_type == 'ollama':
        logger.info(f"Using Ollama with model: {model_name}")
        return OllamaModel(model_name=model_name, logger=logger, **kwargs)
    elif model_type == 'huggingface':
        logger.info(f"Using HuggingFace with model: {model_name}")
        # Support different HuggingFace modes
        mode = kwargs.pop('mode', 'providers')
        return HuggingFaceModel(
            model_name=model_name, 
            logger=logger, 
            mode=mode,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="CFPO - Content Format Prompt Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Ollama (free, local)
  python main.py --model ollama --model-name phi
  
  # Using HuggingFace Inference Providers (cloud)
  export HF_API_TOKEN=your_token
  python main.py --model huggingface
  
  # Using HuggingFace with specific model
  python main.py --model huggingface --model-name Qwen/Qwen2.5-7B-Instruct
  
  # Custom optimization settings
  python main.py --model ollama --rounds 5 --beam-size 4 --valid-size 10
  
  # Run web interface (HuggingFace Spaces compatible)
  python app.py

Based on: "Beyond Prompt Content: Enhancing LLM Performance via Content-Format
Integrated Prompt Optimization" (https://arxiv.org/abs/2502.04295)
        """
    )
    
    # Model settings
    parser.add_argument('--model', type=str, default='huggingface',
                        choices=['ollama', 'huggingface'],
                        help='LLM backend to use (default: huggingface)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name (default: phi for ollama, Llama-3.1-8B for hf)')
    parser.add_argument('--hf-mode', type=str, default='providers',
                        choices=['providers', 'api', 'local'],
                        help='HuggingFace inference mode (default: providers)')
    
    # Optimization settings (matching original CFPO)
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of optimization rounds (default: 3)')
    parser.add_argument('--beam-size', type=int, default=2,
                        help='Beam size for search (default: 2)')
    parser.add_argument('--init-temperature', type=float, default=1.0,
                        help='Initial temperature for mutations (default: 1.0)')
    
    # Mutator settings
    parser.add_argument('--num-feedbacks', type=int, default=1,
                        help='Number of prompts from case diagnosis (default: 1)')
    parser.add_argument('--num-random', type=int, default=2,
                        help='Number of prompts from Monte Carlo (default: 2)')
    parser.add_argument('--num-format', type=int, default=2,
                        help='Number of prompts from format mutation (default: 2)')
    
    # Data settings
    parser.add_argument('--task', type=str, default='GSM8K',
                        choices=['GSM8K'],
                        help='Task to optimize for (default: GSM8K)')
    parser.add_argument('--train-size', type=int, default=10,
                        help='Training examples for diagnosis (default: 10)')
    parser.add_argument('--valid-size', type=int, default=5,
                        help='Validation examples (default: 5)')
    parser.add_argument('--test-size', type=int, default=5,
                        help='Test examples (default: 5)')
    parser.add_argument('--minibatch-size', type=int, default=5,
                        help='Minibatch size for diagnosis (default: 5)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./results/TIMESTAMP)')
    parser.add_argument('--output-marker', type=str, default='',
                        help='Marker to add to output directory name')
    
    args = parser.parse_args()
    
    # Set default model names
    if args.model_name is None:
        args.model_name = 'phi' if args.model == 'ollama' else 'meta-llama/Llama-3.1-8B-Instruct'
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        marker = f"_{args.output_marker}" if args.output_marker else ""
        args.output_dir = f'./results/run_{timestamp}{marker}'
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("CFPO - Content Format Prompt Optimization")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model} ({args.model_name})")
    logger.info(f"Rounds: {args.rounds}, Beam size: {args.beam_size}")
    logger.info(f"Mutator settings: feedbacks={args.num_feedbacks}, random={args.num_random}, format={args.num_format}")
    logger.info(f"Data: train={args.train_size}, valid={args.valid_size}, test={args.test_size}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)
    
    try:
        # Create model
        model_kwargs = {}
        if args.model == 'huggingface':
            model_kwargs['mode'] = args.hf_mode
            
        llm = create_model(args.model, args.model_name, logger, **model_kwargs)
        
        # Create task
        if args.task == 'GSM8K':
            task = GSM8KTask(
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
            )
        else:
            raise ValueError(f"Unknown task: {args.task}")
            
        logger.info(f"Task: {task}")
        
        # Create mutators
        mutators = [
            MonteCarloMutator(llm=llm, task=task, logger=logger),
            FormatMutator(llm=llm, task=task, logger=logger),
        ]
        
        # Add case diagnosis if we have enough training data
        if args.num_feedbacks > 0 and args.train_size >= 5:
            mutators.insert(0, CaseDiagnosisMutator(
                llm=llm, 
                eval_llm=llm, 
                task=task, 
                logger=logger,
                num_errors_to_analyze=2,
                num_correct_to_analyze=1,
            ))
        
        logger.info(f"Mutators: {[m.__class__.__name__ for m in mutators]}")
        
        # Create optimizer
        optimizer = Optimizer(
            task=task,
            eval_llm=llm,
            mutators=mutators,
            beam_size=args.beam_size,
            num_rounds=args.rounds,
            output_dir=args.output_dir,
            logger=logger,
        )
        
        # Get initial prompt
        init_prompt = get_initial_prompt()
        logger.info(f"\nInitial Prompt:\n{init_prompt.render()[:500]}...\n")
        
        # Run optimization
        best_prompt, best_score, history = optimizer.run(init_prompt)
        
        # Print results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Best Score: {best_score:.2%}")
        print(f"\nBest Prompt:\n{best_prompt.render()[:1000]}")
        print("\n" + history.summary())
        print(f"\nResults saved to: {args.output_dir}")
        
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        print(f"\n❌ Error: {e}")
        if args.model == 'ollama':
            print("\nMake sure Ollama is running: ollama serve")
        else:
            print("\nCheck your HuggingFace API token: export HF_API_TOKEN=your_token")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n⚠️  Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
