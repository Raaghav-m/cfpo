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

from models import OllamaModel, HuggingFaceModel, GroqModel
from tasks import GSM8KTask, MultipleChoiceTask, BBHTask, ARCTask, MMLUTask
from prompts import Prompt, PromptHistory
from mutators import CaseDiagnosisMutator, MonteCarloMutator, FormatMutator, UCTMutator
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


def get_initial_prompt(task_name: str = 'GSM8K') -> Prompt:
    """Create the initial prompt for the specified task."""
    
    if task_name.upper() in ['GSM8K', 'MATH']:
        return Prompt(
            task_instruction="Solve the following math word problem step by step.",
            task_detail="Read the problem carefully. Identify the key numbers and operations needed. Break down complex problems into smaller steps. Show all your calculations clearly.",
            output_format="End your response with 'The answer is: [NUMBER]' where [NUMBER] is the final numerical answer.",
            example_hinter="Here is an example:",
            examples="""Q: A movie theater sold 350 tickets for a show. Adult tickets cost $12 and child tickets cost $7. If the total revenue was $3,400, how many adult tickets were sold?
A: Let a = number of adult tickets and c = number of child tickets.
We have two equations:
1) a + c = 350 (total tickets)
2) 12a + 7c = 3400 (total revenue)

From equation 1: c = 350 - a
Substitute into equation 2: 12a + 7(350 - a) = 3400
12a + 2450 - 7a = 3400
5a = 3400 - 2450
5a = 950
a = 190

The answer is: 190""",
            cot_hinter="Let's solve this step by step:",
        )
    
    elif task_name.upper() in ['BBH', 'ARC', 'MMLU', 'MULTIPLECHOICE']:
        return Prompt(
            task_instruction="Answer the following multiple choice question by selecting the correct option.",
            task_detail="Read the question and all options carefully. Consider each option and eliminate incorrect ones. Choose the best answer based on logic and knowledge.",
            output_format="End your response with 'Answer: [LETTER]' where [LETTER] is A, B, C, or D.",
            example_hinter="Here is an example:",
            examples="""Q: What is the capital of France?
A) London  B) Paris  C) Berlin  D) Madrid

Let's analyze each option:
- A) London is the capital of the UK, not France.
- B) Paris is indeed the capital of France.
- C) Berlin is the capital of Germany.
- D) Madrid is the capital of Spain.

Answer: B""",
            cot_hinter="Let's think through this step by step:",
        )
    
    else:
        # Default generic prompt
        return Prompt(
            task_instruction="Complete the following task carefully.",
            task_detail="Read the instructions carefully and provide a complete response.",
            output_format="Provide your final answer clearly at the end.",
            example_hinter="Here is an example:",
            examples="[Examples would go here]",
            cot_hinter="Let's work through this:",
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
    elif model_type == 'groq':
        logger.info(f"Using Groq (FREE) with model: {model_name}")
        return GroqModel(model_name=model_name, logger=logger, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="CFPO - Content Format Prompt Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Groq (FREE - recommended for testing)
  export GROQ_API_KEY=your_key  # Get at https://console.groq.com/keys
  python main.py --model groq
  
  # Using Ollama (free, local)
  python main.py --model ollama --model-name phi
  
  # Using HuggingFace Inference Providers (cloud)
  export HF_API_TOKEN=your_token
  python main.py --model huggingface
  
  # Using HuggingFace with specific model
  python main.py --model huggingface --model-name Qwen/Qwen2.5-7B-Instruct
  
  # Custom optimization settings
  python main.py --model groq --rounds 5 --beam-size 4 --valid-size 10
  
  # Run benchmark with graphs
  python benchmark.py --model groq --quick
  
  # Run web interface (HuggingFace Spaces compatible)
  python app.py

Based on: "Beyond Prompt Content: Enhancing LLM Performance via Content-Format
Integrated Prompt Optimization" (https://arxiv.org/abs/2502.04295)
        """
    )
    
    # Model settings
    parser.add_argument('--model', type=str, default='huggingface',
                        choices=['ollama', 'huggingface', 'groq'],
                        help='LLM backend to use (default: huggingface, groq is FREE)')
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
                        choices=['GSM8K', 'BBH', 'ARC', 'MMLU', 'MultipleChoice'],
                        help='Task to optimize for (default: GSM8K)')
    parser.add_argument('--mmlu-subject', type=str, default='abstract_algebra',
                        help='Subject for MMLU task (default: abstract_algebra)')
    parser.add_argument('--train-size', type=int, default=10,
                        help='Training examples for diagnosis (default: 10)')
    parser.add_argument('--valid-size', type=int, default=5,
                        help='Validation examples (default: 5)')
    parser.add_argument('--test-size', type=int, default=5,
                        help='Test examples (default: 5)')
    parser.add_argument('--minibatch-size', type=int, default=5,
                        help='Minibatch size for diagnosis (default: 5)')
    
    # UCT settings
    parser.add_argument('--use-uct', action='store_true',
                        help='Use UCT mutator for format selection instead of random')
    parser.add_argument('--uct-exploration', type=float, default=1.414,
                        help='UCT exploration constant (default: 1.414, sqrt(2))')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./results/TIMESTAMP)')
    parser.add_argument('--output-marker', type=str, default='',
                        help='Marker to add to output directory name')
    
    args = parser.parse_args()
    
    # Set default model names based on backend
    if args.model_name is None:
        if args.model == 'ollama':
            args.model_name = 'phi'
        elif args.model == 'groq':
            args.model_name = 'llama-3.1-8b-instant'
        else:  # huggingface
            args.model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    
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
        task_name = args.task.upper()
        if task_name == 'GSM8K':
            task = GSM8KTask(
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
            )
        elif task_name == 'BBH':
            task = BBHTask(
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
            )
        elif task_name == 'ARC':
            task = ARCTask(
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
            )
        elif task_name == 'MMLU':
            task = MMLUTask(
                train_size=args.train_size,
                valid_size=args.valid_size,
                test_size=args.test_size,
                subject=args.mmlu_subject,
            )
        elif task_name == 'MULTIPLECHOICE':
            task = MultipleChoiceTask(
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
        ]
        
        # Use UCT mutator or standard format mutator
        if args.use_uct:
            logger.info("Using UCT algorithm for format selection")
            mutators.append(UCTMutator(
                llm=llm, 
                task=task, 
                logger=logger,
                exploration_constant=args.uct_exploration,
            ))
        else:
            mutators.append(FormatMutator(llm=llm, task=task, logger=logger))
        
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
        init_prompt = get_initial_prompt(args.task)
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
