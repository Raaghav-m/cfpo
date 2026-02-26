#!/usr/bin/env python3
"""
CFPO Benchmark Runner - Plots Beam Size vs Accuracy and Rounds vs Accuracy

This script runs multiple CFPO configurations and generates comparison plots.
Uses Groq (FREE) or Ollama (FREE local) to avoid HuggingFace token limits.

Usage:
    # Using Groq (FREE cloud - recommended)
    export GROQ_API_KEY=your_key_here  # Get at https://console.groq.com/keys
    python benchmark.py --model groq
    
    # Using Ollama (FREE local)
    python benchmark.py --model ollama --model-name phi
    
    # Quick test with fewer configurations
    python benchmark.py --model groq --quick

Output:
    - results/benchmark_TIMESTAMP/
      - beam_size_vs_accuracy.png
      - rounds_vs_accuracy.png
      - benchmark_results.json
      - benchmark_log.txt
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import OllamaModel, HuggingFaceModel, GroqModel
from tasks import GSM8KTask
from prompts import Prompt
from mutators import MonteCarloMutator, FormatMutator
from optimizer import Optimizer


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, 'benchmark_log.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_initial_prompt() -> Prompt:
    """Create initial prompt for GSM8K."""
    return Prompt(
        task_instruction="Solve the following math word problem step by step.",
        task_detail="Read the problem carefully. Identify the key numbers and operations needed. Break down complex problems into smaller steps. Show all your calculations clearly.",
        output_format="End your response with 'The answer is: [NUMBER]' where [NUMBER] is the final numerical answer.",
        example_hinter="Here is an example:",
        examples="""Q: A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?
A: Cost of apples = 5 × $2 = $10. Change = $20 - $10 = $10. The answer is: 10""",
        cot_hinter="Let's solve this step by step:",
    )


def create_model(model_type: str, model_name: str, logger: logging.Logger):
    """Create the appropriate LLM model."""
    if model_type == 'groq':
        return GroqModel(model_name=model_name, logger=logger)
    elif model_type == 'ollama':
        return OllamaModel(model_name=model_name, logger=logger)
    elif model_type == 'huggingface':
        return HuggingFaceModel(model_name=model_name, logger=logger)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_single_config(
    model_type: str,
    model_name: str,
    beam_size: int,
    num_rounds: int,
    valid_size: int,
    output_dir: str,
    logger: logging.Logger,
) -> Tuple[float, float]:
    """
    Run a single CFPO configuration and return (initial_score, final_score).
    """
    config_name = f"beam{beam_size}_rounds{num_rounds}"
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: beam_size={beam_size}, rounds={num_rounds}")
    logger.info(f"{'='*60}")
    
    # Create model
    model = create_model(model_type, model_name, logger)
    
    # Create task with harder problems (uses built-in hard problems)
    task = GSM8KTask(
        train_size=5,
        valid_size=valid_size,
        test_size=5,
    )
    
    # Create mutators
    mutators = [
        MonteCarloMutator(llm=model, task=task, logger=logger),
        FormatMutator(llm=model, task=task, logger=logger),
    ]
    
    # Create optimizer
    optimizer = Optimizer(
        task=task,
        eval_llm=model,
        mutators=mutators,
        beam_size=beam_size,
        num_rounds=num_rounds,
        output_dir=config_dir,
        logger=logger,
    )
    
    # Run optimization
    init_prompt = get_initial_prompt()
    
    start_time = time.time()
    best_prompt, best_score, history = optimizer.run(init_prompt)
    elapsed = time.time() - start_time
    
    # Get initial score from history
    initial_score = history.history[0][2] if history.history else 0.0
    
    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Initial: {initial_score:.2%} → Final: {best_score:.2%}")
    
    return initial_score, best_score


def plot_results(results: Dict, output_dir: str, logger: logging.Logger):
    """Generate plots from benchmark results."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("matplotlib not installed. Install with: pip install matplotlib")
        logger.info("Skipping plot generation, but results are saved to JSON.")
        return
    
    # Set up style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # ===== Plot 1: Beam Size vs Accuracy =====
    beam_results = results.get('beam_size_experiment', {})
    if beam_results:
        beam_sizes = sorted([int(k) for k in beam_results.keys()])
        initial_scores = [beam_results[str(b)]['initial_score'] * 100 for b in beam_sizes]
        final_scores = [beam_results[str(b)]['final_score'] * 100 for b in beam_sizes]
        improvements = [f - i for f, i in zip(final_scores, initial_scores)]
        
        fig, ax = plt.subplots()
        
        x = np.arange(len(beam_sizes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, initial_scores, width, label='Initial Score', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_scores, width, label='Final Score', color='#2ecc71', alpha=0.8)
        
        # Add improvement annotations
        for i, (init, final, imp) in enumerate(zip(initial_scores, final_scores, improvements)):
            if imp > 0:
                ax.annotate(f'+{imp:.1f}%', 
                           xy=(i + width/2, final), 
                           ha='center', va='bottom',
                           fontsize=10, color='green', fontweight='bold')
        
        ax.set_xlabel('Beam Size', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('CFPO: Beam Size vs Accuracy', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(beam_sizes)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        beam_plot_path = os.path.join(output_dir, 'beam_size_vs_accuracy.png')
        plt.savefig(beam_plot_path, dpi=150)
        plt.close()
        logger.info(f"✓ Saved: {beam_plot_path}")
    
    # ===== Plot 2: Rounds vs Accuracy =====
    rounds_results = results.get('rounds_experiment', {})
    if rounds_results:
        rounds = sorted([int(k) for k in rounds_results.keys()])
        initial_scores = [rounds_results[str(r)]['initial_score'] * 100 for r in rounds]
        final_scores = [rounds_results[str(r)]['final_score'] * 100 for r in rounds]
        
        fig, ax = plt.subplots()
        
        ax.plot(rounds, initial_scores, 'o-', label='Initial Score', color='#3498db', 
                linewidth=2, markersize=10, alpha=0.8)
        ax.plot(rounds, final_scores, 's-', label='Final Score', color='#2ecc71', 
                linewidth=2, markersize=10, alpha=0.8)
        
        # Fill between to show improvement
        ax.fill_between(rounds, initial_scores, final_scores, alpha=0.2, color='#2ecc71')
        
        # Add annotations for final scores
        for r, score in zip(rounds, final_scores):
            ax.annotate(f'{score:.1f}%', 
                       xy=(r, score), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Number of Optimization Rounds', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('CFPO: Optimization Rounds vs Accuracy', fontsize=16, fontweight='bold')
        ax.set_xticks(rounds)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        rounds_plot_path = os.path.join(output_dir, 'rounds_vs_accuracy.png')
        plt.savefig(rounds_plot_path, dpi=150)
        plt.close()
        logger.info(f"✓ Saved: {rounds_plot_path}")
    
    # ===== Plot 3: Combined Summary =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Beam size subplot
    if beam_results:
        ax = axes[0]
        beam_sizes = sorted([int(k) for k in beam_results.keys()])
        improvements = [(beam_results[str(b)]['final_score'] - beam_results[str(b)]['initial_score']) * 100 
                       for b in beam_sizes]
        
        colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
        ax.bar(range(len(beam_sizes)), improvements, color=colors, alpha=0.8)
        ax.set_xticks(range(len(beam_sizes)))
        ax.set_xticklabels(beam_sizes)
        ax.set_xlabel('Beam Size')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Improvement by Beam Size')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    # Rounds subplot
    if rounds_results:
        ax = axes[1]
        rounds = sorted([int(k) for k in rounds_results.keys()])
        improvements = [(rounds_results[str(r)]['final_score'] - rounds_results[str(r)]['initial_score']) * 100 
                       for r in rounds]
        
        colors = ['#e74c3c' if imp < 0 else '#2ecc71' for imp in improvements]
        ax.bar(range(len(rounds)), improvements, color=colors, alpha=0.8)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_xlabel('Optimization Rounds')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Improvement by Rounds')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('CFPO Optimization Impact', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_plot_path = os.path.join(output_dir, 'optimization_summary.png')
    plt.savefig(summary_plot_path, dpi=150)
    plt.close()
    logger.info(f"✓ Saved: {summary_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CFPO Benchmark - Compare beam sizes and optimization rounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Groq (FREE - recommended)
  export GROQ_API_KEY=your_key_here
  python benchmark.py --model groq
  
  # Quick test mode
  python benchmark.py --model groq --quick
  
  # Using Ollama (FREE local)
  python benchmark.py --model ollama --model-name phi
  
Get FREE Groq API key at: https://console.groq.com/keys
        """
    )
    
    parser.add_argument('--model', type=str, default='groq',
                        choices=['groq', 'ollama', 'huggingface'],
                        help='LLM backend (default: groq - FREE)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name (default: auto-select best for backend)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer configurations for testing')
    parser.add_argument('--valid-size', type=int, default=5,
                        help='Validation set size (default: 5)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/benchmark_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # Set default model names
    if args.model_name is None:
        if args.model == 'groq':
            args.model_name = 'llama-3.1-8b-instant'
        elif args.model == 'ollama':
            args.model_name = 'phi'
        elif args.model == 'huggingface':
            args.model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'./results/benchmark_{timestamp}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("CFPO BENCHMARK RUNNER")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model} ({args.model_name})")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    # Define experiment configurations
    if args.quick:
        beam_sizes = [1, 2, 3]
        round_configs = [1, 2, 3]
        fixed_rounds = 2
        fixed_beam = 2
    else:
        beam_sizes = [1, 2, 3, 4, 5]
        round_configs = [1, 2, 3, 4, 5]
        fixed_rounds = 3
        fixed_beam = 2
    
    results = {
        'config': {
            'model': args.model,
            'model_name': args.model_name,
            'valid_size': args.valid_size,
            'mode': 'quick' if args.quick else 'full',
            'timestamp': datetime.now().isoformat(),
        },
        'beam_size_experiment': {},
        'rounds_experiment': {},
    }
    
    # ===== Experiment 1: Varying Beam Size =====
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1: BEAM SIZE VARIATION")
    logger.info(f"Fixed rounds = {fixed_rounds}, varying beam_size = {beam_sizes}")
    logger.info("=" * 60)
    
    for beam_size in beam_sizes:
        try:
            init_score, final_score = run_single_config(
                model_type=args.model,
                model_name=args.model_name,
                beam_size=beam_size,
                num_rounds=fixed_rounds,
                valid_size=args.valid_size,
                output_dir=args.output_dir,
                logger=logger,
            )
            
            results['beam_size_experiment'][str(beam_size)] = {
                'initial_score': init_score,
                'final_score': final_score,
                'improvement': final_score - init_score,
                'rounds': fixed_rounds,
            }
            
            # Small delay between runs
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error with beam_size={beam_size}: {e}")
            results['beam_size_experiment'][str(beam_size)] = {
                'error': str(e)
            }
    
    # ===== Experiment 2: Varying Rounds =====
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2: ROUNDS VARIATION")
    logger.info(f"Fixed beam_size = {fixed_beam}, varying rounds = {round_configs}")
    logger.info("=" * 60)
    
    for num_rounds in round_configs:
        try:
            init_score, final_score = run_single_config(
                model_type=args.model,
                model_name=args.model_name,
                beam_size=fixed_beam,
                num_rounds=num_rounds,
                valid_size=args.valid_size,
                output_dir=args.output_dir,
                logger=logger,
            )
            
            results['rounds_experiment'][str(num_rounds)] = {
                'initial_score': init_score,
                'final_score': final_score,
                'improvement': final_score - init_score,
                'beam_size': fixed_beam,
            }
            
            # Small delay between runs
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error with rounds={num_rounds}: {e}")
            results['rounds_experiment'][str(num_rounds)] = {
                'error': str(e)
            }
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to: {results_path}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot_results(results, args.output_dir, logger)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\nBeam Size Results:")
    for beam, data in sorted(results['beam_size_experiment'].items(), key=lambda x: int(x[0])):
        if 'error' not in data:
            logger.info(f"  Beam {beam}: {data['initial_score']:.1%} → {data['final_score']:.1%} "
                       f"(Δ {data['improvement']:+.1%})")
    
    logger.info("\nRounds Results:")
    for rounds, data in sorted(results['rounds_experiment'].items(), key=lambda x: int(x[0])):
        if 'error' not in data:
            logger.info(f"  Rounds {rounds}: {data['initial_score']:.1%} → {data['final_score']:.1%} "
                       f"(Δ {data['improvement']:+.1%})")
    
    logger.info(f"\n✓ All results saved to: {args.output_dir}")
    logger.info("  - benchmark_results.json")
    logger.info("  - beam_size_vs_accuracy.png")
    logger.info("  - rounds_vs_accuracy.png")
    logger.info("  - optimization_summary.png")


if __name__ == "__main__":
    main()
