#!/usr/bin/env python3
"""
Generate accuracy graphs from CFPO results.

Usage:
    python generate_graphs.py results/run_20260226_234513_benchmark_test
"""

import os
import sys
import json
import glob

def find_latest_result_dir():
    """Find the most recent results directory."""
    result_dirs = glob.glob('./results/run_*')
    if not result_dirs:
        print("No result directories found!")
        return None
    return max(result_dirs, key=os.path.getmtime)


def load_results(result_dir):
    """Load results from checkpoints and final results."""
    results = {
        'rounds': [],
        'scores': [],
        'best_scores': [],
    }
    
    # Load checkpoints
    checkpoint_files = sorted(glob.glob(os.path.join(result_dir, 'checkpoint_round_*.json')))
    for cp_file in checkpoint_files:
        with open(cp_file) as f:
            cp = json.load(f)
            results['rounds'].append(cp['round'])
            results['best_scores'].append(cp['best_score'])
            
            # Get all candidate scores
            if 'all_candidates' in cp:
                for cand in cp['all_candidates']:
                    results['scores'].append({
                        'round': cp['round'],
                        'score': cand['score'],
                        'action': cand['action'],
                    })
    
    # Load final results
    final_file = os.path.join(result_dir, 'final_results.json')
    if os.path.exists(final_file):
        with open(final_file) as f:
            results['final'] = json.load(f)
    
    return results


def plot_results(results, result_dir):
    """Generate plots from results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    # ===== Plot 1: Accuracy over rounds =====
    if results['rounds'] and results['best_scores']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rounds = results['rounds']
        best_scores = [s * 100 for s in results['best_scores']]
        
        # Add round 0 (initial)
        if 'final' in results and 'complete_history' in results['final']:
            history = results['final']['complete_history']
            round_0_scores = [h['score'] * 100 for h in history if h['round'] == 0]
            if round_0_scores:
                rounds = [0] + rounds
                best_scores = [max(round_0_scores)] + best_scores
        
        ax.plot(rounds, best_scores, 'o-', linewidth=2, markersize=10, color='#2ecc71')
        ax.fill_between(rounds, 0, best_scores, alpha=0.2, color='#2ecc71')
        
        ax.set_xlabel('Optimization Round', fontsize=14)
        ax.set_ylabel('Best Accuracy (%)', fontsize=14)
        ax.set_title('CFPO: Accuracy Improvement Over Rounds', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # Annotate points
        for r, s in zip(rounds, best_scores):
            ax.annotate(f'{s:.1f}%', xy=(r, s), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(result_dir, 'accuracy_over_rounds.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    # ===== Plot 2: Score distribution per round =====
    if results['scores']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group scores by round
        round_scores = {}
        for item in results['scores']:
            r = item['round']
            if r not in round_scores:
                round_scores[r] = []
            round_scores[r].append(item['score'] * 100)
        
        # Box plot
        rounds = sorted(round_scores.keys())
        data = [round_scores[r] for r in rounds]
        
        bp = ax.boxplot(data, labels=[f'Round {r}' for r in rounds], patch_artist=True)
        
        # Color boxes
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('CFPO: Score Distribution Per Round', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(result_dir, 'score_distribution.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {plot_path}")
    
    # ===== Plot 3: All scores timeline =====
    if 'final' in results and 'complete_history' in results['final']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        history = results['final']['complete_history']
        x = list(range(len(history)))
        scores = [h['score'] * 100 for h in history]
        actions = [h['action'] for h in history]
        rounds = [h['round'] for h in history]
        
        # Color by action type
        colors = []
        for action in actions:
            if action == 'init':
                colors.append('#3498db')
            elif 'monte_carlo' in action:
                colors.append('#2ecc71')
            elif 'format' in action:
                colors.append('#e74c3c')
            elif 'case_diagnosis' in action:
                colors.append('#f39c12')
            else:
                colors.append('#9b59b6')
        
        ax.scatter(x, scores, c=colors, s=100, alpha=0.7)
        ax.plot(x, scores, '-', alpha=0.3, color='gray')
        
        # Add best score line
        best_so_far = []
        current_best = 0
        for s in scores:
            current_best = max(current_best, s)
            best_so_far.append(current_best)
        ax.plot(x, best_so_far, '--', color='green', linewidth=2, label='Best so far')
        
        ax.set_xlabel('Prompt Evaluation #', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_title('CFPO: All Prompt Evaluations', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(result_dir, 'all_evaluations.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {plot_path}")


def main():
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = find_latest_result_dir()
        if not result_dir:
            sys.exit(1)
    
    print(f"Loading results from: {result_dir}")
    results = load_results(result_dir)
    
    if not results['rounds']:
        print("No checkpoint files found. Run CFPO first.")
        sys.exit(1)
    
    print(f"Found {len(results['rounds'])} rounds of data")
    plot_results(results, result_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
