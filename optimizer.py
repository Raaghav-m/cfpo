"""
Optimizer - Core Optimization Loop

Implements beam search optimization over prompt space.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from copy import deepcopy

from prompts import Prompt, PromptHistory
from models import LLMModel
from tasks import Task
from mutators import Mutator


class DetailedLogger:
    """Helper class to log to both console and a detailed log file."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.log_lines = []
        self.detailed_log_path = None
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.detailed_log_path = os.path.join(output_dir, 'detailed_log.txt')
            # Clear the file at start
            with open(self.detailed_log_path, 'w') as f:
                f.write("")
    
    def log(self, message: str, also_print: bool = True):
        """Log a message to file and optionally print to console."""
        self.log_lines.append(message)
        if also_print:
            print(message)
        
        # Write immediately to file
        if self.detailed_log_path:
            with open(self.detailed_log_path, 'a') as f:
                f.write(message + '\n')
    
    def save(self):
        """Save all logged lines to file."""
        if self.detailed_log_path:
            with open(self.detailed_log_path, 'w') as f:
                f.write('\n'.join(self.log_lines))


# Global detailed logger instance for use by helper functions
_global_dlog: Optional[DetailedLogger] = None


def set_global_logger(dlog: DetailedLogger):
    """Set the global detailed logger."""
    global _global_dlog
    _global_dlog = dlog


def get_global_logger() -> Optional[DetailedLogger]:
    """Get the global detailed logger."""
    return _global_dlog


class Optimizer:
    """
    Prompt Optimizer using beam search.
    
    Algorithm:
    1. Start with initial prompt
    2. For each round:
       a. Apply mutators to generate new prompts
       b. Evaluate all prompts on validation set
       c. Keep top-K prompts (beam search)
    3. Return best prompt found
    """
    
    def __init__(
        self,
        task: Task,
        eval_llm: LLMModel,
        mutators: List[Mutator],
        beam_size: int = 4,
        num_rounds: int = 5,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            task: Task to optimize for
            eval_llm: LLM for evaluating prompts
            mutators: List of mutation strategies
            beam_size: Number of prompts to keep per round
            num_rounds: Total optimization rounds
            output_dir: Directory to save results
            logger: Logger instance
        """
        self.task = task
        self.eval_llm = eval_llm
        self.mutators = mutators
        self.beam_size = beam_size
        self.num_rounds = num_rounds
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detailed logger for complete output
        self.detailed_log = DetailedLogger(output_dir)
        set_global_logger(self.detailed_log)
    
    def run(self, init_prompt: Prompt) -> Tuple[Prompt, float, PromptHistory]:
        """
        Run the optimization loop.

        Args:
            init_prompt: Starting prompt

        Returns:
            (best_prompt, best_score, history)
        """
        self.logger.info("=" * 60)
        self.logger.info("CFPO OPTIMIZATION STARTED")
        self.logger.info("=" * 60)
        
        # Use detailed logger for complete output
        dlog = self.detailed_log
        
        # Print configuration summary
        dlog.log("\n" + "=" * 60)
        dlog.log("CFPO OPTIMIZATION STARTED")
        dlog.log("=" * 60)
        dlog.log(f"Configuration:")
        dlog.log(f"   - Rounds: {self.num_rounds}")
        dlog.log(f"   - Beam Size: {self.beam_size}")
        dlog.log(f"   - Mutators: {[m.__class__.__name__ for m in self.mutators]}")
        dlog.log(f"   - Validation Set: {len(self.task.get_valid_data())} examples")
        dlog.log("=" * 60)
        
        # Log initial prompt
        dlog.log("\n" + "=" * 60)
        dlog.log("INITIAL PROMPT")
        dlog.log("=" * 60)
        self._log_full_prompt(init_prompt, dlog)
        
        # Initialize
        history = PromptHistory(init_prompt)
        beam = [init_prompt]  # Current best prompts
        
        # Evaluate initial prompt
        dlog.log("\nEvaluating initial prompt...")
        init_score = self._evaluate_prompt(init_prompt)
        init_prompt.score = init_score
        history.add(init_prompt, init_score, 0)
        
        self.logger.info(f"Initial prompt score: {init_score:.2%}")
        dlog.log(f"Initial prompt score: {init_score:.2%}")
        
        # Optimization loop
        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ROUND {round_num}/{self.num_rounds}")
            self.logger.info(f"{'='*60}")
            
            # Console output for round
            dlog.log(f"\n{'='*60}")
            dlog.log(f"ROUND {round_num}/{self.num_rounds}")
            dlog.log(f"{'='*60}")
            dlog.log(f"   Current beam size: {len(beam)} prompts")
            dlog.log(f"   Best score so far: {history.best_score:.2%}")
            
            # Generate new prompts using mutators
            candidates = []
            candidate_details = []  # Track details for each candidate
            
            dlog.log(f"\nMUTATION PHASE:")
            dlog.log(f"   {'─'*50}")
            
            for beam_idx, prompt in enumerate(beam):
                dlog.log(f"\n   Mutating beam prompt {beam_idx + 1}/{len(beam)}:")
                for mutator in self.mutators:
                    try:
                        # Pass the detailed logger to mutators
                        new_prompts = mutator.mutate(
                            prompt,
                            num_mutations=2,
                            temperature=0.7,
                            round=round_num
                        )
                        candidates.extend(new_prompts)
                        
                        # Track candidate details
                        for np in new_prompts:
                            candidate_details.append({
                                'mutator': mutator.__class__.__name__,
                                'action': np.action,
                                'parent_beam': beam_idx + 1,
                                'prompt': np
                            })
                        
                        self.logger.info(f"{mutator.__class__.__name__}: generated {len(new_prompts)} prompts")
                        dlog.log(f"      - {mutator.__class__.__name__}: generated {len(new_prompts)} variants")
                        
                        # Show what was modified with full content for ALL keywords
                        for idx, np in enumerate(new_prompts):
                            modified = self._get_modified_components(prompt, np)
                            dlog.log(f"\n        +{'─'*58}+")
                            dlog.log(f"        | VARIANT {idx+1} | Action: {np.action or 'mutation'}")
                            if modified:
                                dlog.log(f"        | Modified: [{', '.join(modified)}]")
                            dlog.log(f"        +{'─'*58}+")
                            
                            # Show TASK_INSTRUCTION
                            dlog.log(f"        | TASK_INSTRUCTION:")
                            for line in np.task_instruction.split('\n'):
                                dlog.log(f"        |    {line}")
                            
                            # Show TASK_DETAIL
                            dlog.log(f"        | TASK_DETAIL:")
                            for line in np.task_detail.split('\n'):
                                dlog.log(f"        |    {line}")
                            
                            # Show OUTPUT_FORMAT
                            dlog.log(f"        | OUTPUT_FORMAT:")
                            for line in np.output_format.split('\n'):
                                dlog.log(f"        |    {line}")
                            
                            # Show EXAMPLES
                            dlog.log(f"        | EXAMPLES:")
                            for line in np.examples.split('\n'):
                                dlog.log(f"        |    {line}")
                            
                            # Show COT_HINTER
                            dlog.log(f"        | COT_HINTER:")
                            for line in np.cot_hinter.split('\n'):
                                dlog.log(f"        |    {line}")
                            
                            dlog.log(f"        +{'─'*58}+")
                            
                    except Exception as e:
                        self.logger.error(f"Mutator error: {e}")
                        dlog.log(f"      [ERROR] {mutator.__class__.__name__}: Error - {e}")
            
            # Add current beam to candidates
            candidates.extend(beam)
            for p in beam:
                candidate_details.append({
                    'mutator': 'beam_carry',
                    'action': 'previous_best',
                    'parent_beam': 0,
                    'prompt': p
                })
            
            # Remove duplicates and track deleted prompts
            seen = set()
            unique_candidates = []
            unique_details = []
            deleted_duplicates = []  # Track deleted duplicate prompts
            deleted_duplicate_details = []
            for i, p in enumerate(candidates):
                p_str = str(p)
                if p_str not in seen:
                    seen.add(p_str)
                    unique_candidates.append(p)
                    if i < len(candidate_details):
                        unique_details.append(candidate_details[i])
                else:
                    # This is a duplicate - track it
                    deleted_duplicates.append(p)
                    if i < len(candidate_details):
                        deleted_duplicate_details.append(candidate_details[i])
            
            duplicates_removed = len(candidates) - len(unique_candidates)
            
            self.logger.info(f"Total unique candidates: {len(unique_candidates)}")
            dlog.log(f"\nCANDIDATE SUMMARY:")
            dlog.log(f"   {'─'*50}")
            dlog.log(f"   - Total generated: {len(candidates)}")
            dlog.log(f"   - Duplicates removed: {duplicates_removed}")
            dlog.log(f"   - Unique candidates: {len(unique_candidates)}")
            
            # List all candidates with full content BEFORE evaluation
            dlog.log(f"\nALL CANDIDATES FOR ROUND {round_num}:")
            dlog.log(f"   {'─'*50}")
            for i, prompt in enumerate(unique_candidates):
                source = unique_details[i]['mutator'] if i < len(unique_details) else 'unknown'
                dlog.log(f"\n   +{'─'*56}+")
                dlog.log(f"   | CANDIDATE {i+1} | Source: {source}")
                dlog.log(f"   +{'─'*56}+")
                
                # Show TASK_INSTRUCTION
                dlog.log(f"   | TASK_INSTRUCTION:")
                for line in prompt.task_instruction.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show TASK_DETAIL
                dlog.log(f"   | TASK_DETAIL:")
                for line in prompt.task_detail.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show OUTPUT_FORMAT
                dlog.log(f"   | OUTPUT_FORMAT:")
                for line in prompt.output_format.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show EXAMPLES
                dlog.log(f"   | EXAMPLES:")
                for line in prompt.examples.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show COT_HINTER
                dlog.log(f"   | COT_HINTER:")
                for line in prompt.cot_hinter.split('\n'):
                    dlog.log(f"   |    {line}")
                
                dlog.log(f"   +{'─'*56}+")
            
            # Evaluate all candidates
            dlog.log(f"\nEVALUATION PHASE:")
            dlog.log(f"   {'─'*50}")
            
            scored_candidates = []
            for i, prompt in enumerate(unique_candidates):
                dlog.log(f"\n   Evaluating candidate {i+1}/{len(unique_candidates)}...")
                score = self._evaluate_prompt(prompt)
                prompt.score = score
                scored_candidates.append((prompt, score))
                history.add(prompt, score, round_num)
                
                # Determine source info
                source = prompt.action or 'init'
                if i < len(unique_details):
                    source = unique_details[i]['mutator']
                
                self.logger.info(f"  Candidate {i+1}: {score:.2%} ({prompt.action or 'init'})")
                
                # Show result
                status = "[BEST]" if score >= history.best_score else ""
                dlog.log(f"   Candidate {i+1}: Score {score:.2%} | Source: {source} {status}")
            
            # Select top-K (beam search)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [p for p, s in scored_candidates[:self.beam_size]]
            
            best_this_round = scored_candidates[0]
            worst_this_round = scored_candidates[-1] if scored_candidates else (None, 0)
            
            self.logger.info(f"\nBest this round: {best_this_round[1]:.2%}")
            self.logger.info(f"Beam: {[f'{p.score:.2%}' for p in beam]}")
            
            # Round summary with full evaluation results
            dlog.log(f"\nROUND {round_num} RESULTS:")
            dlog.log(f"   {'─'*50}")
            dlog.log(f"   EVALUATION SCORES (sorted best to worst):")
            for rank, (p, s) in enumerate(scored_candidates, 1):
                status = "-> SELECTED FOR BEAM" if rank <= self.beam_size else ""
                dlog.log(f"   {rank}. Score: {s:.2%} | Action: {p.action or 'init'} {status}")
            
            dlog.log(f"\n   ROUND STATISTICS:")
            dlog.log(f"   - Best this round:  {best_this_round[1]:.2%}")
            dlog.log(f"   - Worst this round: {worst_this_round[1]:.2%}")
            dlog.log(f"   - Score range:      {worst_this_round[1]:.2%} -> {best_this_round[1]:.2%}")
            dlog.log(f"   - Global best:      {history.best_score:.2%}")
            
            dlog.log(f"\n   NEW BEAM (top {self.beam_size} prompts for next round):")
            for idx, p in enumerate(beam):
                dlog.log(f"\n   +{'─'*56}+")
                dlog.log(f"   | BEAM PROMPT {idx+1}/{len(beam)} | Score: {p.score:.2%}")
                dlog.log(f"   | Action: {p.action or 'init'}")
                dlog.log(f"   +{'─'*56}+")
                
                # Show TASK_INSTRUCTION
                dlog.log(f"   | TASK_INSTRUCTION:")
                for line in p.task_instruction.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show TASK_DETAIL
                dlog.log(f"   | TASK_DETAIL:")
                for line in p.task_detail.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show OUTPUT_FORMAT
                dlog.log(f"   | OUTPUT_FORMAT:")
                for line in p.output_format.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show EXAMPLES
                dlog.log(f"   | EXAMPLES:")
                for line in p.examples.split('\n'):
                    dlog.log(f"   |    {line}")
                
                # Show COT_HINTER
                dlog.log(f"   | COT_HINTER:")
                for line in p.cot_hinter.split('\n'):
                    dlog.log(f"   |    {line}")
                
                dlog.log(f"   +{'─'*56}+")
            
            # Save checkpoint with full details
            if self.output_dir:
                self._save_checkpoint_detailed(round_num, beam, scored_candidates, history, 
                                               deleted_duplicates, deleted_duplicate_details)
                dlog.log(f"\n   Checkpoint saved: checkpoint_round_{round_num}.json")
        
        # Get best result
        best_prompt, best_score = history.get_best()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info(f"Best score: {best_score:.2%}")
        self.logger.info("=" * 60)
        
        # Final summary - log to both console and file
        dlog.log(f"\n{'='*60}")
        dlog.log(f"OPTIMIZATION COMPLETE")
        dlog.log(f"{'='*60}")
        dlog.log(f"Final Statistics:")
        dlog.log(f"   - Total prompts evaluated: {len(history.history)}")
        dlog.log(f"   - Best score achieved: {best_score:.2%}")
        dlog.log(f"   - Rounds completed: {self.num_rounds}")
        
        dlog.log(f"\n{'='*60}")
        dlog.log(f"BEST PROMPT (All Components)")
        dlog.log(f"{'='*60}")
        
        # Show TASK_INSTRUCTION
        dlog.log(f"\nTASK_INSTRUCTION:")
        for line in best_prompt.task_instruction.split('\n'):
            dlog.log(f"   {line}")
        
        # Show TASK_DETAIL
        dlog.log(f"\nTASK_DETAIL:")
        for line in best_prompt.task_detail.split('\n'):
            dlog.log(f"   {line}")
        
        # Show OUTPUT_FORMAT
        dlog.log(f"\nOUTPUT_FORMAT:")
        for line in best_prompt.output_format.split('\n'):
            dlog.log(f"   {line}")
        
        # Show EXAMPLES
        dlog.log(f"\nEXAMPLES:")
        for line in best_prompt.examples.split('\n'):
            dlog.log(f"   {line}")
        
        # Show COT_HINTER
        dlog.log(f"\nCOT_HINTER:")
        for line in best_prompt.cot_hinter.split('\n'):
            dlog.log(f"   {line}")
        
        dlog.log(f"\n{'='*60}")
        dlog.log(f"RENDERED PROMPT (What gets sent to LLM)")
        dlog.log(f"{'='*60}")
        for line in best_prompt.render().split('\n'):
            dlog.log(line)
        dlog.log(f"{'='*60}")
        
        # Log complete history
        dlog.log(f"\n{'='*60}")
        dlog.log(f"COMPLETE OPTIMIZATION HISTORY")
        dlog.log(f"{'='*60}")
        for round_num, prompt, score in history.history:
            dlog.log(f"Round {round_num}: {prompt.action or 'init'} -> Score: {score:.2%}")
        dlog.log(f"{'='*60}\n")
        
        # Save final results
        if self.output_dir:
            self._save_results_detailed(best_prompt, best_score, history)
            dlog.log(f"Results saved to {self.output_dir}")
        
        return best_prompt, best_score, history
    
    def _get_modified_components(self, old_prompt: Prompt, new_prompt: Prompt) -> List[str]:
        """Identify which components were modified between two prompts."""
        modified = []
        components = ['task_instruction', 'task_detail', 'output_format', 'examples', 'cot_hinter']
        
        for comp in components:
            old_val = getattr(old_prompt, comp, '')
            new_val = getattr(new_prompt, comp, '')
            if old_val != new_val:
                modified.append(comp.upper())
        
        return modified
    
    def _print_full_candidate(self, prompt: Prompt, candidate_num: int) -> None:
        """Print full content of a candidate prompt to console and log file."""
        dlog = self.detailed_log
        
        dlog.log(f"\n      ┌{'─'*56}┐")
        dlog.log(f"      │ CANDIDATE {candidate_num} - FULL CONTENT{' '*(38-len(str(candidate_num)))}│")
        dlog.log(f"      ├{'─'*56}┤")
        
        # Task Instruction
        dlog.log(f"      │  TASK_INSTRUCTION:")
        self._log_wrapped_content(prompt.task_instruction, dlog, prefix="      │    ")
        
        # Output Format
        dlog.log(f"      │  OUTPUT_FORMAT:")
        self._log_wrapped_content(prompt.output_format, dlog, prefix="      │    ")
        
        # Examples
        dlog.log(f"      │  EXAMPLES:")
        self._log_wrapped_content(prompt.examples, dlog, prefix="      │    ")
        
        dlog.log(f"      └{'─'*56}┘")
    
    def _print_full_prompt_components(self, prompt: Prompt) -> None:
        """Print all components of a prompt to console and log file."""
        dlog = self.detailed_log
        
        dlog.log(f"\n   ┌{'─'*56}┐")
        dlog.log(f"   │ PROMPT COMPONENTS{' '*38}│")
        dlog.log(f"   ├{'─'*56}┤")
        
        # Task Instruction
        dlog.log(f"   │  TASK_INSTRUCTION:")
        self._log_wrapped_content(prompt.task_instruction, dlog, prefix="   │    ")
        dlog.log(f"   │")
        
        # Output Format
        dlog.log(f"   │ � OUTPUT_FORMAT:")
        self._log_wrapped_content(prompt.output_format, dlog, prefix="   │    ")
        dlog.log(f"   │")
        
        # Examples
        dlog.log(f"   │ � EXAMPLES:")
        self._log_wrapped_content(prompt.examples, dlog, prefix="   │    ")
        
        dlog.log(f"   └{'─'*56}┘")
    
    def _print_wrapped_content(self, content: str, prefix: str = "", max_width: int = 52) -> None:
        """Print content wrapped to fit within box, handling multi-line strings."""
        dlog = self.detailed_log
        if not content:
            dlog.log(f"{prefix}(empty)")
            return
        
        lines = content.split('\n')
        for line in lines:
            # Wrap long lines
            while len(line) > max_width:
                dlog.log(f"{prefix}{line[:max_width]}")
                line = line[max_width:]
            dlog.log(f"{prefix}{line}")
    
    def _log_wrapped_content(self, content: str, dlog: DetailedLogger, prefix: str = "", max_width: int = 52) -> None:
        """Log content wrapped to fit within box, to both console and file."""
        if not content:
            dlog.log(f"{prefix}(empty)")
            return
        
        lines = content.split('\n')
        for line in lines:
            # Wrap long lines
            while len(line) > max_width:
                dlog.log(f"{prefix}{line[:max_width]}")
                line = line[max_width:]
            dlog.log(f"{prefix}{line}")
    
    def _evaluate_prompt(self, prompt: Prompt) -> float:
        """
        Evaluate a prompt on the validation set.
        
        Returns accuracy (0.0 to 1.0).
        """
        valid_data = self.task.get_valid_data()
        
        if not valid_data:
            return 0.0
        
        correct = 0
        total = len(valid_data)
        dlog = self.detailed_log
        
        for i, example in enumerate(valid_data):
            question = example.get('question', '')
            ground_truth = example.get('answer', '')
            
            # Progress indicator
            print(f"      [{i+1}/{total}] Evaluating...", end=" ", flush=True)
            
            try:
                # Generate prediction
                full_prompt = prompt.render(question)
                prediction = self.eval_llm.generate(full_prompt, temperature=0)
                
                # Check if correct
                is_correct = self.task.evaluate(prediction, ground_truth)
                if is_correct:
                    correct += 1
                    print("✓")
                else:
                    print("✗")
                    
                # Log details
                dlog.log(f"      [{i+1}/{total}] Q: {question[:50]}... -> {'✓' if is_correct else '✗'}", also_print=False)
                
            except Exception as e:
                print(f"ERROR: {e}")
                dlog.log(f"      [{i+1}/{total}] ERROR: {e}", also_print=False)
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"      Result: {correct}/{total} = {accuracy:.1%}")
        return accuracy
    
    def _save_checkpoint(self, round_num: int, beam: List[Prompt], history: PromptHistory):
        """Save a checkpoint after each round."""
        checkpoint = {
            'round': round_num,
            'beam': [p.to_dict() for p in beam],
            'best_score': history.best_score,
        }
        
        path = os.path.join(self.output_dir, f'checkpoint_round_{round_num}.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_results(self, best_prompt: Prompt, best_score: float, history: PromptHistory):
        """Save final results."""
        results = {
            'best_score': best_score,
            'best_prompt': best_prompt.to_dict(),
            'history_summary': history.summary(),
            'num_prompts_tried': len(history.history),
        }
        
        path = os.path.join(self.output_dir, 'final_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save best prompt as text
        prompt_path = os.path.join(self.output_dir, 'best_prompt.txt')
        with open(prompt_path, 'w') as f:
            f.write(str(best_prompt))
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def _log_full_prompt(self, prompt: Prompt, dlog: DetailedLogger) -> None:
        """Log full prompt content to the detailed logger."""
        dlog.log("TASK_INSTRUCTION:")
        dlog.log(f"   {prompt.task_instruction}")
        dlog.log("TASK_DETAIL:")
        dlog.log(f"   {prompt.task_detail}")
        dlog.log("OUTPUT_FORMAT:")
        dlog.log(f"   {prompt.output_format}")
        dlog.log("EXAMPLE_HINTER:")
        dlog.log(f"   {prompt.example_hinter or '(none)'}")
        dlog.log("EXAMPLES:")
        for line in prompt.examples.split('\n'):
            dlog.log(f"   {line}")
        dlog.log("COT_HINTER:")
        dlog.log(f"   {prompt.cot_hinter}")
    def _log_prompt_components(self, prompt: Prompt, dlog: DetailedLogger, prefix: str = "") -> None:
        """Log all components of a prompt to the detailed logger."""
        dlog.log(f"{prefix} TASK_INSTRUCTION:")
        for line in prompt.task_instruction.split('\n'):
            dlog.log(f"{prefix}   {line}")
        
        dlog.log(f"{prefix} TASK_DETAIL:")
        for line in prompt.task_detail.split('\n'):
            dlog.log(f"{prefix}   {line}")
        
        dlog.log(f"{prefix} OUTPUT_FORMAT:")
        for line in prompt.output_format.split('\n'):
            dlog.log(f"{prefix}   {line}")
        
        dlog.log(f"{prefix} EXAMPLE_HINTER:")
        dlog.log(f"{prefix}   {prompt.example_hinter or '(none)'}")
        
        dlog.log(f"{prefix} EXAMPLES:")
        for line in prompt.examples.split('\n'):
            dlog.log(f"{prefix}   {line}")
        
        dlog.log(f"{prefix}COT_HINTER:")
        dlog.log(f"{prefix}   {prompt.cot_hinter}")
    
    def _save_checkpoint_detailed(self, round_num: int, beam: List[Prompt], 
                                   scored_candidates: List[Tuple[Prompt, float]], 
                                   history: PromptHistory,
                                   deleted_duplicates: List[Prompt] = None,
                                   deleted_duplicate_details: List[dict] = None):
        """Save a detailed checkpoint after each round."""
        checkpoint = {
            'round': round_num,
            'beam': [p.to_dict() for p in beam],
            'best_score': history.best_score,
            'all_candidates': [
                {
                    'rank': i + 1,
                    'score': score,
                    'action': p.action or 'init',
                    'prompt': p.to_dict(),
                    'rendered_prompt': p.render()
                }
                for i, (p, score) in enumerate(scored_candidates)
            ],
            'deleted_prompts': [
                {
                    'reason': 'duplicate',
                    'action': p.action or 'init',
                    'mutator': deleted_duplicate_details[i]['mutator'] if deleted_duplicate_details and i < len(deleted_duplicate_details) else 'unknown',
                    'prompt': p.to_dict(),
                    'rendered_prompt': p.render()
                }
                for i, p in enumerate(deleted_duplicates or [])
            ]
        }
        
        path = os.path.join(self.output_dir, f'checkpoint_round_{round_num}.json')
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_results_detailed(self, best_prompt: Prompt, best_score: float, history: PromptHistory):
        """Save detailed final results including all prompts tried."""
        # Build complete history
        all_prompts = []
        for round_num, prompt, score in history.history:
            all_prompts.append({
                'round': round_num,
                'score': score,
                'action': prompt.action or 'init',
                'prompt_components': prompt.to_dict(),
                'rendered_prompt': prompt.render()
            })
        
        results = {
            'best_score': best_score,
            'best_prompt': {
                'components': best_prompt.to_dict(),
                'rendered': best_prompt.render()
            },
            'statistics': {
                'total_prompts_evaluated': len(history.history),
                'rounds_completed': self.num_rounds,
                'beam_size': self.beam_size
            },
            'complete_history': all_prompts,
            'history_summary': history.summary(),
        }
        
        path = os.path.join(self.output_dir, 'final_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best prompt as text
        prompt_path = os.path.join(self.output_dir, 'best_prompt.txt')
        with open(prompt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BEST PROMPT - COMPONENTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"TASK_INSTRUCTION:\n{best_prompt.task_instruction}\n\n")
            f.write(f"TASK_DETAIL:\n{best_prompt.task_detail}\n\n")
            f.write(f"OUTPUT_FORMAT:\n{best_prompt.output_format}\n\n")
            f.write(f"EXAMPLE_HINTER:\n{best_prompt.example_hinter or '(none)'}\n\n")
            f.write(f"EXAMPLES:\n{best_prompt.examples}\n\n")
            f.write(f"COT_HINTER:\n{best_prompt.cot_hinter}\n\n")
            f.write("=" * 60 + "\n")
            f.write("RENDERED PROMPT\n")
            f.write("=" * 60 + "\n\n")
            f.write(best_prompt.render())
        
        self.logger.info(f"Results saved to {self.output_dir}")
