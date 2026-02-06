"""
Case Diagnosis Mutator

Learns from mistakes: analyzes incorrect predictions and 
modifies the prompt to fix those errors.
"""

import re
from typing import List, Optional, Tuple
import logging

from .base import Mutator
from prompts import Prompt
from models import LLMModel
from tasks import Task


class CaseDiagnosisMutator(Mutator):
    """
    Mutator that learns from errors.
    
    Strategy:
    1. Find examples where current prompt fails
    2. Ask LLM: "Why did this fail? How should we modify the prompt?"
    3. Generate improved prompt based on diagnosis
    """
    
    def __init__(
        self,
        llm: LLMModel,
        eval_llm: LLMModel,
        task: Task,
        num_errors_to_analyze: int = 2,
        num_correct_to_analyze: int = 1,
        component_keys: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Case Diagnosis mutator.

        Args:
            llm: LLM for generating improvements
            eval_llm: LLM for evaluating prompts
            task: The task being optimized
            num_errors_to_analyze: How many errors to analyze per mutation
            num_correct_to_analyze: How many correct examples to include
            component_keys: Components that can be modified
            logger: Logger instance
        """
        super().__init__(llm, task, component_keys, logger)
        self.eval_llm = eval_llm
        self.num_errors = num_errors_to_analyze
        self.num_correct = num_correct_to_analyze
    
    def mutate(
        self,
        prompt: Prompt,
        num_mutations: int = 1,
        temperature: float = 0.7,
        round: int = 0,
    ) -> List[Prompt]:
        """
        Generate mutations by analyzing errors.

        Args:
            prompt: Current prompt
            num_mutations: Number of improved prompts to generate
            temperature: LLM temperature
            round: Current optimization round

        Returns:
            List of improved prompts
        """
        self.logger.info(f"[CaseDiagnosis] Analyzing errors for round {round}")
        
        # Step 1: Get training examples and find errors
        train_data = self.task.get_train_batch(20)
        errors, corrects = self._find_errors(prompt, train_data)
        
        if not errors:
            self.logger.info("[CaseDiagnosis] No errors found, skipping mutation")
            return []
        
        self.logger.info(f"[CaseDiagnosis] Found {len(errors)} errors, {len(corrects)} correct")
        
        # Step 2: Generate improvements
        new_prompts = []
        for i in range(num_mutations):
            # Select errors to analyze
            selected_errors = errors[:self.num_errors]
            selected_corrects = corrects[:self.num_correct]
            
            # Generate improved prompt
            improved_prompt = self._generate_improvement(
                prompt, 
                selected_errors, 
                selected_corrects,
                temperature,
                round
            )
            
            if improved_prompt and str(improved_prompt) != str(prompt):
                new_prompts.append(improved_prompt)
                self.logger.info(f"[CaseDiagnosis] Generated improvement {i+1}")
        
        return new_prompts
    
    def _find_errors(
        self, 
        prompt: Prompt, 
        examples: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """
        Find examples where the prompt fails.
        
        Returns:
            (errors, corrects) - Lists of failed and successful examples
        """
        errors = []
        corrects = []
        
        for ex in examples:
            question = ex.get('question', '')
            ground_truth = ex.get('answer', '')
            
            # Generate prediction using current prompt
            full_prompt = prompt.render(question)
            prediction = self.eval_llm.generate(full_prompt, temperature=0)
            
            # Check if correct
            is_correct = self.task.evaluate(prediction, ground_truth)
            
            example_record = {
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': is_correct,
            }
            
            if is_correct:
                corrects.append(example_record)
            else:
                errors.append(example_record)
        
        return errors, corrects
    
    def _generate_improvement(
        self,
        prompt: Prompt,
        errors: List[dict],
        corrects: List[dict],
        temperature: float,
        round: int,
    ) -> Optional[Prompt]:
        """
        Generate an improved prompt based on error analysis.
        """
        # Build the diagnosis prompt
        diagnosis_prompt = self._build_diagnosis_prompt(prompt, errors, corrects)
        
        # Ask LLM to analyze and improve
        response = self.llm.generate(diagnosis_prompt, temperature=temperature)
        
        # Parse the improved components
        improvements = self._parse_improvements(response)
        
        if not improvements:
            return None
        
        # Generate new prompt with improvements
        component_keys = list(improvements.keys())
        component_values = list(improvements.values())
        
        new_prompt = prompt.generate(
            round=round,
            component_key=component_keys,
            component_content=component_values,
            action_desc="case_diagnosis",
            reason="Fixed errors from case analysis"
        )
        
        return new_prompt
    
    def _build_diagnosis_prompt(
        self, 
        prompt: Prompt, 
        errors: List[dict], 
        corrects: List[dict]
    ) -> str:
        """Build the prompt for error diagnosis."""
        
        # Format errors
        error_text = ""
        for i, err in enumerate(errors, 1):
            error_text += f"""
Error Case {i}:
- Question: {err['question'][:200]}...
- Expected: {self.task.extract_answer(err['ground_truth'])}
- Got: {self.task.extract_answer(err['prediction']) or 'Could not extract answer'}
"""
        
        # Format corrects
        correct_text = ""
        for i, cor in enumerate(corrects, 1):
            correct_text += f"""
Correct Case {i}:
- Question: {cor['question'][:200]}...
- Answer: {self.task.extract_answer(cor['ground_truth'])}
"""
        
        return f"""{self._get_meta_prompt_header(prompt)}

## Current Prompt
```
{prompt.render()[:1000]}
```

## Cases Where Prompt FAILED
{error_text}

## Cases Where Prompt SUCCEEDED
{correct_text}

## Your Task
Analyze why the prompt failed on the error cases. Then suggest improvements.

Provide improved versions of these components (only the ones that need changes):

<TASK_INSTRUCTION>
[Improved instruction here]
</TASK_INSTRUCTION>

<OUTPUT_FORMAT>
[Improved output format here]
</OUTPUT_FORMAT>

<COT_HINTER>
[Improved chain-of-thought hint here]
</COT_HINTER>

Focus on making the instructions clearer and more specific to prevent the observed errors.
"""
    
    def _parse_improvements(self, response: str) -> dict:
        """Parse improved components from LLM response."""
        improvements = {}
        
        # Parse each component
        for component in ['TASK_INSTRUCTION', 'OUTPUT_FORMAT', 'COT_HINTER', 'TASK_DETAIL']:
            pattern = rf'<{component}>\s*(.*?)\s*</{component}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and content.lower() not in ['[improved instruction here]', '']:
                    improvements[component] = content
        
        return improvements
