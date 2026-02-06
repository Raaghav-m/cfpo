"""
Monte Carlo Sampling Mutator

Random exploration: generates random variations of prompt
components to explore the search space.
"""

import re
import random
from typing import List, Optional
import logging
import sys
import os

# Add parent directory to path to import from optimizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import Mutator
from prompts import Prompt
from models import LLMModel
from tasks import Task


def _log_to_file_and_console(message: str):
    """Helper to log to both console and the global log file."""
    print(message)
    # Try to get the global logger from optimizer
    try:
        from optimizer import get_global_logger
        dlog = get_global_logger()
        if dlog:
            dlog.log(message, also_print=False)  # Already printed above
    except:
        pass  # Silently fail if logger not available


class MonteCarloMutator(Mutator):
    """
    Mutator that randomly explores variations.
    
    Strategy:
    1. Randomly select components to modify
    2. Ask LLM to generate variations (synonyms, rephrases)
    3. Return diverse set of new prompts
    """
    
    def __init__(
        self,
        llm: LLMModel,
        task: Task,
        component_keys: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Monte Carlo mutator.

        Args:
            llm: LLM for generating variations
            task: The task being optimized
            component_keys: Components that can be modified
            logger: Logger instance
        """
        super().__init__(llm, task, component_keys, logger)
    
    def mutate(
        self,
        prompt: Prompt,
        num_mutations: int = 3,
        temperature: float = 0.8,
        round: int = 0,
        max_components: int = 2,
    ) -> List[Prompt]:
        """
        Generate random variations of the prompt.

        Args:
            prompt: Current prompt
            num_mutations: Number of variations to generate
            temperature: LLM temperature (higher = more creative)
            round: Current optimization round
            max_components: Maximum components to modify per mutation

        Returns:
            List of varied prompts
        """
        self.logger.info(f"[MonteCarlo] Generating {num_mutations} variations for round {round}")
        
        # Log available keywords
        _log_to_file_and_console(f"\n      [MonteCarlo] Available keywords for mutation:")
        _log_to_file_and_console(f"         {self.component_keys}")
        
        new_prompts = []
        
        for i in range(num_mutations):
            # Randomly select components to modify
            num_to_modify = random.randint(1, min(max_components, len(self.component_keys)))
            components_to_modify = random.sample(self.component_keys, num_to_modify)
            
            self.logger.info(f"[MonteCarlo] Variation {i+1}: modifying {components_to_modify}")
            
            # Log selected keywords with details
            _log_to_file_and_console(f"\n       [MonteCarlo] Variation {i+1}:")
            _log_to_file_and_console(f"         - Keywords selected: {components_to_modify}")
            _log_to_file_and_console(f"         - Number of keywords: {num_to_modify}/{len(self.component_keys)}")
            
            # Generate variations for each component
            component_keys = []
            component_values = []
            
            for component_key in components_to_modify:
                current_value = prompt.get_component(component_key)
                current_preview = current_value[:50] + "..." if len(current_value) > 50 else current_value
                current_preview = current_preview.replace('\n', ' ')
                _log_to_file_and_console(f"         - Modifying {component_key}:")
                _log_to_file_and_console(f"           Old: \"{current_preview}\"")
                
                if component_key == 'EXAMPLES':
                    new_value = self._vary_examples(prompt, temperature)
                else:
                    new_value = self._vary_component(prompt, component_key, current_value, temperature)
                
                if new_value:
                    component_keys.append(component_key)
                    component_values.append(new_value)
                    new_preview = new_value[:50] + "..." if len(new_value) > 50 else new_value
                    new_preview = new_preview.replace('\n', ' ')
                    _log_to_file_and_console(f"           New: \"{new_preview}\"")
            
            # Create new prompt
            if component_keys:
                new_prompt = prompt.generate(
                    round=round,
                    component_key=component_keys,
                    component_content=component_values,
                    action_desc="monte_carlo",
                    reason=f"Random variation of: {', '.join(component_keys)}"
                )
                
                if str(new_prompt) != str(prompt):
                    new_prompts.append(new_prompt)
                    self.logger.info(f"[MonteCarlo] Created variation {i+1}")
        
        return new_prompts
    
    def _vary_component(
        self, 
        prompt: Prompt, 
        component_key: str, 
        current_value: str,
        temperature: float
    ) -> Optional[str]:
        """Generate a variation of a single component."""
        
        if not current_value.strip():
            # Generate new content if empty
            variation_prompt = f"""{self._get_meta_prompt_header(prompt)}

The {component_key} component is currently empty.

Please generate a suitable {component_key} for this task.
Keep it concise and effective.

Your {component_key}:"""
        else:
            variation_prompt = f"""{self._get_meta_prompt_header(prompt)}

Please create a DIFFERENT version of the {component_key} component.
Keep the same meaning but vary the wording, structure, or style.

Current {component_key}:
\"\"\"{current_value}\"\"\"

Your varied {component_key} (just the content, no explanation):"""
        
        response = self.llm.generate(variation_prompt, temperature=temperature)
        
        # Clean up response
        response = re.sub(r'^[\n"\' ]+|[\n"\' ]+$', '', response)
        response = response.strip()
        
        return response if response else None
    
    def _vary_examples(self, prompt: Prompt, temperature: float) -> Optional[str]:
        """Generate a variation of the examples section."""
        
        current_examples = prompt.examples
        
        variation_prompt = f"""{self._get_meta_prompt_header(prompt)}

Current EXAMPLES section:
\"\"\"{current_examples}\"\"\"

Please generate a variation of the EXAMPLES. Choose ONE action:
1. ADD one new example
2. REMOVE one example  
3. MODIFY an existing example (change numbers or scenario)

Provide the complete new EXAMPLES section:"""
        
        response = self.llm.generate(variation_prompt, temperature=temperature)
        
        # Clean up
        response = re.sub(r'^[\n"\' ]+|[\n"\' ]+$', '', response)
        
        return response if response else None
