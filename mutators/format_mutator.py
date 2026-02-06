"""
Format Mutator

Changes the structural format of prompts:
- XML tags vs Markdown vs Plain text
- Different example formats
- Various query structures
"""

import random
from typing import List, Optional, Callable, Tuple
import logging
import sys
import os

# Add parent directory to path
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


# Pre-defined format templates
FORMAT_TEMPLATES = {
    'markdown': {
        'name': 'Markdown',
        'render': lambda p: f"""# Task
{p.task_instruction}

## Details
{p.task_detail}

## Output Format
{p.output_format}

## Examples
{p.examples}

{p.cot_hinter}
""",
    },
    'xml': {
        'name': 'XML Tags',
        'render': lambda p: f"""<task>
<instruction>{p.task_instruction}</instruction>
<details>{p.task_detail}</details>
<output_format>{p.output_format}</output_format>
<examples>{p.examples}</examples>
</task>

{p.cot_hinter}
""",
    },
    'plain': {
        'name': 'Plain Text',
        'render': lambda p: f"""{p.task_instruction}

{p.task_detail}

{p.output_format}

{p.examples}

{p.cot_hinter}
""",
    },
    'structured': {
        'name': 'Structured',
        'render': lambda p: f"""=== TASK ===
{p.task_instruction}

=== INSTRUCTIONS ===
{p.task_detail}

=== OUTPUT REQUIREMENTS ===
{p.output_format}

=== EXAMPLES ===
{p.examples}

=== BEGIN ===
{p.cot_hinter}
""",
    },
    'conversational': {
        'name': 'Conversational',
        'render': lambda p: f"""I need your help with a task.

Here's what I need you to do: {p.task_instruction}

Some additional details: {p.task_detail}

Please format your answer like this: {p.output_format}

Here are some examples to guide you:
{p.examples}

{p.cot_hinter}
""",
    },
}


class FormatMutator(Mutator):
    """
    Mutator that changes prompt formatting.
    
    Strategy:
    1. Choose a different format template
    2. Re-render the prompt in new format
    3. This explores how presentation affects performance
    """
    
    COMPONENT_KEYS = ['PROMPT_FORMAT']  # Special component
    
    def __init__(
        self,
        llm: LLMModel,
        task: Task,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Format mutator.

        Args:
            llm: LLM (not heavily used, but kept for consistency)
            task: The task being optimized
            logger: Logger instance
        """
        super().__init__(llm, task, self.COMPONENT_KEYS, logger)
        self.formats = list(FORMAT_TEMPLATES.keys())
        self.current_format = 'markdown'
    
    def mutate(
        self,
        prompt: Prompt,
        num_mutations: int = 2,
        temperature: float = 0.7,
        round: int = 0,
    ) -> List[Prompt]:
        """
        Generate prompts with different formats.

        Args:
            prompt: Current prompt
            num_mutations: Number of format variations
            temperature: Not used (deterministic)
            round: Current optimization round

        Returns:
            List of reformatted prompts
        """
        self.logger.info(f"[FormatMutator] Generating {num_mutations} format variations")
        
        # Log available format keywords
        _log_to_file_and_console(f"\n      [FormatMutator] Available format keywords:")
        _log_to_file_and_console(f"         {self.formats}")
        _log_to_file_and_console(f"         Current format: {self.current_format}")
        
        new_prompts = []
        available_formats = [f for f in self.formats if f != self.current_format]
        
        _log_to_file_and_console(f"         Formats to try: {available_formats[:num_mutations]}")
        
        for i in range(min(num_mutations, len(available_formats))):
            new_format = available_formats[i]
            
            # Log format selection details
            _log_to_file_and_console(f"\n      [FormatMutator] Variation {i+1}:")
            _log_to_file_and_console(f"         - Format keyword selected: {new_format}")
            _log_to_file_and_console(f"         - Format style: {FORMAT_TEMPLATES[new_format]['name']}")
            
            # Show format preview
            template_preview = FORMAT_TEMPLATES[new_format]['render'](prompt)[:100].replace('\n', '\\n')
            _log_to_file_and_console(f"         - Preview: \"{template_preview}...\"")
            
            # Create a new prompt with the different format
            new_prompt = self._apply_format(prompt, new_format, round)
            
            if new_prompt:
                new_prompts.append(new_prompt)
                self.logger.info(f"[FormatMutator] Created {new_format} format")
                _log_to_file_and_console(f"         [OK] Created {new_format} format variant")
        
        return new_prompts
        
        return new_prompts
    
    def _apply_format(self, prompt: Prompt, format_name: str, round: int) -> Optional[Prompt]:
        """Apply a format template to create a new prompt."""
        
        if format_name not in FORMAT_TEMPLATES:
            return None
        
        template = FORMAT_TEMPLATES[format_name]
        
        # Create a copy with format metadata
        new_prompt = prompt.generate(
            round=round,
            component_key=['PROMPT_FORMAT'],
            component_content=[format_name],
            action_desc=f"format_{format_name}",
            reason=f"Changed to {template['name']} format"
        )
        
        # Store the format for rendering
        new_prompt.prompt_format = (format_name, template['render'])
        
        return new_prompt
    
    def get_available_formats(self) -> List[str]:
        """List available format templates."""
        return list(FORMAT_TEMPLATES.keys())
