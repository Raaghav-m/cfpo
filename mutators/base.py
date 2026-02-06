"""
Base Mutator Interface

Mutators are strategies for modifying prompts to find improvements.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from prompts import Prompt
from models import LLMModel
from tasks import Task


class Mutator(ABC):
    """
    Abstract base class for prompt mutators.
    
    A mutator takes a prompt and generates variations of it.
    Different mutators use different strategies:
    - CaseDiagnosis: Learn from mistakes
    - MonteCarlo: Random exploration
    - Format: Change structure/formatting
    """
    
    # Components that can be mutated
    COMPONENT_KEYS = [
        'TASK_INSTRUCTION',
        'TASK_DETAIL', 
        'OUTPUT_FORMAT',
        'EXAMPLES',
        'COT_HINTER',
    ]
    
    def __init__(
        self,
        llm: LLMModel,
        task: Task,
        component_keys: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the mutator.

        Args:
            llm: LLM to use for generating mutations
            task: Task for context
            component_keys: Which components this mutator can modify
            logger: Logger instance
        """
        self.llm = llm
        self.task = task
        self.component_keys = component_keys or self.COMPONENT_KEYS
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def mutate(
        self,
        prompt: Prompt,
        num_mutations: int = 1,
        temperature: float = 0.7,
        round: int = 0,
    ) -> List[Prompt]:
        """
        Generate mutations of the prompt.

        Args:
            prompt: The prompt to mutate
            num_mutations: How many variations to generate
            temperature: LLM temperature for creativity
            round: Current optimization round

        Returns:
            List of mutated prompts
        """
        pass
    
    def __call__(self, *args, **kwargs) -> List[Prompt]:
        """Allow calling mutator like a function."""
        return self.mutate(*args, **kwargs)
    
    def _get_meta_prompt_header(self, prompt: Prompt) -> str:
        """
        Get the meta-prompt header that describes the task.
        
        This is used to give the LLM context about what
        the prompt is trying to accomplish.
        """
        return f"""You are an expert prompt engineer helping to improve a prompt.

The prompt is designed for the following task: {self.task.name}

Current prompt structure:
- Task Instruction: {prompt.task_instruction[:100]}...
- Output Format: {prompt.output_format[:100] if prompt.output_format else 'Not specified'}
- Has Examples: {'Yes' if prompt.examples else 'No'}
"""
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(components={self.component_keys})"
