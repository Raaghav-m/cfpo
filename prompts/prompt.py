"""
Prompt Management

Handles prompt structure, components, and history tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy


@dataclass
class Prompt:
    """
    Represents a structured prompt with multiple components.
    
    Components:
        - task_instruction: Main instruction (what to do)
        - task_detail: Additional details/constraints
        - output_format: How to format the answer
        - examples: Few-shot examples
        - cot_hinter: Chain-of-thought hint ("Let's think step by step")
    """
    
    task_instruction: str = ""
    task_detail: str = ""
    output_format: str = ""
    example_hinter: str = ""
    examples: str = ""
    cot_hinter: str = ""
    
    # Metadata
    round: int = 0
    score: float = 0.0
    action: str = ""
    parent_id: Optional[int] = None
    
    # Format functions (for rendering)
    prompt_format: Optional[Tuple] = None
    query_format: Optional[Tuple] = None
    
    def render(self, question: str = "") -> str:
        """
        Render the complete prompt with a question.
        
        Args:
            question: The question to include in the prompt
            
        Returns:
            Fully rendered prompt string
        """
        parts = []
        
        if self.task_instruction:
            parts.append(f"# Task\n{self.task_instruction}")
        
        if self.task_detail:
            parts.append(f"# Details\n{self.task_detail}")
        
        if self.output_format:
            parts.append(f"# Output Format\n{self.output_format}")
        
        if self.examples:
            hinter = self.example_hinter or "# Examples"
            parts.append(f"{hinter}\n{self.examples}")
        
        if self.cot_hinter:
            parts.append(self.cot_hinter)
        
        if question:
            parts.append(f"# Question\n{question}")
        
        return "\n\n".join(parts)
    
    def get_component(self, key: str) -> str:
        """Get a component by key name."""
        key_map = {
            'TASK_INSTRUCTION': 'task_instruction',
            'TASK_DETAIL': 'task_detail',
            'OUTPUT_FORMAT': 'output_format',
            'EXAMPLE_HINTER': 'example_hinter',
            'EXAMPLES': 'examples',
            'COT_HINTER': 'cot_hinter',
        }
        attr = key_map.get(key, key.lower())
        return getattr(self, attr, "")
    
    def set_component(self, key: str, value: str) -> None:
        """Set a component by key name."""
        key_map = {
            'TASK_INSTRUCTION': 'task_instruction',
            'TASK_DETAIL': 'task_detail',
            'OUTPUT_FORMAT': 'output_format',
            'EXAMPLE_HINTER': 'example_hinter',
            'EXAMPLES': 'examples',
            'COT_HINTER': 'cot_hinter',
        }
        attr = key_map.get(key, key.lower())
        setattr(self, attr, value)
    
    def generate(
        self,
        round: int,
        component_key: List[str],
        component_content: List[str],
        action_desc: str = "",
        reason: Optional[str] = None,
    ) -> "Prompt":
        """
        Generate a new prompt with modified components.
        
        Args:
            round: Current optimization round
            component_key: List of component keys to modify
            component_content: List of new content for each component
            action_desc: Description of the action taken
            reason: Reason for the modification
            
        Returns:
            New Prompt with modifications
        """
        new_prompt = deepcopy(self)
        new_prompt.round = round
        new_prompt.action = action_desc
        
        for key, content in zip(component_key, component_content):
            new_prompt.set_component(key, content)
        
        return new_prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'TASK_INSTRUCTION': self.task_instruction,
            'TASK_DETAIL': self.task_detail,
            'OUTPUT_FORMAT': self.output_format,
            'EXAMPLE_HINTER': self.example_hinter,
            'EXAMPLES': self.examples,
            'COT_HINTER': self.cot_hinter,
            'round': self.round,
            'score': self.score,
            'action': self.action,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        """Create from dictionary."""
        return cls(
            task_instruction=data.get('TASK_INSTRUCTION', ''),
            task_detail=data.get('TASK_DETAIL', ''),
            output_format=data.get('OUTPUT_FORMAT', ''),
            example_hinter=data.get('EXAMPLE_HINTER', ''),
            examples=data.get('EXAMPLES', ''),
            cot_hinter=data.get('COT_HINTER', ''),
            round=data.get('round', 0),
            score=data.get('score', 0.0),
            action=data.get('action', ''),
        )
    
    def __str__(self) -> str:
        return self.render()


class PromptHistory:
    """
    Tracks the history of prompts during optimization.
    
    Maintains a record of all prompts tried, their scores,
    and the modifications made at each step.
    """
    
    def __init__(self, init_prompt: Prompt, init_round: int = 0):
        """
        Initialize history with starting prompt.
        
        Args:
            init_prompt: Initial prompt to start with
            init_round: Starting round number
        """
        self.history: List[Tuple[int, Prompt, float]] = []
        self.current_prompt = init_prompt
        self.current_round = init_round
        self.best_prompt = init_prompt
        self.best_score = 0.0
        
        # Add initial prompt to history
        self.add(init_prompt, 0.0, init_round)
    
    def add(self, prompt: Prompt, score: float, round: int) -> None:
        """
        Add a prompt to history.
        
        Args:
            prompt: The prompt to add
            score: Its evaluation score
            round: The optimization round
        """
        prompt.score = score
        prompt.round = round
        self.history.append((round, prompt, score))
        
        # Update best if improved
        if score > self.best_score:
            self.best_score = score
            self.best_prompt = prompt
    
    def update_current(self, prompt: Prompt, round: int) -> None:
        """Update the current prompt."""
        self.current_prompt = prompt
        self.current_round = round
    
    def get_best(self) -> Tuple[Prompt, float]:
        """Get the best prompt and its score."""
        return self.best_prompt, self.best_score
    
    def get_history(self) -> List[Tuple[int, Prompt, float]]:
        """Get full history."""
        return self.history
    
    def summary(self) -> str:
        """Get a summary of optimization history."""
        lines = ["=" * 50, "OPTIMIZATION HISTORY", "=" * 50]
        
        for round_num, prompt, score in self.history:
            action = prompt.action or "init"
            lines.append(f"Round {round_num}: {action} -> Score: {score:.2%}")
        
        lines.append("=" * 50)
        lines.append(f"Best Score: {self.best_score:.2%}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
