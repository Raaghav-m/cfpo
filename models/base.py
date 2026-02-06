"""
Base LLM Model Interface

All LLM backends must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import logging


class LLMModel(ABC):
    """Abstract base class for all LLM models."""
    
    def __init__(
        self,
        model_name: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the LLM model.

        Args:
            model_name: Name/path of the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic, 1 = creative)
            logger: Logger instance for debugging
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate a response for a single prompt.

        Args:
            prompt: The input prompt
            temperature: Override default temperature (optional)

        Returns:
            Generated text response
        """
        pass

    def generate_batch(self, prompts: List[str], temperature: Optional[float] = None) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            temperature: Override default temperature (optional)

        Returns:
            List of generated text responses
        """
        return [self.generate(p, temperature) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
