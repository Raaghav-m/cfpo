"""
Ollama Model - Free Local LLM Inference

Ollama runs LLMs locally on your machine (CPU or GPU).
Install: curl -fsSL https://ollama.com/install.sh | sh
Pull a model: ollama pull phi (or llama2, mistral, etc.)
"""

import requests
from typing import Optional
import logging

from .base import LLMModel


class OllamaModel(LLMModel):
    """Ollama local LLM inference."""
    
    def __init__(
        self,
        model_name: str = "phi",  # Small, fast model for CPU
        max_tokens: int = 2048,
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Ollama model.

        Args:
            model_name: Ollama model name (phi, llama2, mistral, etc.)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            base_url: Ollama server URL
            logger: Logger instance
        """
        super().__init__(model_name, max_tokens, temperature, logger)
        self.base_url = base_url
        
        # Check if Ollama is running
        self._check_connection()

    def _check_connection(self) -> None:
        """Verify Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            self.logger.info(f"✓ Connected to Ollama at {self.base_url}")
        except requests.exceptions.ConnectionError:
            self.logger.warning(
                f"⚠ Ollama not running at {self.base_url}. "
                "Start with: ollama serve"
            )

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate response using Ollama.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional)

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": self.max_tokens,
                    }
                },
                timeout=120  # 2 min timeout - prevents hanging
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError:
            error_msg = "Ollama not running. Start with: ollama serve"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            self.logger.error(f"Ollama error: {e}")
            raise
