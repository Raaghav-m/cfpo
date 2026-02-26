"""
Groq Model Integration - FREE Inference Provider

Groq offers a generous free tier with high-speed inference:
- FREE: No credit card required
- Fast: LPU (Language Processing Unit) inference
- Models: Llama 3.1, Mixtral, Gemma

Get your free API key at: https://console.groq.com/keys
Set environment variable: export GROQ_API_KEY=your_key_here
"""

import os
import time
import requests
from typing import Optional, List
import logging

from .base import LLMModel


class GroqModel(LLMModel):
    """
    Groq model - FREE high-speed inference.
    
    Free Tier Limits (very generous):
    - Llama 3.1 8B: 30 requests/min, 14,400 requests/day
    - Llama 3.1 70B: 30 requests/min, 14,400 requests/day
    - Mixtral 8x7B: 30 requests/min, 14,400 requests/day
    - Gemma 2 9B: 30 requests/min, 14,400 requests/day
    
    Get FREE key at: https://console.groq.com/keys
    """
    
    AVAILABLE_MODELS = {
        "llama-3.1-8b-instant": "Fast, good for most tasks",
        "llama-3.1-70b-versatile": "Most capable, slower",
        "llama-3.3-70b-versatile": "Latest Llama 3.3",
        "mixtral-8x7b-32768": "Good for long contexts",
        "gemma2-9b-it": "Google's Gemma 2",
    }
    
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Groq model.

        Args:
            model_name: Groq model ID (see AVAILABLE_MODELS)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            api_key: Groq API key (or set GROQ_API_KEY env var)
            logger: Logger instance
        """
        super().__init__(model_name, max_tokens, temperature, logger)
        
        self.api_key = api_key or os.environ.get('GROQ_API_KEY', '')
        
        if not self.api_key:
            raise ValueError(
                "Groq API key required!\n"
                "Get your FREE key at: https://console.groq.com/keys\n"
                "Then set: export GROQ_API_KEY=your_key_here"
            )
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.logger.info(f"✓ Groq Model: {model_name} (FREE tier)")
        self.logger.info(f"  Available models: {list(self.AVAILABLE_MODELS.keys())}")
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate response using Groq API.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional)

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": temp,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 10 * (attempt + 1)
                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    self.logger.error(f"Groq API error {response.status_code}: {response.text}")
                    return f"[Error: {response.status_code}]"
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                time.sleep(5)
                continue
                
            except Exception as e:
                self.logger.error(f"Error: {e}")
                return f"[Error: {str(e)}]"
        
        return "[Error: Max retries exceeded]"
    
    def generate_batch(self, prompts: List[str], temperature: Optional[float] = None) -> List[str]:
        """Generate responses for multiple prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, temperature)
            results.append(result)
            # Small delay to avoid rate limits
            if i < len(prompts) - 1:
                time.sleep(0.5)
        return results
