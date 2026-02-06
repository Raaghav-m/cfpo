"""
HuggingFace Model Integration

Supports multiple modes:
1. Inference API - Remote serverless inference
2. Inference Providers - Via router.huggingface.co (Cerebras, Groq, Together, etc.)
3. Local Transformers - Load model locally via transformers library

Get your free API token at: https://huggingface.co/settings/tokens
Set environment variable: export HF_API_TOKEN=your_token_here
"""

import os
import time
import requests
from typing import Optional, Union
import logging

from .base import LLMModel


class HuggingFaceModel(LLMModel):
    """
    HuggingFace model supporting multiple inference modes.
    
    Modes:
        - "api": Use HuggingFace Inference API (serverless)
        - "providers": Use Inference Providers (router.huggingface.co)
        - "local": Load model locally via transformers
    """
    
    # Models that work well with different providers
    RECOMMENDED_MODELS = {
        "providers": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ],
        "api": [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta",
            "google/gemma-2b-it",
        ],
        "local": [
            "microsoft/phi-2",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]
    }
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        api_token: Optional[str] = None,
        mode: str = "providers",  # "api", "providers", or "local"
        provider: str = "auto",  # For providers mode: auto, together, fireworks, groq, etc.
        logger: Optional[logging.Logger] = None,
        device: str = "auto",  # For local mode
    ):
        """
        Initialize HuggingFace model.

        Args:
            model_name: HuggingFace model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            api_token: HF API token (or set HF_API_TOKEN env var)
            mode: Inference mode - "api", "providers", or "local"
            provider: Provider for "providers" mode
            logger: Logger instance
            device: Device for local inference
        """
        super().__init__(model_name, max_tokens, temperature, logger)
        
        self.mode = mode
        self.provider = provider
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Get API token for remote inference
        self.api_token = api_token or os.environ.get('HF_API_TOKEN', os.environ.get('HUGGINGFACE_TOKEN', ''))
        
        if mode in ("api", "providers") and not self.api_token:
            self.logger.warning(
                "No HuggingFace API token found. Set HF_API_TOKEN environment variable.\n"
                "Get free token at: https://huggingface.co/settings/tokens"
            )
        
        # Set up based on mode
        if mode == "providers":
            self.api_url = "https://router.huggingface.co/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            self.logger.info(f"✓ HuggingFace Providers: {model_name} (provider: {provider})")
            
        elif mode == "api":
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            self.headers = {"Authorization": f"Bearer {self.api_token}"}
            self.logger.info(f"✓ HuggingFace Inference API: {model_name}")
            
        elif mode == "local":
            self._init_local_model()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'api', 'providers', or 'local'.")

    def _init_local_model(self):
        """Initialize local transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.logger.info(f"Loading local model: {self.model_name}...")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.logger.info(f"✓ Local model loaded on {device}")
            
        except ImportError:
            raise ImportError(
                "transformers package required for local mode.\n"
                "Install with: pip install transformers torch"
            )

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate response using the configured mode.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional)

        Returns:
            Generated text
        """
        if self.mode == "local":
            return self._generate_local(prompt, temperature)
        elif self.mode == "providers":
            return self._generate_providers(prompt, temperature)
        else:
            return self._generate_api(prompt, temperature)

    def _generate_providers(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Generate using HuggingFace Inference Providers."""
        temp = temperature if temperature is not None else self.temperature
        
        # Build model string with provider suffix if specified
        model_str = self.model_name
        if self.provider != "auto":
            model_str = f"{self.model_name}:{self.provider}"
        
        payload = {
            "model": model_str,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": max(temp, 0.01),
            "stream": False
        }
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 429:
                    wait_time = int(response.headers.get("Retry-After", 30))
                    self.logger.info(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 503:
                    result = response.json()
                    wait_time = result.get("estimated_time", 30)
                    self.logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                
                return str(result)
                
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HF Providers API error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise
        
        return ""

    def _generate_api(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Generate using HuggingFace Inference API."""
        temp = temperature if temperature is not None else self.temperature
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": max(temp, 0.01),
                "do_sample": temp > 0,
                "return_full_text": False,
            }
        }
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 503:
                    result = response.json()
                    wait_time = result.get("estimated_time", 30)
                    self.logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                
                return str(result)
                
            except Exception as e:
                self.logger.error(f"HF API error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))
                else:
                    raise
        
        return ""

    def _generate_local(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Generate using local transformers model."""
        temp = temperature if temperature is not None else self.temperature
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=max(temp, 0.01) if temp > 0 else None,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return text

    def inference(self, prompt: str, desc: str = "", temperature: Optional[float] = None) -> str:
        """
        Inference wrapper for compatibility with CFPO interface.
        
        Args:
            prompt: Input prompt
            desc: Description of the inference task (for logging)
            temperature: Override temperature
            
        Returns:
            Generated text
        """
        if desc:
            self.logger.debug(f"Inference: {desc}")
        return self.generate(prompt, temperature)
