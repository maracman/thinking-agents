"""
LLM Service module with flexible provider support and graceful fallbacks
"""

import os
import json
import logging
import time
import requests
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

# Set up logging
logger = logging.getLogger('llm_service_logger')

# Default configuration
DEFAULT_CONFIG = {
    'provider': 'openai',  # Default to OpenAI instead of local
    'model': None,
    'temperature': 0.7,
    'max_tokens': 150,
    'top_p': 0.9,
    'timeout': 60,
    'retries': 2,
    'retry_delay': 1,
    'fallback_enabled': True
}

LOCAL_MODELS = {}


class LLMService:
    """Service for interacting with various LLM providers with graceful fallbacks"""
    
    def __init__(self, config=None):
        """Initialize the LLM service with the given configuration."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.api_keys = {}
        self.provider = self.config['provider']
        
        # Try to load local model support - don't fail if not available
        self.has_local_support = False
        try:
            import torch
            import llama_cpp
            self.has_local_support = True
            logger.info("Local model support is available")
        except ImportError:
            logger.info("Local model support is not available - will use API providers only")
        
        # Load API keys from environment variables
        self._load_api_keys_from_env()
    
    def _load_api_keys_from_env(self):
        """Load API keys from environment variables."""
        providers = ['openai', 'anthropic', 'cohere', 'huggingface']
        for provider in providers:
            env_var = f"{provider.upper()}_API_KEY"
            if env_var in os.environ:
                self.api_keys[provider] = os.environ[env_var]
                logger.info(f"Loaded API key for {provider}")
    
    def complete(self, prompt: Union[str, List[Dict[str, str]]], options: Dict[str, Any] = None) -> str:
        """Generate a completion with automatic fallback handling."""
        options = options or {}
        provider = options.get('provider', self.provider)
        
        # Handle local model case
        if provider == 'local':
            if not self.has_local_support:
                logger.warning("Local model support not available - falling back to API provider")
                provider = 'openai'  # Fallback to OpenAI
                options['provider'] = provider
            else:
                return self.complete_with_local_model(prompt, options)
        
        try:
            return self.complete_with_api(prompt, options)
        except Exception as e:
            logger.error(f"Error with provider {provider}: {str(e)}")
            if self.config['fallback_enabled']:
                # Try alternative providers
                alternative_providers = ['openai', 'anthropic', 'cohere']
                for alt_provider in alternative_providers:
                    if alt_provider != provider and alt_provider in self.api_keys:
                        try:
                            logger.info(f"Attempting fallback to {alt_provider}")
                            options['provider'] = alt_provider
                            return self.complete_with_api(prompt, options)
                        except Exception as fallback_error:
                            logger.error(f"Fallback to {alt_provider} failed: {str(fallback_error)}")
                
                # If all API providers fail and local support is available, try local
                if self.has_local_support:
                    try:
                        logger.info("Attempting fallback to local model")
                        return self.complete_with_local_model(prompt, options)
                    except Exception as local_error:
                        logger.error(f"Fallback to local model failed: {str(local_error)}")
            
            # If all fallbacks fail or fallbacks are disabled
            raise Exception(f"Failed to generate completion with all available providers: {str(e)}")
    
    def complete_with_api(self, prompt: Union[str, List[Dict[str, str]]], options: Dict[str, Any]) -> str:
        """Generate a completion using an API provider."""
        provider = options.get('provider', self.provider)
        
        # Format the API request
        url = f"{self.get_base_url(provider)}"
        headers = self.create_headers(provider)
        data = self.format_request(prompt, options)
        
        # Make API request with retry logic
        response = self.make_api_request(url, data, headers)
        
        # Parse and return the response
        return self.parse_response(response, provider)
    
    def complete_with_local_model(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate a completion using a local model."""
        model_path = options.get('model_path', self.local_model_path)
        model_file = options.get('model_file', self.local_model_filename)
        
        if not model_file:
            # Find any .gguf file in the model path
            from pathlib import Path
            model_files = list(Path(model_path).glob("*.gguf"))
            if not model_files:
                raise ValueError(f"No GGUF model files found in {model_path}")
            model_file = str(model_files[0])
        else:
            model_file = os.path.join(model_path, model_file)
        
        # Check if model exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            import llama_cpp
            import torch
            
            # Use cached model or create a new one
            model_key = model_file
            if model_key not in LOCAL_MODELS:
                logger.info(f"Loading local model: {model_file}")
                
                # Settings for llama-cpp-python
                model_kwargs = {
                    'model_path': model_file,
                    'n_ctx': 2048,
                    'n_batch': 512,
                    'n_gpu_layers': 0  # CPU only by default
                }
                
                # Use GPU if requested and available
                if options.get('use_gpu', False) and torch.cuda.is_available():
                    model_kwargs['n_gpu_layers'] = -1  # Use all layers on GPU
                    logger.info("Using GPU for local model inference")
                
                LOCAL_MODELS[model_key] = llama_cpp.Llama(**model_kwargs)
            
            model = LOCAL_MODELS[model_key]
            
            # Extract parameters from options
            max_tokens = options.get('max_tokens', 150)
            temperature = options.get('temperature', 0.7)
            top_p = options.get('top_p', 0.9)
            stop = options.get('stop', [])
            
            # Format the prompt properly
            request_data = self.format_request(prompt, options)
            prompt_text = request_data['prompt']
            
            logger.info(f"Generating completion with local model, prompt length: {len(prompt_text)}")
            
            result = model(
                prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop
            )
            
            # Extract and return the generated text
            return result['choices'][0]['text']
            
        except Exception as e:
            logger.error(f"Error with local model: {str(e)}")
            raise
    
    # Helper methods for API interaction
    def get_base_url(self, provider: str) -> str:
        """Get the base URL for the provider's API."""
        urls = {
            'openai': 'https://api.openai.com/v1/chat/completions',
            'anthropic': 'https://api.anthropic.com/v1/complete',
            'cohere': 'https://api.cohere.ai/v1/generate',
            'huggingface': 'https://api-inference.huggingface.co/models'
        }
        return urls.get(provider, '')
    
    def create_headers(self, provider: str) -> Dict[str, str]:
        """Create headers for API requests."""
        api_key = self.api_keys.get(provider)
        if not api_key:
            raise ValueError(f"No API key found for provider: {provider}")
            
        headers = {
            'Content-Type': 'application/json'
        }
        
        if provider == 'openai':
            headers['Authorization'] = f"Bearer {api_key}"
        elif provider == 'anthropic':
            headers['x-api-key'] = api_key
        elif provider == 'cohere':
            headers['Authorization'] = f"Bearer {api_key}"
        elif provider == 'huggingface':
            headers['Authorization'] = f"Bearer {api_key}"
            
        return headers
    
    def format_request(self, prompt: Union[str, List[Dict[str, str]]], options: Dict[str, Any]) -> Dict[str, Any]:
        """Format the request based on the provider's requirements."""
        provider = options.get('provider', self.provider)
        
        if provider == 'openai':
            messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
            return {
                'model': options.get('model', 'gpt-3.5-turbo'),
                'messages': messages,
                'temperature': options.get('temperature', 0.7),
                'max_tokens': options.get('max_tokens', 150)
            }
        # Add formatting for other providers as needed
        return {}
    
    def parse_response(self, response: Dict[str, Any], provider: str) -> str:
        """Parse the response from the provider."""
        try:
            if provider == 'openai':
                return response['choices'][0]['message']['content'].strip()
            # Add parsing for other providers as needed
            return str(response)
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise

# Create a singleton instance
llm_service = LLMService()