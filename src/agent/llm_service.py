"""
LLM Service module for handling various LLM providers.
Supports OpenAI, Anthropic, Cohere, HuggingFace, and local models.
"""

import os
import json
import logging
import time
import requests
from typing import Union, List, Dict, Any, Optional
import llama_cpp
import torch
from pathlib import Path

# Set up logging
logger = logging.getLogger('llm_service_logger')

# Default configuration
DEFAULT_CONFIG = {
    'provider': 'local',
    'model': None,
    'temperature': 0.7,
    'max_tokens': 150,
    'top_p': 0.9,
    'timeout': 60,  # seconds
    'retries': 2,
    'retry_delay': 1,  # seconds
    'fallback_to_local': True,
    'offline_allowed': False
}

# Local model cache
LOCAL_MODELS = {}

class LLMService:
    """Service for interacting with various LLM providers"""
    
    def __init__(self, config=None):
        """Initialize the LLM service with the given configuration."""
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.api_keys = {}
        self.provider = self.config['provider']
        self.local_model_path = os.environ.get('LOCAL_MODEL_PATH', '/content/local_model')
        self.local_model_filename = os.environ.get('LOCAL_MODEL_FILENAME')
        
        # Load API keys from environment variables if available
        self._load_api_keys_from_env()
    
    def _load_api_keys_from_env(self):
        """Load API keys from environment variables."""
        providers = ['openai', 'anthropic', 'cohere', 'huggingface']
        for provider in providers:
            env_var = f"{provider.upper()}_API_KEY"
            if env_var in os.environ:
                self.api_keys[provider] = os.environ[env_var]
                logger.info(f"Loaded API key for {provider} from environment variable")
    
    def set_provider(self, provider: str) -> 'LLMService':
        """Set the LLM provider."""
        supported_providers = ['openai', 'anthropic', 'cohere', 'huggingface', 'local']
        if provider not in supported_providers:
            logger.warning(f"Unsupported provider: {provider}. Falling back to local.")
            provider = 'local'
        
        self.provider = provider
        return self
    
    def set_api_key(self, provider: str, api_key: str) -> 'LLMService':
        """Set the API key for a provider."""
        self.api_keys[provider] = api_key
        return self
    
    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get the API key for the current or specified provider."""
        provider = provider or self.provider
        return self.api_keys.get(provider)
    
    def get_base_url(self, provider: Optional[str] = None) -> str:
        """Get the base URL for the provider's API."""
        provider = provider or self.provider
        
        if provider == 'openai':
            return 'https://api.openai.com/v1'
        elif provider == 'anthropic':
            return 'https://api.anthropic.com/v1'
        elif provider == 'cohere':
            return 'https://api.cohere.ai/v1'
        elif provider == 'huggingface':
            return 'https://api-inference.huggingface.co/models'
        elif provider == 'local':
            return 'http://localhost:8000'
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def create_headers(self, provider: Optional[str] = None) -> Dict[str, str]:
        """Create headers for API requests."""
        provider = provider or self.provider
        api_key = self.get_api_key(provider)
        
        if provider == 'local':
            return {'Content-Type': 'application/json'}
        
        if not api_key:
            raise ValueError(f"API key not set for provider: {provider}")
        
        if provider == 'openai':
            return {
                'Authorization': f"Bearer {api_key}",
                'Content-Type': 'application/json'
            }
        elif provider == 'anthropic':
            return {
                'x-api-key': api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        elif provider == 'cohere':
            return {
                'Authorization': f"Bearer {api_key}",
                'Content-Type': 'application/json'
            }
        elif provider == 'huggingface':
            return {
                'Authorization': f"Bearer {api_key}",
                'Content-Type': 'application/json'
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def format_request(self, prompt: Union[str, List[Dict[str, str]]], options: Dict[str, Any]) -> Dict[str, Any]:
        """Format the request for the specified provider."""
        provider = options.get('provider', self.provider)
        model = options.get('model')
        
        if provider == 'openai':
            # OpenAI requires a list of messages
            if isinstance(prompt, str):
                messages = [{'role': 'user', 'content': prompt}]
            else:
                messages = prompt
            
            return {
                'model': model or 'gpt-3.5-turbo',
                'messages': messages,
                'temperature': options.get('temperature', 0.7),
                'max_tokens': options.get('max_tokens', 150),
                'top_p': options.get('top_p', 1),
                'frequency_penalty': options.get('frequency_penalty', 0),
                'presence_penalty': options.get('presence_penalty', 0),
                'stream': False
            }
        
        elif provider == 'anthropic':
            # Format prompt for Anthropic's Claude models
            if isinstance(prompt, list):
                # Convert message list to Anthropic format
                formatted_prompt = ""
                for msg in prompt:
                    role = "Human" if msg['role'] == 'user' else "Assistant"
                    formatted_prompt += f"{role}: {msg['content']}\n\n"
                if not formatted_prompt.endswith("Assistant: "):
                    formatted_prompt += "Assistant: "
            else:
                # Simple string prompt
                formatted_prompt = f"Human: {prompt}\n\nAssistant: "
            
            return {
                'model': model or 'claude-2',
                'prompt': formatted_prompt,
                'max_tokens_to_sample': options.get('max_tokens', 150),
                'temperature': options.get('temperature', 0.7),
                'top_p': options.get('top_p', 1),
                'stop_sequences': options.get('stop_sequences', ["\n\nHuman:"]),
                'stream': False
            }
        
        elif provider == 'cohere':
            # Format prompt for Cohere
            if isinstance(prompt, list):
                # Convert message list to Cohere format
                formatted_prompt = ""
                for msg in prompt:
                    prefix = "User: " if msg['role'] == 'user' else "Chatbot: "
                    formatted_prompt += f"{prefix}{msg['content']}\n"
            else:
                formatted_prompt = prompt
            
            return {
                'model': model or 'command',
                'prompt': formatted_prompt,
                'max_tokens': options.get('max_tokens', 150),
                'temperature': options.get('temperature', 0.7),
                'p': options.get('top_p', 1),
                'k': options.get('top_k', 0),
                'stop_sequences': options.get('stop_sequences', []),
                'return_likelihoods': 'NONE'
            }
        
        elif provider == 'huggingface':
            # Format prompt for HuggingFace
            if isinstance(prompt, list):
                # Convert message list to string format
                formatted_prompt = ""
                for msg in prompt:
                    prefix = "Human: " if msg['role'] == 'user' else "Assistant: "
                    formatted_prompt += f"{prefix}{msg['content']}\n\n"
            else:
                formatted_prompt = prompt
            
            return {
                'inputs': formatted_prompt,
                'parameters': {
                    'max_new_tokens': options.get('max_tokens', 150),
                    'temperature': options.get('temperature', 0.7),
                    'top_p': options.get('top_p', 1),
                    'top_k': options.get('top_k', 50),
                    'repetition_penalty': options.get('repetition_penalty', 1.0),
                    'do_sample': options.get('do_sample', True)
                }
            }
        
        elif provider == 'local':
            # Format prompt for local model
            if isinstance(prompt, list):
                # Convert message list to string format
                formatted_prompt = ""
                for msg in prompt:
                    prefix = "User: " if msg['role'] == 'user' else "Assistant: "
                    formatted_prompt += f"{prefix}{msg['content']}\n\n"
                formatted_prompt += "Assistant: "
            else:
                if not prompt.endswith("Assistant: "):
                    formatted_prompt = f"{prompt}\n\nAssistant: "
                else:
                    formatted_prompt = prompt
            
            return {
                'prompt': formatted_prompt,
                'max_tokens': options.get('max_tokens', 150),
                'temperature': options.get('temperature', 0.7),
                'top_p': options.get('top_p', 1),
                'stop': options.get('stop', ["\n\nUser:", "\n\nHuman:"])
            }
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def parse_response(self, response, provider: Optional[str] = None) -> str:
        """Parse the response from the provider."""
        provider = provider or self.provider
        
        try:
            if provider == 'openai':
                return response['choices'][0]['message']['content'].strip()
            elif provider == 'anthropic':
                return response['completion'].strip()
            elif provider == 'cohere':
                return response['generations'][0]['text'].strip()
            elif provider == 'huggingface':
                if isinstance(response, list) and response:
                    return response[0]['generated_text'].strip()
                else:
                    return response['generated_text'].strip()
            elif provider == 'local':
                return response['response'].strip()
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response from {provider}: {e}")
            logger.debug(f"Response: {response}")
            return "I apologize, but I encountered an error processing your request."
    
    def make_api_request(self, url: str, data: Dict[str, Any], headers: Dict[str, str], 
                         retries: int = None, timeout: int = None) -> Dict[str, Any]:
        """Make an API request with retry logic."""
        retries = retries if retries is not None else self.config['retries']
        timeout = timeout if timeout is not None else self.config['timeout']
        
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries + 1}): {e}")
                if attempt < retries:
                    time.sleep(self.config['retry_delay'])
                else:
                    raise
    
    def complete_with_local_model(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate a completion using a local model."""
        model_path = options.get('model_path', self.local_model_path)
        model_file = options.get('model_file', self.local_model_filename)
        
        if not model_file:
            # Find any .gguf file in the model path
            model_files = list(Path(model_path).glob("*.gguf"))
            if not model_files:
                raise ValueError(f"No GGUF model files found in {model_path}")
            model_file = str(model_files[0])
        else:
            model_file = os.path.join(model_path, model_file)
        
        # Check if model exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
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
        
        # Generate completion
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
        generated_text = result['choices'][0]['text']
        return generated_text
    
    def complete(self, prompt: Union[str, List[Dict[str, str]]], options: Dict[str, Any] = None) -> str:
        """Generate a completion based on the prompt."""
        options = options or {}
        provider = options.get('provider', self.provider)
        fallback_to_local = options.get('fallback_to_local', self.config['fallback_to_local'])
        offline_allowed = options.get('offline_allowed', self.config['offline_allowed'])
        
        # Check for offline mode
        if options.get('offline', False):
            if offline_allowed:
                logger.info("Using offline mode - returning simulated response")
                return "This is a simulated response generated in offline mode."
            else:
                logger.warning("Offline mode requested but not allowed")
        
        try:
            if provider == 'local':
                return self.complete_with_local_model(prompt, options)
            
            # Prepare API request
            url = f"{self.get_base_url(provider)}"
            if provider == 'openai':
                url += "/chat/completions"
            elif provider == 'anthropic':
                url += "/complete"
            elif provider == 'cohere':
                url += "/generate"
            elif provider == 'huggingface':
                model = options.get('model', 'gpt2')
                url += f"/{model}"
            
            headers = self.create_headers(provider)
            data = self.format_request(prompt, options)
            
            # Make API request
            logger.info(f"Making request to {provider} API")
            response = self.make_api_request(url, data, headers, 
                                           options.get('retries', self.config['retries']),
                                           options.get('timeout', self.config['timeout']))
            
            # Parse response
            return self.parse_response(response, provider)
        
        except Exception as e:
            logger.error(f"Error completing with {provider}: {str(e)}")
            
            # Try fallback to local model if enabled
            if fallback_to_local and provider != 'local':
                logger.info(f"Falling back to local model due to error: {str(e)}")
                try:
                    options['provider'] = 'local'
                    return self.complete(prompt, options)
                except Exception as fallback_error:
                    logger.error(f"Fallback to local model failed: {str(fallback_error)}")
            
            # If offline mode is allowed, return simulated response
            if offline_allowed:
                logger.info("Falling back to offline mode")
                return "I apologize, but I encountered an error processing your request. This is a simulated response."
            
            # Otherwise, re-raise the error
            raise
    
    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, str]]:
        """List available models for the specified provider."""
        provider = provider or self.provider
        
        if provider == 'openai':
            # OpenAI models
            try:
                url = f"{self.get_base_url(provider)}/models"
                headers = self.create_headers(provider)
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Filter and format models
                models = []
                for model in data['data']:
                    if 'gpt' in model['id']:
                        models.append({
                            'id': model['id'],
                            'name': model['id']
                        })
                return models
            except Exception as e:
                logger.error(f"Error listing OpenAI models: {str(e)}")
                return []
        
        elif provider == 'anthropic':
            # Anthropic models (hardcoded as they don't have a list endpoint)
            return [
                {'id': 'claude-2', 'name': 'Claude 2'},
                {'id': 'claude-instant-1', 'name': 'Claude Instant 1'}
            ]
        
        elif provider == 'cohere':
            # Cohere models (hardcoded as they don't have a public list endpoint)
            return [
                {'id': 'command', 'name': 'Command'},
                {'id': 'command-light', 'name': 'Command Light'},
                {'id': 'generate', 'name': 'Generate'}
            ]
        
        elif provider == 'huggingface':
            # HuggingFace models (just a sample of popular models)
            return [
                {'id': 'gpt2', 'name': 'GPT-2'},
                {'id': 'distilgpt2', 'name': 'DistilGPT-2'},
                {'id': 'facebook/opt-350m', 'name': 'OPT 350M'},
                {'id': 'facebook/opt-1.3b', 'name': 'OPT 1.3B'},
                {'id': 'EleutherAI/gpt-neo-1.3B', 'name': 'GPT-Neo 1.3B'}
            ]
        
        elif provider == 'local':
            # List local models in the model path
            models = []
            try:
                model_files = list(Path(self.local_model_path).glob("*.gguf"))
                for model_file in model_files:
                    model_id = model_file.name
                    model_name = model_file.stem.replace('-', ' ').title()
                    models.append({
                        'id': model_id,
                        'name': model_name
                    })
                return models
            except Exception as e:
                logger.error(f"Error listing local models: {str(e)}")
                return []
        
        else:
            logger.warning(f"Unknown provider: {provider}")
            return []

# Create a singleton instance
llm_service = LLMService()