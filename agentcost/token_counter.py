"""
AgentCost Token Counter

Handles token counting for different LLM providers using tiktoken.
"""

import logging
import time
import tiktoken
from typing import Dict, List, Any, Tuple

# Logger for token counter warnings
logger = logging.getLogger("agentcost.token_counter")


class TokenCounter:
    """Handles token counting for different LLM providers"""
    
    # Track if we've warned about estimation for this model (capped)
    _estimation_warnings_shown: set = set()
    _MAX_WARNINGS_CACHE = 500
    
    MODEL_ENCODINGS: Dict[str, str] = {
        # OpenAI models
        'gpt-4': 'cl100k_base',
        'gpt-4-turbo': 'cl100k_base',
        'gpt-4-turbo-preview': 'cl100k_base',
        # The gpt-4o family tokenizes with o200k_base, not cl100k_base. This
        # exact-match table is consulted before the family-prefix rules in
        # _get_encoding_name, so leaving these wrong silently overrode the
        # o200k branch for the two most common model strings a caller passes.
        'gpt-4o': 'o200k_base',
        'gpt-4o-mini': 'o200k_base',
        'gpt-3.5-turbo': 'cl100k_base',
        'gpt-3.5-turbo-16k': 'cl100k_base',
        'text-davinci-003': 'p50k_base',
        
    }
    
    _encoding_cache: Dict[str, Any] = {}

    # Encoding name -> monotonic deadline before which we will not retry its
    # load. tiktoken fetches the BPE file over HTTP with no timeout, from the
    # event loop, so an unreachable CDN otherwise costs that fetch on every
    # single call. Time-boxed rather than permanent, so one transient failure
    # does not leave the process estimating characters for its whole lifetime.
    _load_failed_until: Dict[str, float] = {}
    _ENCODING_RETRY_SECONDS = 300.0


    @classmethod
    def count_tokens(cls, text: str, model: str) -> int:
        """
        Count tokens in text for given model
        
        Args:
            text: Input text to count
            model: Model name (e.g., 'gpt-4', 'llama-3.1-8b-instant')
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        try:
            encoding = cls._get_encoding(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Log warning about estimation fallback (once per model)
            if model not in cls._estimation_warnings_shown:
                logger.warning(
                    f"Token counting for '{model}' failed, using character-based estimation. "
                    f"Counts may be inaccurate. Error: {e}"
                )
                if len(cls._estimation_warnings_shown) < cls._MAX_WARNINGS_CACHE:
                    cls._estimation_warnings_shown.add(model)
            return cls._estimate_tokens(text)
    
    @classmethod
    def count_tokens_with_accuracy(
        cls, text: str, model: str
    ) -> Tuple[int, bool]:
        """
        Count tokens with accuracy indicator.
        
        Args:
            text: Input text to count
            model: Model name
        
        Returns:
            Tuple of (token_count, is_exact)
            is_exact is False when using character-based estimation
        """
        if not text:
            return (0, True)
        
        try:
            encoding = cls._get_encoding(model)
            tokens = encoding.encode(text)
            return (len(tokens), True)
        except Exception:
            return (cls._estimate_tokens(text), False)
    
    @classmethod
    def _get_encoding(cls, model: str):
        """Get or create encoding for model (with caching)"""

        cache_key = cls._get_encoding_name(model)
        cached = cls._encoding_cache.get(cache_key)
        if cached is not None:
            return cached

        if cls._load_blocked(cache_key):
            return cls._fallback_encoding()

        try:
            encoding = tiktoken.get_encoding(cache_key)
            cls._encoding_cache[cache_key] = encoding
            cls._load_failed_until.pop(cache_key, None)
            return encoding
        except Exception:
            cls._load_failed_until[cache_key] = time.monotonic() + cls._ENCODING_RETRY_SECONDS
            return cls._fallback_encoding()

    @classmethod
    def _load_blocked(cls, name: str) -> bool:
        """True while a recently failed encoding is in its cool-off window."""
        until = cls._load_failed_until.get(name)
        return until is not None and time.monotonic() < until

    @classmethod
    def _fallback_encoding(cls):
        """cl100k_base, negative-cached like any other encoding.

        Raises if it is also unavailable; count_tokens then falls back to
        character estimation.
        """
        cached = cls._encoding_cache.get('cl100k_base')
        if cached is not None:
            return cached
        if cls._load_blocked('cl100k_base'):
            raise RuntimeError("tiktoken encodings are temporarily unavailable")
        try:
            encoding = tiktoken.get_encoding('cl100k_base')
            cls._encoding_cache['cl100k_base'] = encoding
            return encoding
        except Exception:
            cls._load_failed_until['cl100k_base'] = time.monotonic() + cls._ENCODING_RETRY_SECONDS
            raise
    
    @classmethod
    def _get_encoding_name(cls, model: str) -> str:
        """Determine which encoding to use for a model"""
        if model in cls.MODEL_ENCODINGS:
            return cls.MODEL_ENCODINGS[model]
        
        model_lower = model.lower()

        # o200k_base covers the current OpenAI families. Matched explicitly
        # because gpt-4o and gpt-4.1 would otherwise reach the cl100k_base
        # fallback below and be counted with the older, wrong encoding.
        if any(
            token in model_lower
            for token in ('gpt-4o', 'gpt-4.1', 'gpt-5', 'o1', 'o3', 'o4-mini', 'chatgpt-4o')
        ):
            return 'o200k_base'

        # Anthropic, Google and open-weight models do not publish a tiktoken
        # encoding. cl100k_base is an approximation used only when the provider
        # itself reports no usage — every interceptor prefers the provider's
        # own counts, so this should rarely be load-bearing.
        return 'cl100k_base'
    
    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        """
        Rough estimation when tokenizer fails.
        
        Uses different ratios for different character sets:
        - English/Latin: 1 token ≈ 4 characters
        - CJK/Unicode: 1 token ≈ 1.5 characters (these tokenize less efficiently)
        """
        if not text:
            return 0
        
        # Check for high proportion of non-ASCII characters (CJK, etc)
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / len(text) if text else 0
        
        if non_ascii_ratio > 0.3:
            # Multilingual text - use more conservative ratio
            return max(1, int(len(text) / 2))
        else:
            # Primarily English - standard ratio
            return max(1, len(text) // 4)
    
    @classmethod
    def count_message_tokens(cls, messages: List[Dict[str, str]], model: str) -> int:
        """
        Count tokens for chat completion messages.
        Accounts for special tokens added by the API.
        
        Example messages format:
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        Args:
            messages: List of message dictionaries
            model: Model name
        
        Returns:
            Total token count including overhead
        """
        model_lower = model.lower()
        
        if 'gpt-4' in model_lower:
            tokens_per_message = 3
            tokens_per_name = 1
        elif 'gpt-3.5' in model_lower:
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1
        
        total_tokens = 0
        
        for message in messages:
            total_tokens += tokens_per_message
            
            for key, value in message.items():
                if value:
                    total_tokens += cls.count_tokens(str(value), model)
                
                if key == "name":
                    total_tokens += tokens_per_name
        
        total_tokens += 3
        
        return total_tokens
    
    @classmethod
    def extract_text_from_input(cls, input_data: Any) -> str:
        """
        Extract text content from various input formats.
        Handles LangChain message types, strings, lists, etc.
        
        Args:
            input_data: Input in various formats
        
        Returns:
            Extracted text as string
        """
        if isinstance(input_data, str):
            return input_data
        
        if isinstance(input_data, list):
            texts = []
            for item in input_data:
                if hasattr(item, 'content'):
                    texts.append(str(item.content))
                elif isinstance(item, dict) and 'content' in item:
                    texts.append(str(item['content']))
                else:
                    texts.append(str(item))
            return " ".join(texts)
        
        if hasattr(input_data, 'content'):
            return str(input_data.content)
        
        if hasattr(input_data, 'text'):
            return str(input_data.text)
        
        return str(input_data)
    
    @classmethod
    def extract_text_from_output(cls, output_data: Any) -> str:
        """
        Extract text content from LLM response.
        
        Args:
            output_data: LLM response in various formats
        
        Returns:
            Extracted text as string
        """
        if isinstance(output_data, str):
            return output_data
        
        if hasattr(output_data, 'content'):
            return str(output_data.content)
        
        if hasattr(output_data, 'text'):
            return str(output_data.text)
        
        if isinstance(output_data, dict):
            for key in ['content', 'text', 'output', 'response']:
                if key in output_data:
                    return str(output_data[key])
        
        return str(output_data)
