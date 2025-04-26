# File: token_observer.py

import threading
import json
import logging

logger = logging.getLogger(__name__)

class LLMTokenObserver:
    """Observes and accumulates token usage for different LLM APIs."""
    def __init__(self):
        self._lock = threading.Lock()
        self._token_counts = self._initialize_counts()
        logger.info("LLMTokenObserver singleton instance initialized.")

    def _initialize_counts(self):
        """Initializes the token count structure."""
        return {
            "openai": {"prompt": 0, "completion": 0, "total": 0},
            "bedrock": {"prompt": 0, "completion": 0, "total": 0},
            "google": {"prompt": 0, "completion": 0, "total": 0},
            "local": {"prompt": 0, "completion": 0, "total": 0},
            "embeddings": {"prompt": 0, "completion": 0, "total": 0}
        }

    def update_tokens(self, api_type: str, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        """Updates the token count for a specific API."""
        prompt_tokens = prompt_tokens or 0
        completion_tokens = completion_tokens or 0
        total_tokens = total_tokens or (prompt_tokens + completion_tokens)

        with self._lock:
            if api_type not in self._token_counts:
                logger.warning(f"Unknown API type '{api_type}' encountered. Initializing.")
                self._token_counts[api_type] = {"prompt": 0, "completion": 0, "total": 0}

            self._token_counts[api_type]["prompt"] += prompt_tokens
            self._token_counts[api_type]["completion"] += completion_tokens
            self._token_counts[api_type]["total"] += total_tokens
            # logger.debug(f"Tokens updated for {api_type}: P={prompt_tokens}, C={completion_tokens}, T={total_tokens}. Running Total: {self._token_counts[api_type]}")

    def get_total_tokens(self) -> dict:
        """Returns a copy of the current token counts for all APIs."""
        with self._lock:
            return json.loads(json.dumps(self._token_counts))

    def reset(self):
        """Resets all token counts back to zero."""
        with self._lock:
            self._token_counts = self._initialize_counts()
            logger.info("LLMTokenObserver counts reset.")

# --- Instantiate the "Global" Shared Instance ---
# This line is executed only ONCE when the module is first imported.
llm_token_observer = LLMTokenObserver()