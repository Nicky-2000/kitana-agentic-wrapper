# File: token_observer.py

import threading
import json
import logging

from numpy import copy

logger = logging.getLogger(__name__)

class LLMTokenObserver:
    """
    Observes and accumulates token usage for different LLM APIs
    and maintains an overall total count across all APIs.
    Designed to be used as a singleton via the module-level instance.
    """
    def __init__(self):
        self._lock = threading.Lock()
        # Initialize per-API counts and overall counts
        self._token_counts = self._initialize_api_counts()
        self._overall_totals = {"prompt": 0, "completion": 0, "total": 0}
        logger.info("LLMTokenObserver singleton instance initialized.")

    def _initialize_api_counts(self) -> dict:
        """Initializes the token count structure for individual APIs."""
        return {
            "openai": {"prompt": 0, "completion": 0, "total": 0},
            "bedrock": {"prompt": 0, "completion": 0, "total": 0},
            "google": {"prompt": 0, "completion": 0, "total": 0},
            "local": {"prompt": 0, "completion": 0, "total": 0},
            "embeddings": {"prompt": 0, "completion": 0, "total": 0}
            # Add other potential API types here if known beforehand
        }

    def update_tokens(self, api_type: str, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        """
        Updates the token count for a specific API and the overall total.
        Ensures thread safety using a lock.
        """
        # Ensure token counts are non-negative integers
        prompt_tokens = max(0, prompt_tokens or 0)
        completion_tokens = max(0, completion_tokens or 0)

        # Calculate total_tokens if not provided or if it's inconsistent
        calculated_total = prompt_tokens + completion_tokens
        # Use provided total_tokens only if it's >= calculated_total, otherwise use calculated
        # (Some APIs might report slightly different totals, e.g., including internal tokens)
        # If total_tokens is explicitly 0 or None, recalculate.
        if total_tokens is None or total_tokens < calculated_total:
            total_tokens = calculated_total
        else:
             total_tokens = max(0, total_tokens or 0) # Ensure non-negative

        with self._lock:
            # Initialize API type if not seen before
            if api_type not in self._token_counts:
                logger.warning(f"Unknown API type '{api_type}' encountered. Initializing.")
                self._token_counts[api_type] = {"prompt": 0, "completion": 0, "total": 0}

            # Update per-API counts
            self._token_counts[api_type]["prompt"] += prompt_tokens
            self._token_counts[api_type]["completion"] += completion_tokens
            self._token_counts[api_type]["total"] += total_tokens

            # Update overall counts
            self._overall_totals["prompt"] += prompt_tokens
            self._overall_totals["completion"] += completion_tokens
            self._overall_totals["total"] += total_tokens

            # Optional: Debug logging
            # logger.debug(f"Tokens updated for {api_type}: P={prompt_tokens}, C={completion_tokens}, T={total_tokens}.")
            # logger.debug(f"Running API Total ({api_type}): {self._token_counts[api_type]}")
            # logger.debug(f"Running Overall Total: {self._overall_totals}")

    def get_api_token_counts(self) -> dict:
        """
        Returns a deep copy of the current token counts for each individual API.
        """
        with self._lock:
            return json.loads(json.dumps(self._token_counts))

    def get_overall_totals(self) -> dict:
        """
        Returns a copy of the overall total token counts across all APIs.
        """
        with self._lock:
            # Return a copy to prevent external modification
            return self._overall_totals.copy()

    def reset(self):
        """Resets all per-API and overall token counts back to zero."""
        with self._lock:
            self._token_counts = self._initialize_api_counts()
            self._overall_totals = {"prompt": 0, "completion": 0, "total": 0}
            logger.info("LLMTokenObserver counts (per-API and overall) reset.")

# --- Instantiate the "Global" Shared Instance ---
# This line is executed only ONCE when the module is first imported.
llm_token_observer = LLMTokenObserver()