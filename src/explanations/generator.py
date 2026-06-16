import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generates natural-language explanations for neurons and attention heads.

    Supports multiple explanation providers:
    - google/gemini (default, free tier)
    - anthropic/claude (alternative, requires API key)

    The provider is configured via the EXPLANATION_PROVIDER env var or config.
    If no API key is available for the selected provider, explanations are
    marked as "unavailable" and the core activation analysis still works.
    """

    def __init__(
        self,
        provider: str = "gemini",
        gemini_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        claude_model: str = "claude-3-haiku-20240307",
        groq_model: str = "llama-3.1-8b-instant",
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        self.provider = provider.lower()
        self.gemini_api_key = gemini_api_key
        self.claude_api_key = claude_api_key
        self.groq_api_key = groq_api_key
        self.gemini_model = gemini_model
        self.claude_model = claude_model
        self.groq_model = groq_model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self._gemini_client = None
        self._claude_client = None
        self._initialised = False

    def _init_gemini(self):
        """Initialise the Gemini API client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(self.gemini_model)
            self._initialised = True
            logger.info("Initialised Gemini provider with model %s", self.gemini_model)
        except Exception as e:
            logger.warning("Failed to initialise Gemini: %s", e)
            self._gemini_client = None

    def _init_claude(self):
        """Initialise the Anthropic Claude API client."""
        try:
            import anthropic
            self._claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
            self._initialised = True
            logger.info("Initialised Claude provider with model %s", self.claude_model)
        except Exception as e:
            logger.warning("Failed to initialise Claude: %s", e)
            self._claude_client = None

    def is_available(self) -> bool:
        """Check if the configured provider is available (has an API key)."""
        if self.provider == "gemini":
            return bool(self.gemini_api_key)
        elif self.provider == "claude":
            return bool(self.claude_api_key)
        elif self.provider == "groq":
            return bool(self.groq_api_key)
        return False

    def generate(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Generate an explanation from a list of message dicts.

        Each message dict has "role" (system/user/assistant) and "content" keys.
        Returns the response text, or None if the provider is unavailable.
        """
        if not self.is_available():
            logger.warning("Explanation provider '%s' is not available (no API key)", self.provider)
            return None

        for attempt in range(self.max_retries):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(messages)
                elif self.provider == "claude":
                    return self._call_claude(messages)
                elif self.provider == "groq":
                    return self._call_groq(messages)
                else:
                    logger.error("Unknown explanation provider: %s", self.provider)
                    return None
            except Exception as e:
                logger.warning(
                    "Provider call failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("All %d retries exhausted", self.max_retries)
                    return None

    def _call_gemini(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Call the Gemini API and return the response text."""
        if self._gemini_client is None:
            self._init_gemini()
        if self._gemini_client is None:
            return None

        system_content = None
        user_contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_contents.append(msg["content"])

        combined_user = "\n".join(user_contents) if user_contents else ""

        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "max_output_tokens": 512,
        }

        contents = [combined_user]
        if system_content:
            response = self._gemini_client.generate_content(
                contents,
                generation_config=generation_config,
            )
        else:
            response = self._gemini_client.generate_content(
                contents,
                generation_config=generation_config,
            )

        if response and hasattr(response, "text"):
            return response.text
        return None

    def _call_claude(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Call the Anthropic Claude API and return the response text."""
        if self._claude_client is None:
            self._init_claude()
        if self._claude_client is None:
            return None

        system_content = None
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        kwargs = {
            "model": self.claude_model,
            "max_tokens": 512,
            "temperature": 0.3,
            "messages": api_messages,
        }
        if system_content:
            kwargs["system"] = system_content

        response = self._claude_client.messages.create(**kwargs)

        if response and response.content:
            return response.content[0].text
        return None


    def _call_groq(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Call the Groq API (OpenAI-compatible chat completions) via httpx."""
        import httpx

        api_messages = []
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": self.groq_model,
            "messages": api_messages,
            "temperature": 0.3,
            "max_tokens": 512,
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content")
        return None


class NullExplanationGenerator(ExplanationGenerator):
    """A fallback generator that returns None for all requests.

    Used when no API provider is configured, so the analysis pipeline can
    run without explanations.
    """

    def __init__(self):
        super().__init__()
        self.provider = "none"

    def is_available(self) -> bool:
        return False

    def generate(self, messages: list[dict[str, str]]) -> None:
        return None
