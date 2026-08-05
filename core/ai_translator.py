"""
AI Translation service — Generic interface ready for Gemma 4 (or any open-source / hosted model).

Architecture
------------
BaseTranslator   : Abstract contract that any provider must implement.
GenericAITranslator : Placeholder implementation via HTTP (OpenAI-compatible API format).
                  Point TRANSLATOR_BASE_URL at your Gemma 4 host (Ollama, vLLM, LM Studio…)
                  and it will work without changing this code.
NullTranslator   : No-op implementation used when TRANSLATOR_PROVIDER=none.
                  Falls back to Whisper's own translation.

Environment variables
---------------------
TRANSLATOR_PROVIDER   : "none" (default) | "generic"
TRANSLATOR_BASE_URL   : Base URL of the OpenAI-compatible API
                        e.g. http://localhost:11434/v1  (Ollama)
                             http://localhost:8000/v1   (vLLM)
TRANSLATOR_API_KEY    : API key (use "ollama" or any string for local servers that don't check)
TRANSLATOR_MODEL      : Model name as the API expects it (e.g. "gemma4", "gemma2:27b")
TRANSLATOR_MAX_TOKENS : Max tokens per translated segment (default: 256)

Usage
-----
from core.ai_translator import get_translator_service

translator = get_translator_service()

# Check if AI translation is enabled
if translator.is_enabled:
    async for token in translator.translate_stream(text, source_lang, target_lang):
        # stream each token back to the OBS plugin
        ...
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# ─── System prompt template ───────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a professional real-time speech translator. "
    "Translate the following text from {source_lang} to {target_lang}. "
    "Rules:\n"
    "- Output ONLY the translated text, nothing else.\n"
    "- Preserve tone, style, and punctuation.\n"
    "- Do NOT add explanations, prefixes, or quotation marks.\n"
    "- If the text is already in {target_lang}, return it unchanged."
)


# ─── Abstract base ────────────────────────────────────────────────────────────

class BaseTranslator(ABC):
    """
    Contract that every AI translation provider must implement.
    All implementations must support async streaming output.
    """

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Returns True if this translator is active (not a no-op)."""

    @abstractmethod
    async def translate_stream(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[str]:
        """
        Translate *text* and yield the result token by token.

        :param text:        Text to translate (one Whisper segment).
        :param source_lang: ISO 639-1 source language code (e.g. "es").
        :param target_lang: ISO 639-1 target language code (e.g. "en").
        :yields:            Successive string tokens of the translation.
        """


# ─── No-op implementation ────────────────────────────────────────────────────

class NullTranslator(BaseTranslator):
    """
    No-op translator. Used when TRANSLATOR_PROVIDER=none.
    Whisper handles translation natively in this case.
    """

    @property
    def is_enabled(self) -> bool:
        return False

    async def translate_stream(
        self, text: str, source_lang: str, target_lang: str
    ) -> AsyncIterator[str]:
        # Yield the text as-is (Whisper already translated it)
        yield text


# ─── Generic OpenAI-compatible implementation ─────────────────────────────────

class GenericAITranslator(BaseTranslator):
    """
    Streaming translator that works with any OpenAI-compatible HTTP API.

    Compatible hosts:
      - Ollama  (http://localhost:11434/v1)
      - vLLM    (http://localhost:8000/v1)
      - LM Studio (http://localhost:1234/v1)
      - Any hosted endpoint that mirrors the OpenAI /chat/completions API

    When you decide where to host Gemma 4, set TRANSLATOR_BASE_URL to that
    server's base URL and TRANSLATOR_MODEL to the model identifier.
    """

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 256):
        try:
            from openai import AsyncOpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GenericAITranslator. "
                "Install it with: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self._model = model
        self._max_tokens = max_tokens

        logger.info(
            f"[AITranslator] Initialized GenericAITranslator "
            f"(base_url={base_url}, model={model})"
        )

    @property
    def is_enabled(self) -> bool:
        return True

    async def translate_stream(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> AsyncIterator[str]:
        """
        Stream the translation of *text* token by token.
        Uses the OpenAI-compatible /chat/completions endpoint with stream=True.
        """
        extra_kwargs = {}
        if "translategemma" in self._model.lower():
            # El modelo translategemma en vLLM (y HuggingFace) exige un formato estructurado en el content
            # OJO: La librería OpenAI Python descarta (strip) campos desconocidos como source_lang_code.
            # Para evitarlo, usamos 'extra_body' que inyecta el JSON tal cual en la petición HTTP.
            actual_messages = [{
                "role": "user", 
                "content": [{
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text
                }]
            }]
            messages = [{"role": "user", "content": text}]  # Dummy para pasar la validación local de pydantic
            extra_kwargs["extra_body"] = {"messages": actual_messages}
        else:
            # Modelos genéricos usan system prompt
            system_msg = _SYSTEM_PROMPT.format(
                source_lang=source_lang, target_lang=target_lang
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ]

        logger.info(
            f"[AITranslator] Requesting translation via {self._client.base_url} (model={self._model})"
        )

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                stream=True,
                temperature=0.1,  # Low temperature for consistent, accurate translations
                **extra_kwargs
            )
        except Exception as err:
            cause = f" Cause: {err.__cause__}" if getattr(err, '__cause__', None) else ""
            logger.error(f"[AITranslator] Connection error to {self._client.base_url}: {err}{cause}")
            raise

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─── Factory ─────────────────────────────────────────────────────────────────

_translator_service: BaseTranslator | None = None


def get_translator_service() -> BaseTranslator:
    """
    Return the global translator instance (singleton).
    Reads configuration from environment variables.

    TRANSLATOR_PROVIDER controls which implementation is used:
      - "none"    → NullTranslator (Whisper handles translation, default)
      - "generic" → GenericAITranslator (OpenAI-compatible API)
    """
    global _translator_service

    if _translator_service is not None:
        return _translator_service

    provider = os.getenv("TRANSLATOR_PROVIDER", "none").strip().lower()

    if provider == "none" or not provider:
        logger.info("[AITranslator] Provider=none — AI translation disabled. Whisper will translate.")
        _translator_service = NullTranslator()

    elif provider == "generic":
        base_url = os.getenv("TRANSLATOR_BASE_URL", "http://localhost:11434/v1")
        api_key = os.getenv("TRANSLATOR_API_KEY", "ollama")
        model = os.getenv("TRANSLATOR_MODEL", "gemma4")
        max_tokens = int(os.getenv("TRANSLATOR_MAX_TOKENS", "256"))

        _translator_service = GenericAITranslator(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
        )
    else:
        logger.warning(
            f"[AITranslator] Unknown TRANSLATOR_PROVIDER='{provider}'. "
            f"Falling back to NullTranslator."
        )
        _translator_service = NullTranslator()

    return _translator_service
