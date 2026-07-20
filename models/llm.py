"""
LLM provider singletons for the Editorial Intelligence System.

Two clients are exposed so every layer of the app uses a shared, lazily-
initialized connection rather than creating its own:

  llm            — LangChain ChatOpenAI
                   Used by agents that rely on LangChain / LangGraph LCEL chains.

  openai_client  — Raw OpenAI SDK client
                   Used by services that call the API directly for tighter
                   control over prompts and response parsing.

Import pattern
--------------
    from models.llm import llm              # LangChain agent
    from models.llm import openai_client    # direct SDK call
    from models.llm import LLMProvider      # one-shot helper
"""

import logging
from typing import Optional

from openai import OpenAI
from langchain_openai import ChatOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    """Lazy singleton factory for OpenAI client flavours."""

    _chat_llm: ChatOpenAI | None = None
    _openai: OpenAI | None = None

    # ------------------------------------------------------------------
    # LangChain ChatOpenAI
    # ------------------------------------------------------------------

    @classmethod
    def get_chat_llm(cls) -> ChatOpenAI:
        """
        Return the shared LangChain ChatOpenAI instance.
        Compatible with LangGraph nodes, LCEL chains, and LangChain tools.
        """
        if cls._chat_llm is None:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured.")
            cls._chat_llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            logger.info(
                "ChatOpenAI initialized | model=%s | temperature=%.1f | max_tokens=%d",
                settings.OPENAI_MODEL,
                settings.DEFAULT_TEMPERATURE,
                settings.MAX_TOKENS,
            )
        return cls._chat_llm

    # ------------------------------------------------------------------
    # Raw OpenAI SDK client
    # ------------------------------------------------------------------

    @classmethod
    def get_openai_client(cls) -> OpenAI:
        """
        Return the shared raw OpenAI SDK client.
        Use this when you need direct control over system prompts,
        response parsing, or multi-turn message construction without
        LangChain abstractions.
        """
        if cls._openai is None:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured.")
            cls._openai = OpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(
                "OpenAI SDK client initialized | model=%s",
                settings.OPENAI_MODEL,
            )
        return cls._openai

    # ------------------------------------------------------------------
    # Convenience: call the raw client with a single user message
    # ------------------------------------------------------------------

    @classmethod
    def call(
        cls,
        user: str,
        system: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        One-shot helper for services that need a plain text response.

        Returns the text content of the first choice.
        Falls back to settings defaults for model / temperature / max_tokens.
        """
        client = cls.get_openai_client()
        response = client.chat.completions.create(
            model=model or settings.OPENAI_MODEL,
            max_tokens=max_tokens or settings.MAX_TOKENS,
            temperature=temperature if temperature is not None else 0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()


# ==========================================================================
# Module-level singletons — import these directly
# ==========================================================================

llm = LLMProvider.get_chat_llm()
openai_client = LLMProvider.get_openai_client()
