"""
LLM provider singletons for the Editorial Intelligence System.

Two clients are exposed so every layer of the app uses a shared, lazily-
initialized connection rather than creating its own:

  llm              — LangChain ChatAnthropic
                     Used by agents that rely on LangChain / LangGraph LCEL chains.

  anthropic_client — Raw Anthropic SDK client
                     Used by services (e.g. SEOService) that call the API directly
                     for tighter control over prompts and response parsing.

Import pattern
--------------
    from models.llm import llm                 # LangChain agent
    from models.llm import anthropic_client    # direct SDK call
"""

import logging

from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    """Lazy singleton factory for both Anthropic client flavours."""

    _chat_llm: ChatAnthropic = None
    _anthropic: Anthropic = None

    # ------------------------------------------------------------------
    # LangChain ChatAnthropic
    # ------------------------------------------------------------------

    @classmethod
    def get_chat_llm(cls) -> ChatAnthropic:
        """
        Return the shared LangChain ChatAnthropic instance.
        Compatible with LangGraph nodes, LCEL chains, and LangChain tools.
        """
        if cls._chat_llm is None:
            cls._chat_llm = ChatAnthropic(
                model=settings.ANTHROPIC_MODEL,
                api_key=settings.ANTHROPIC_API_KEY,
                temperature=settings.DEFAULT_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
            )
            logger.info(
                "ChatAnthropic initialized | model=%s | temperature=%.1f | max_tokens=%d",
                settings.ANTHROPIC_MODEL,
                settings.DEFAULT_TEMPERATURE,
                settings.MAX_TOKENS,
            )
        return cls._chat_llm

    # ------------------------------------------------------------------
    # Raw Anthropic SDK client
    # ------------------------------------------------------------------

    @classmethod
    def get_anthropic_client(cls) -> Anthropic:
        """
        Return the shared raw Anthropic SDK client.
        Use this when you need direct control over system prompts,
        response parsing, or multi-turn message construction without
        LangChain abstractions.
        """
        if cls._anthropic is None:
            cls._anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info(
                "Anthropic SDK client initialized | model=%s",
                settings.ANTHROPIC_MODEL,
            )
        return cls._anthropic

    # ------------------------------------------------------------------
    # Convenience: call the raw client with a single user message
    # ------------------------------------------------------------------

    @classmethod
    def call(
        cls,
        user: str,
        system: str = "You are a helpful assistant.",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        One-shot helper for services that need a plain text response.

        Returns the text content of the first message block.
        Falls back to settings defaults for model / temperature / max_tokens.
        """
        client = cls.get_anthropic_client()
        response = client.messages.create(
            model=model or settings.ANTHROPIC_MODEL,
            max_tokens=max_tokens or settings.MAX_TOKENS,
            temperature=temperature if temperature is not None else 0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# ==========================================================================
# Module-level singletons — import these directly
# ==========================================================================

llm = LLMProvider.get_chat_llm()
anthropic_client = LLMProvider.get_anthropic_client()
