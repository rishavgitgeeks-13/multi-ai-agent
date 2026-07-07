"""
Centralized application configuration.

- Loads environment variables from .env
- Provides a single Settings instance throughout the project
- Prevents hardcoding secrets and configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================================================
    # Application
    # ==========================================================
    APP_NAME: str = "Editorial Intelligence System"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # ==========================================================
    # Anthropic
    # ==========================================================
    ANTHROPIC_API_KEY: str = "ANTHROPIC_API_KEY"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # ==========================================================
    # MongoDB
    # ==========================================================
    MONGODB_URI: str = "mongodb+srv://n8n_authentication:Rishav@gitgeeks25@cluster0.9v4ydlw.mongodb.net/?appName=Cluster0"
    MONGODB_DATABASE: str = "editorial_ai"

    # ==========================================================
    # OpenAI  (embeddings for SEO pipeline + Pinecone vector store)
    # ==========================================================
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSION: int = 1536    # text-embedding-3-small default

    # ==========================================================
    # Pinecone
    # ==========================================================
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "multi-agent"
    PINECONE_EMBEDDING_DIMENSION: int = 1536  # must match OPENAI_EMBEDDING_DIMENSION

    # ==========================================================
    # Tavily
    # ==========================================================
    TAVILY_API_KEY: str = "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"

    # ==========================================================
    # LangSmith (Optional but Recommended)
    # ==========================================================
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "Editorial-Agent"

    # ==========================================================
    # LLM Configuration
    # ==========================================================
    DEFAULT_TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 4096

    # ==========================================================
    # Agent Configuration
    # ==========================================================
    MAX_REVIEW_ITERATIONS: int = 3
    MAX_RESEARCH_RESULTS: int = 10

    # ==========================================================
    # Content Configuration
    # ==========================================================
    DEFAULT_LANGUAGE: str = "English"
    MAX_ARTICLE_WORDS: int = 2500
    MIN_ARTICLE_WORDS: int = 1200

    # ==========================================================
    # News Configuration
    # ==========================================================
    NEWS_API_KEY: str
    NEWS_PAGE_SIZE: int = 5

    YOUTUBE_API_KEY: str
    YOUTUBE_MAX_RESULTS: int = 5

    TAVILY_API_KEY: str
    TAVILY_MAX_RESULTS: int = 5
    

    # ==========================================================
    # Prompt Configuration
    # ==========================================================
    SYSTEM_PROMPT_VERSION: str = "v1"

    # ==========================================================
    # Pydantic Settings
    # ==========================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance
settings = Settings()