"""
Centralized application configuration.

- Loads environment variables from .env
- Provides a singleton Settings object
- Keeps secrets and deployment config outside the codebase
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
    # LLM Providers
    # ==========================================================
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSION: int = 1536

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # ==========================================================
    # Vector Database
    # ==========================================================
    PINECONE_API_KEY: str | None = None
    PINECONE_INDEX_NAME: str = "multi-agent"
    PINECONE_EMBEDDING_DIMENSION: int = 1536

    # ==========================================================
    # Research APIs (Optional)
    # ==========================================================
    TAVILY_API_KEY: str | None = None
    TAVILY_MAX_RESULTS: int = 5

    NEWS_API_KEY: str | None = None
    NEWS_PAGE_SIZE: int = 5

    YOUTUBE_API_KEY: str | None = None
    YOUTUBE_MAX_RESULTS: int = 5

    # ==========================================================
    # Reddit (Optional)
    # ==========================================================
    REDDIT_CLIENT_ID: str | None = None
    REDDIT_CLIENT_SECRET: str | None = None
    REDDIT_USERNAME: str | None = None
    REDDIT_PASSWORD: str | None = None
    REDDIT_USER_AGENT: str | None = None

    # ==========================================================
    # Database
    # ==========================================================
    MONGODB_URI: str | None = None
    MONGODB_DATABASE: str = "editorial_ai"

    # ==========================================================
    # LangSmith (Optional)
    # ==========================================================
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "Editorial-Agent"

    # ==========================================================
    # Agent Configuration
    # ==========================================================
    MAX_REVIEW_ITERATIONS: int = 3
    MAX_RESEARCH_RESULTS: int = 10
    DEFAULT_TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 4096

    # ==========================================================
    # Content Configuration
    # ==========================================================
    DEFAULT_LANGUAGE: str = "English"
    MIN_ARTICLE_WORDS: int = 1200
    MAX_ARTICLE_WORDS: int = 2500

    # ==========================================================
    # Environment file configuration
    # ==========================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()