"""
Application Entry Point
=======================

Starts the Editorial Intelligence System API server.

Usage
-----
    # Start API only
    python main.py

    # Start Streamlit frontend (separate terminal)
    streamlit run frontend/app.py

Environment
-----------
    HOST        : bind address (default: 0.0.0.0)
    PORT        : listen port (default: 8000)
    ENVIRONMENT : development enables --reload and debug logging
"""

import os

import uvicorn

from config.settings import settings


def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = settings.ENVIRONMENT == "development"

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
