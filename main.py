"""Application entry point for running the server."""

from src.main import app

if __name__ == "__main__":
    import uvicorn
    from src.core.config import settings
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload and settings.debug,
        log_level=settings.log_level.lower(),
    )