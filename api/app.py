from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import get_settings, configure_logging
from api.dependencies import startup_handler, shutdown_handler
from api.routers import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle the startup and shutdown lifecycles of the application.
    Initializes the RecommenderService on boot and cleans up on exit.
    """
    await startup_handler()
    yield
    await shutdown_handler()

def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.
    Sets up CORS, middleware, and registers all routers.
    """
    settings = get_settings()
    configure_logging(settings)
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True
    )
