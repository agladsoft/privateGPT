"""FastAPI app creation, logger configuration and main API routes."""
import os
import logging
from celery import Celery

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector

from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app_fastapi = FastAPI(dependencies=[Depends(bind_injector_to_request)])
    app_fastapi.include_router(completions_router)
    app_fastapi.include_router(chat_router)
    app_fastapi.include_router(chunks_router)
    app_fastapi.include_router(ingest_router)
    app_fastapi.include_router(embeddings_router)
    app_fastapi.include_router(health_router)

    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app_fastapi.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    if settings.ui.enabled:
        logger.debug("Importing the UI module")
        from private_gpt.ui.ui import PrivateGptUi, app

        ui = root_injector.get(PrivateGptUi)

        @app.task
        def run():
            return ui.mount_in_app(app_fastapi, settings.ui.path)

    return app_fastapi
