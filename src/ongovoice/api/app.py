"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ongovoice.api.routes import health, sessions, turns


def create_app() -> FastAPI:
    app = FastAPI(
        title="OngoVoice-RT API",
        version="0.2.0",
        description=(
            "Edge-first conversational pipeline for the Ongo companion robot. "
            "POST a transcript or stream bidi audio over WebSocket."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:8001"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(turns.router, prefix="/api/v1", tags=["turns"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    return app


app = create_app()
