"""MaskWall firewall API (FastAPI).

`create_app()` builds the app; components (injection scanner, LLM client, event log) can be injected, which is how the tests run
without the exported model, the network or real API keys. Endpoints are added step by step (see docs/decisions.md Day 2).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .config import Settings

log = logging.getLogger("maskwall")


def create_app(settings: Settings | None = None, *, scanner=None, llm=None, db=None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.scanner = scanner
        app.state.llm = llm
        app.state.db = db
        yield

    app = FastAPI(title="MaskWall firewall", version=__version__, lifespan=lifespan)
    # available before startup too (tests that do not use the context manager)
    app.state.settings, app.state.scanner, app.state.llm, app.state.db = settings, scanner, llm, db

    @app.get("/health")
    def health():
        s = app.state
        return {
            "status": "ok",
            "version": __version__,
            "server_model": settings.server_model,
            "scan_mode": settings.scan_mode,
            "model_loaded": s.scanner is not None,
            "llm_providers": [p for p in settings.llm_order if settings.provider_key(p)],
        }

    return app
