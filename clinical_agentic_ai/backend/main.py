"""
FastAPI entrypoint.

Run locally with:

    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import __version__
from backend.api.routes import audit, dag, eval as eval_route, hitl, hitl_quality, memory, workflows
from backend.core.config import settings
from backend.db.session import init_db
from backend.utils.logging_setup import configure_logging, get_logger


def create_app() -> FastAPI:
    configure_logging()
    init_db()
    app = FastAPI(
        title="Clinical Agentic AI - Workflow API",
        version=__version__,
        description="Multi-agent pipeline for clinical data derivation, "
                    "verification, and full traceability.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(workflows.router)
    app.include_router(hitl.router)
    app.include_router(hitl_quality.router)
    app.include_router(audit.router)
    app.include_router(memory.router)
    app.include_router(eval_route.router)
    app.include_router(dag.router)

    log = get_logger("api")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "llm_enabled": settings.llm_enabled,
        }

    @app.get("/")
    def root() -> dict:
        return {
            "name": "Clinical Agentic AI",
            "version": __version__,
            "docs": "/docs",
            "endpoints": [
                "POST /runs",
                "POST /runs/upload-and-run",
                "POST /runs/{run_id}/start",
                "GET  /runs/{run_id}",
                "GET  /runs/{run_id}/state",
                "GET  /runs/{run_id}/events",
                "GET  /runs/{run_id}/validations",
                "GET  /runs/{run_id}/derivations",
                "GET  /runs/{run_id}/output",
                "GET  /runs/{run_id}/audit",
                "GET  /runs/{run_id}/audit/report",
                "GET  /runs/{run_id}/dag.dot",
                "GET  /runs/{run_id}/hitl/pending",
                "POST /runs/{run_id}/hitl/decision",
                "GET  /memory/patterns",
                "GET  /eval/{run_id}",
            ],
        }

    log.info("api.ready", llm_enabled=settings.llm_enabled)
    return app


app = create_app()
