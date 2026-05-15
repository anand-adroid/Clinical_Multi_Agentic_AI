"""
Structured logging.

Every log event carries: run_id, agent_id, step_id, and a stable event_type.
This is what makes the system *auditable* — log lines can be replayed back
into a coherent timeline that matches the audit trail in the DB.

Why structlog + JSON?
  * Machine-readable for downstream observability (Datadog/Splunk/CloudWatch).
  * Stable schema, so SREs can build dashboards (e.g. agent latency, retry
    rate, guardrail-block rate) without parsing free-form strings.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from backend.core.config import settings


def configure_logging() -> None:
    """Idempotent — safe to call from FastAPI, Streamlit, and tests."""

    log_path = Path(settings.log_dir) / "agentic.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Stdlib config so 3rd-party libs (uvicorn, sqlalchemy) flow through us.
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), encoding="utf-8"),
        ],
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name or "agentic")
