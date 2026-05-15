"""
Configuration — single source of truth for tunables, paths, and feature flags.

I use ``pydantic-settings`` so every value can be overridden by an env var
(and therefore by a Kubernetes ConfigMap or Secret in production) without
code changes. The defaults below are tuned for a local dev run; production
deployments lock down the security-sensitive ones (``allow_network_in_sandbox``,
``log_json``) and pin the LLM model / token budget per environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    # Primary provider
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    # Fallback provider — used automatically when Anthropic is unavailable
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    llm_max_tokens: int = Field(default=1500, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # ---- API ----
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")

    # ---- Storage ----
    db_url: str = Field(default=f"sqlite:///{PROJECT_ROOT}/storage/agentic.db", alias="DB_URL")
    checkpoint_dir: Path = Field(default=PROJECT_ROOT / "storage" / "checkpoints", alias="CHECKPOINT_DIR")
    run_artifact_dir: Path = Field(default=PROJECT_ROOT / "storage" / "runs", alias="RUN_ARTIFACT_DIR")
    memory_dir: Path = Field(default=PROJECT_ROOT / "storage" / "memory", alias="MEMORY_DIR")
    log_dir: Path = Field(default=PROJECT_ROOT / "storage" / "logs", alias="LOG_DIR")

    # ---- Guardrails ----
    max_refine_retries: int = Field(default=3, alias="MAX_REFINE_RETRIES")
    max_runtime_seconds: int = Field(default=120, alias="MAX_RUNTIME_SECONDS")
    allow_network_in_sandbox: bool = Field(default=False, alias="ALLOW_NETWORK_IN_SANDBOX")
    # Phase 3: when True, pipeline pauses for human approval of every generated
    # ``derive(row)`` before execution. Default False — only low-confidence
    # derivations pause automatically (Phase 5).
    require_code_preapproval: bool = Field(default=False, alias="REQUIRE_CODE_PREAPPROVAL")
    # Phase 5: derivations whose LLM-reported confidence falls below this
    # threshold trigger HITL automatically, regardless of the preapproval flag.
    min_confidence_threshold: float = Field(default=0.7, alias="MIN_CONFIDENCE_THRESHOLD")
    # Phase 3: how many rows of the dataset to dry-run each derive on for the
    # preapproval preview.
    preapproval_preview_rows: int = Field(default=3, alias="PREAPPROVAL_PREVIEW_ROWS")
    # Scalability: when True the executor runs independent derivations in the
    # same DAG level concurrently using a thread pool. Off by default because
    # single-threaded execution gives the auditor a deterministic per-row
    # event order; flip this for large datasets where wall-clock dominates.
    parallel_executor: bool = Field(default=False, alias="PARALLEL_EXECUTOR")
    parallel_executor_workers: int = Field(default=4, alias="PARALLEL_EXECUTOR_WORKERS")

    # ---- Telemetry ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    @property
    def llm_enabled(self) -> bool:
        """True when at least one LLM provider is configured.
        The client tries Anthropic first, then OpenAI, then falls back to the
        deterministic stub / HITL path — so the pipeline always completes with
        a full audit trail regardless of provider availability."""
        return bool(self.anthropic_api_key) or bool(
            self.openai_api_key and self.openai_api_key != "your-openai-api-key-here"
        )

    def ensure_dirs(self) -> None:
        for d in (self.checkpoint_dir, self.run_artifact_dir, self.memory_dir, self.log_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
