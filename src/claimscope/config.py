"""Application settings, loaded from the environment and `.env`."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Never hardcode secrets; they come from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLAIMSCOPE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    model_name: str = "claude-sonnet-5"

    runs_dir: Path = Path("runs")

    # Triage and budget limits (PLAN.md sections 6 and 7).
    max_claims_extracted: int = 10
    """Upper bound on claims the extraction node returns, before triage."""

    max_claims: int = 2
    """K: how many claims triage may select for execution."""

    budget_minutes_total: float = 60.0
    seeds_per_arm: int = 3
    max_debug_attempts: int = 3

    # Sandbox limits.
    docker_image: str = "claimscope-cpu:latest"
    sandbox_cpus: float = 2.0
    sandbox_memory_mb: int = 4096
    sandbox_timeout_s: int = 1800

    # Tracing is optional; the app must work without it.
    tracing_enabled: bool = False


def get_settings() -> Settings:
    """Return the settings for this process."""
    return Settings()
