"""Application settings, loaded from the environment and `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "google"]

# Used when the provider is switched without naming a model.
DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-sonnet-5",
    "google": "gemini-3.6-flash",
}


class Settings(BaseSettings):
    """Runtime configuration. Never hardcode secrets; they come from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLAIMSCOPE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    provider: Provider = "anthropic"
    """Which LLM backend to use. The plan specifies Anthropic; Google is
    supported so the pipeline can run when no Anthropic credit is available."""

    model_name: str = ""
    """Model id. Left empty, it defaults to the provider's model."""

    @model_validator(mode="after")
    def _default_model_for_provider(self) -> Settings:
        if not self.model_name:
            self.model_name = DEFAULT_MODELS[self.provider]
        return self

    @property
    def active_api_key(self) -> str | None:
        """The API key for the selected provider."""
        return self.anthropic_api_key if self.provider == "anthropic" else self.gemini_api_key

    runs_dir: Path = Path("runs")

    # Triage and budget limits (PLAN.md sections 6 and 7).
    max_claims_extracted: int = 10
    """Upper bound on claims the extraction node returns, before triage."""

    max_claims: int = 2
    """K: how many claims triage may select for execution."""

    budget_minutes_total: float = 60.0
    seeds_per_arm: int = 3
    max_debug_attempts: int = 3

    full_run_steps: int = 500
    """Steps a full run is assumed to take, used to extrapolate the dry run."""

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
