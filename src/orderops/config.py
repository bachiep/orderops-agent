"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for OrderOps Agent service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8302)
    api_base_url: str = Field(default="http://localhost:8302")
    streamlit_port: int = Field(default=8502)
    aux_port: int = Field(default=6332)

    database_path: str = Field(default="./data/orderops.db")

    llm_provider: str = Field(default="local")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_base_url: str = Field(default="")
    llm_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    llm_timeout_seconds: float = Field(default=20.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    max_steps: int = Field(default=5, ge=1, le=12)
    max_token_budget: int = Field(default=4096, ge=256)
    max_input_chars: int = Field(default=1500, ge=100)
    auto_refund_limit_vnd: int = Field(default=500_000, gt=0)
    return_window_days: int = Field(default=14, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()
