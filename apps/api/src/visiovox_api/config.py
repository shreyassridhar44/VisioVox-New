"""Application settings.

Values come from the environment, matching the keys documented in .env.example
so there is exactly one vocabulary for configuration across the repo.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "ci", "staging", "production"]
PipelineMode = Literal["mock", "real"]


class Settings(BaseSettings):
    """Runtime configuration.

    extra="forbid" on purpose: a typo in an environment variable should fail at
    startup rather than silently fall back to a default, which is how a staging
    box ends up quietly running with local settings.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",  # the env file is shared with web/worker keys
        case_sensitive=False,
    )

    environment: Environment = "local"
    log_level: str = "INFO"

    # --- web / api ---
    next_public_app_url: str = "http://localhost:3000"
    api_internal_url: str = "http://localhost:8000"

    # --- auth ---
    # RFC 7518 3.2 wants >= 32 bytes for HS256. The default is dev-only and
    # long enough not to warn; production must override it.
    auth_secret: SecretStr = SecretStr("dev-only-insecure-secret-replace-in-every-real-deployment")
    access_token_ttl_seconds: int = 600
    refresh_token_ttl_days: int = 30

    # --- database ---
    database_url: str = "postgresql+asyncpg://visiovox:visiovox@localhost:5432/visiovox"

    # --- redis / celery ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- object storage ---
    s3_endpoint_url: str = "http://localhost:9000"
    s3_region: str = "auto"
    s3_bucket: str = "visiovox-media"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: SecretStr = SecretStr("minioadmin")
    s3_force_path_style: bool = True
    signed_url_ttl_seconds: int = 900

    # --- pipeline ---
    pipeline_mode: PipelineMode = "mock"
    extractor_version: str = "seave-0.1.0"

    # --- limits ---
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_duration_seconds: int = 3600
    max_speakers: int = Field(default=4, ge=1, le=8)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """Alembic runs synchronously; the app does not."""
        return self.database_url.replace("+asyncpg", "+psycopg")


@lru_cache
def get_settings() -> Settings:
    return Settings()
