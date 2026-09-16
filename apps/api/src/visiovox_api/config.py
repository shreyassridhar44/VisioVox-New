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
    # A hung Redis must not hold an API worker open; the limiter treats a
    # timeout as "unavailable" and falls back.
    redis_timeout_seconds: float = 2.0

    # --- rate limiting (docs/11 §10) ---
    rate_limit_enabled: bool = True
    # Redis down: allow requests rather than take the API down with the cache.
    # A gap in limiting is recoverable; a hard outage is not, and this matches
    # how the rest of the codebase treats Redis. Set false where the opposite
    # trade is wanted.
    rate_limit_fail_open: bool = True
    # Behind a tunnel or CDN, request.client.host is the proxy and every user
    # shares one bucket. Set this to the header the proxy overwrites (e.g.
    # "cf-connecting-ip"). Leave unset otherwise: a spoofable header is worse
    # than none, because it hands every caller a fresh identity per request.
    trusted_client_ip_header: str | None = None
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

    # --- media volume (docs/track-w/W0) ---
    # Where uploads and derived artifacts live. This must be a dedicated
    # filesystem: measuring "/" inside the WSL distro returns the vhdx's
    # virtual maximum rather than real free space, which is how a disk check
    # passes on a full drive.
    media_root: str = "/srv/media"
    # Held back from the usable figure. A job needs source, working copy and
    # outputs on disk at once; running the volume to zero corrupts whatever is
    # mid-write, not only the job that overshot.
    disk_reserved_bytes: int = 20 * 1024**3
    # Peak disk a job occupies relative to its source size. Provisional until
    # measured in W2 — the working copy is small, the export is not.
    upload_peak_multiplier: float = 2.5
    # Off locally and in CI, which legitimately have no separate volume. On for
    # the deployed workstation, which is the machine whose df lies.
    require_dedicated_media_volume: bool = False

    # --- limits ---
    # A backstop against one absurd upload, not the limit shown to users: that
    # is computed from live headroom (docs/28 D2).
    max_upload_bytes: int = 50 * 1024**3
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
