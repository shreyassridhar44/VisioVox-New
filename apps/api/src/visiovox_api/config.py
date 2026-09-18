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
    # "lax" is right when the app and API share a registrable domain, which
    # they do behind one tunnel. A genuinely cross-site deployment needs
    # "none", which browsers only honour together with Secure.
    refresh_cookie_samesite: Literal["lax", "strict", "none"] = "lax"

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
    # The trained SEAVE checkpoint the GPU worker loads. Pinned by path rather
    # than discovered, so a job records which weights produced it.
    #
    # NOT c2-v2, despite it being the newest and scoring +11.43 dB on
    # VoxCeleb2. Measured on the product condition -- real speech, simulated
    # room, verified overlap -- c2-v2 scores -1.61 dB at two speakers: it makes
    # user audio worse than leaving it alone. c3 is the only checkpoint that
    # holds up there (+8.25 / +6.34 dB at two and three speakers), because it
    # is the only one trained with room simulation and not subsequently
    # fine-tuned away from it. See scripts/eval_overlap.py.
    #
    # c2-v3 is training now with both corpora blended; move this on to it only
    # once eval_overlap.py says it beats c3 on the product condition, not on
    # the strength of a VoxCeleb2 number.
    extractor_checkpoint: str = "~/runs/c3/best.pt"
    # "cuda" on the workstation, "cpu" anywhere else. Stage code falls back on
    # its own if CUDA is absent, but being explicit keeps CI honest.
    torch_device: str = "cuda"
    # Confinement image for ffmpeg/ffprobe on user media (ADR-0009).
    media_sandbox_image: str = "visiovox/media-sandbox:1"
    # ECAPA, for scoring enrolment regions. Read from a local directory so a
    # job never depends on the network.
    speaker_embedder_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_embedder_dir: str = "~/models/ecapa"
    # pyannote diarization is gated; without a token S2a degrades to VAD only
    # and the pipeline falls back to single-speaker handling.
    hf_token: SecretStr = SecretStr("")
    # Where packaged artifacts are served from. The packager writes bare
    # filenames; this is the one place that decides how they are reached.
    public_media_base_url: str = "http://localhost:9000/visiovox-media"

    # --- media volume (docs/track-w/W0) ---
    # Where uploads and derived artifacts live. This must be a dedicated
    # filesystem: measuring "/" inside the WSL distro returns the vhdx's
    # virtual maximum rather than real free space, which is how a disk check
    # passes on a full drive.
    media_root: str = "/srv/media"
    # Job scratch. A subdirectory rather than the volume root, because the root
    # also holds MinIO's data and the served artifacts, and a job's temporary
    # tree must be safe to delete wholesale without touching either.
    media_work_dir: str = "/srv/media/work"
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

    # --- quotas (docs/15 §9) ---
    # One set of limits for everyone: whether plans exist at all is still open
    # (docs/track-w DECISIONS.md D6.1), and a plan dimension added now would be
    # structure built for a decision nobody has made.
    #
    # Nothing is rented, so exceeding these costs GPU time and disk rather than
    # money. The controls are the same; what they protect is queue fairness.
    quota_uploads_per_day: int = 20
    quota_media_seconds_per_month: int = 36_000  # 10 hours of source media
    quota_gpu_seconds_per_month: int = 36_000  # 10 GPU-hours
    quota_concurrent_jobs: int = 2

    # Salt for hashing IPs before they reach the audit log. The IPv4 space is
    # small enough to exhaust, so an unsalted hash is decorative. Rotate it and
    # historical addresses become permanently unlinkable, which is the point.
    audit_ip_salt: SecretStr = SecretStr("dev-only-audit-salt-replace-in-production")

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
