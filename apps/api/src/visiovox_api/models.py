"""SQLAlchemy 2.0 models (docs/10-data-model.md).

IDs are prefixed ULIDs stored as TEXT: sortable by creation time, safe to
expose, and free of the enumeration risk sequential integers carry. The prefix
makes a stray id in a log or a bug report self-describing.

Timestamps are TIMESTAMPTZ in UTC everywhere. Media timing is integer
milliseconds (invariant 2) and never appears as float seconds.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from ulid import ULID

PROJECT_STATUSES = (
    "pending",
    "validating",
    "queued",
    "processing",
    "ready",
    "failed",
    "cancelled",
    "expired",
)
JOB_STATUSES = ("queued", "running", "succeeded", "partial", "failed", "cancelled")
STAGE_STATUSES = ("pending", "running", "succeeded", "skipped", "degraded", "failed")


def new_id(prefix: str) -> str:
    """Prefixed ULID, e.g. prj_01HX8ZQ3M7N4P5R6S7T8V9W0XY."""
    return f"{prefix}_{ULID()}"


def _in_list(column: str, values: tuple[str, ...]) -> str:
    """Render a SQL IN check constraint from a Python tuple.

    Keeps the allowed values in one place so the CHECK constraint and the
    application cannot drift apart.
    """
    quote = chr(39)
    joined = ", ".join(f"{quote}{v}{quote}" for v in values)
    return f"{column} IN ({joined})"


class Base(DeclarativeBase):
    pass


def _pk(prefix: str) -> Mapped[str]:
    return mapped_column(Text, primary_key=True, default=lambda: new_id(prefix))


def _now_col() -> Mapped[dt.datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = _pk("usr")
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    email_verified_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    password_hash: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    plan: Mapped[str] = mapped_column(Text, nullable=False, server_default="free")
    data_region: Mapped[str] = mapped_column(Text, nullable=False, server_default="eu")
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default="30")

    # NFR-PRIV-02 / NFR-PRIV-05: the privacy-preserving setting is the default
    # and requires no action from the user (Charter principle 5).
    allow_training_use: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    persist_voiceprints: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    created_at: Mapped[dt.datetime] = _now_col()
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    projects: Mapped[list[Project]] = relationship(back_populates="user")

    __table_args__ = (
        CheckConstraint(_in_list("plan", ("free", "pro", "team")), name="users_plan_check"),
        CheckConstraint(_in_list("data_region", ("eu", "us", "in")), name="users_region_check"),
        CheckConstraint("retention_days BETWEEN 1 AND 365", name="users_retention_check"),
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = _pk("ses")
    user_id: Mapped[str] = mapped_column(
        Text, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    user_agent: Mapped[str | None] = mapped_column(Text)
    ip_hash: Mapped[str | None] = mapped_column(Text)  # hashed, never raw
    created_at: Mapped[dt.datetime] = _now_col()
    last_seen_at: Mapped[dt.datetime] = _now_col()
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class RefreshToken(Base):
    """Rotating refresh tokens with family reuse detection (FR-ACC-04).

    Only the SHA-256 of the token is stored. Presenting one whose used_at is
    already set means it was captured and replayed, so the whole family is
    revoked rather than only that token.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[str] = _pk("rft")
    family_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(
        Text, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    parent_id: Mapped[str | None] = mapped_column(Text, ForeignKey("refresh_tokens.id"))
    used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_refresh_tokens_family", "family_id"),)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = _pk("prj")
    user_id: Mapped[str] = mapped_column(
        Text, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")

    source_key: Mapped[str | None] = mapped_column(Text)
    source_bytes: Mapped[int | None] = mapped_column(BigInteger)
    source_sha256: Mapped[str | None] = mapped_column(String(64))

    duration_ms: Mapped[int | None] = mapped_column(Integer)  # invariant 2
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    has_video: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")

    speaker_count: Mapped[int | None] = mapped_column(Integer)
    overlap_ratio: Mapped[float | None] = mapped_column(Float)
    difficulty: Mapped[str | None] = mapped_column(Text)
    manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    warnings: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")

    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = _now_col()
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="projects")
    job: Mapped[Job | None] = relationship(back_populates="project", uselist=False)

    __table_args__ = (
        CheckConstraint(_in_list("status", PROJECT_STATUSES), name="projects_status_check"),
        CheckConstraint(
            _in_list("difficulty", ("easy", "moderate", "hard")),
            name="projects_difficulty_check",
        ),
        CheckConstraint("speaker_count BETWEEN 0 AND 8", name="projects_speaker_count_check"),
        Index("ix_projects_user_created", "user_id", "created_at"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = _pk("job")
    project_id: Mapped[str] = mapped_column(
        Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    pipeline_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="mock")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_code: Mapped[str | None] = mapped_column(Text)
    error_detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = _now_col()

    project: Mapped[Project] = relationship(back_populates="job")
    stages: Mapped[list[JobStage]] = relationship(back_populates="job", order_by="JobStage.ordinal")

    __table_args__ = (
        CheckConstraint(_in_list("status", JOB_STATUSES), name="jobs_status_check"),
        CheckConstraint("progress BETWEEN 0 AND 100", name="jobs_progress_check"),
    )


class JobStage(Base):
    """One pipeline stage attempt.

    Unique on (job_id, stage, version) so retries are idempotent (invariant 7):
    a resumed job skips stages already recorded as succeeded at that version.
    """

    __tablename__ = "job_stages"

    id: Mapped[str] = _pk("jst")
    job_id: Mapped[str] = mapped_column(
        Text, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(Text, nullable=False)  # S0_ingest ... S9_package
    version: Mapped[str] = mapped_column(Text, nullable=False, server_default="1.0.0")
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    warnings: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    detail: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    job: Mapped[Job] = relationship(back_populates="stages")

    __table_args__ = (
        Index("uq_job_stage_version", "job_id", "stage", "version", unique=True),
        CheckConstraint(_in_list("status", STAGE_STATUSES), name="job_stages_status_check"),
    )
