"""Pydantic v2 request and response models.

`extra="forbid"` on every input model (repository convention): an unexpected
field is a client bug or an injection attempt, and silently discarding it hides
both. Responses never carry password hashes, token hashes or raw IPs.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints

Password = Annotated[str, StringConstraints(min_length=12, max_length=128)]
Title = Annotated[str, StringConstraints(min_length=1, max_length=200, strip_whitespace=True)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


class RegisterRequest(StrictModel):
    email: EmailStr
    # 12 chars minimum with no composition rules: length beats character-class
    # requirements, which mostly push people toward Passw0rd! patterns.
    password: Password
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(StrictModel):
    email: EmailStr
    password: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class RefreshRequest(StrictModel):
    refresh_token: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    # S105 false positive: this is the OAuth2 token *type*, not a secret.
    token_type: Literal["bearer"] = "bearer"  # noqa: S105
    expires_at: dt.datetime
    refresh_token: str
    refresh_expires_at: dt.datetime


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    plan: str
    retention_days: int
    allow_training_use: bool
    persist_voiceprints: bool
    created_at: dt.datetime


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------


class CreateProjectRequest(StrictModel):
    title: Title
    # FR-UPL-08: the user attests they have the right to process this media.
    rights_attested: bool = Field(description="User confirms they may process this recording")


class ProjectResponse(BaseModel):
    id: str
    title: str
    status: str
    duration_ms: int | None
    has_video: bool
    speaker_count: int | None
    overlap_ratio: float | None
    difficulty: str | None
    warnings: list[str]
    created_at: dt.datetime


class ProjectListResponse(BaseModel):
    items: list[ProjectResponse]
    next_cursor: str | None = None


class UploadInitRequest(StrictModel):
    filename: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    content_type: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    size_bytes: int = Field(gt=0)
    part_count: int = Field(default=1, ge=1, le=10_000)


class UploadPart(BaseModel):
    part_number: int
    url: str


class CompletedPart(StrictModel):
    part_number: int = Field(ge=1)
    etag: str = Field(min_length=1)


class UploadInitResponse(BaseModel):
    upload_id: str
    key: str
    # The client slices the file on this boundary; it is chosen by the server so
    # very large files stay inside S3's 10,000-part ceiling.
    part_size_bytes: int
    part_count: int
    # Only the first batch. More come from /upload/{id}/parts, because
    # presigning thousands of URLs up front hands out mostly-expired links.
    parts: list[UploadPart]


class UploadPartsRequest(StrictModel):
    """Confirm finished parts and ask for the next batch of URLs."""

    completed: list[CompletedPart] = Field(default_factory=list)
    # Where the client wants to resume from. The server takes the greater of
    # this and what it has already recorded, so a confused client cannot skip
    # parts by asking for a later batch.
    after: int = Field(default=0, ge=0)


class UploadPartsResponse(BaseModel):
    parts: list[UploadPart]
    completed_count: int
    part_count: int


class UploadStatusResponse(BaseModel):
    """What a resuming client needs after a refresh."""

    upload_id: str
    key: str
    status: str
    size_bytes: int
    part_size_bytes: int
    part_count: int
    completed_parts: list[int]
    expires_at: dt.datetime


class CreateExportRequest(StrictModel):
    """Ask for one rendered file."""

    kind: Literal["video", "audio"] = "video"
    # None means every speaker, for an audio-stems export.
    speaker_ordinal: int | None = Field(default=None, ge=1, le=8)
    # Checked against what the source can honestly produce, never trusted.
    quality: str | None = None


class ExportResponse(BaseModel):
    id: str
    project_id: str
    speaker_ordinal: int | None
    kind: str
    quality: str | None
    status: str
    size_bytes: int | None
    expires_at: dt.datetime | None
    created_at: dt.datetime


class ExportListResponse(BaseModel):
    items: list[ExportResponse]


class RenditionOption(BaseModel):
    """A rung this specific recording can serve."""

    name: str
    height: int | None
    kind: str


class LimitsResponse(BaseModel):
    """Live upload limits (docs/28 §D2).

    Computed per request rather than configured: these move with free space and
    with other uploads in flight, and a cap that ignores the disk is a promise
    the machine cannot keep.
    """

    max_upload_bytes: int
    max_duration_seconds: int
    max_speakers: int
    available_bytes: int
    reserved_bytes: int
    quotas: dict[str, dict[str, int]]


class UploadCompleteRequest(StrictModel):
    upload_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    parts: list[CompletedPart] = Field(min_length=1)


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


class JobStageResponse(BaseModel):
    stage: str
    status: str
    ordinal: int
    duration_ms: int | None
    warnings: list[str]


class JobResponse(BaseModel):
    id: str
    project_id: str
    status: str
    progress: int
    pipeline_mode: str
    error_code: str | None
    stages: list[JobStageResponse]


class ErrorResponse(BaseModel):
    """RFC 7807-ish. `code` is stable and machine-readable; `detail` is prose."""

    code: str
    detail: str
