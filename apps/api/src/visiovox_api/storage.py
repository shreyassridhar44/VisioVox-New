"""S3-compatible object storage: presigned multipart upload (ADR-0012).

Uploads go **browser to storage directly**, never through the API. A 2 GB video
streamed through FastAPI would occupy a worker for minutes and make the control
plane's memory profile depend on upload size. Presigned URLs keep the API in the
control path and out of the data path.

Multipart matters for the same reason retries do: a 2 GB upload over a
university connection will drop, and re-sending only the failed part is the
difference between a resumable upload and an unusable one.

Keys are namespaced by user id, so a bug in key construction cannot produce a
path that reads another user's object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import aioboto3
from botocore.config import Config

from .config import Settings

# Anything outside this set is replaced; a filename reaches us from a browser
# and must never be able to escape its prefix or inject a path segment.
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
MAX_FILENAME = 120


class StorageError(RuntimeError):
    """Object storage refused the operation."""


@dataclass(frozen=True)
class PresignedPart:
    part_number: int
    url: str


@dataclass(frozen=True)
class MultipartUpload:
    upload_id: str
    key: str
    parts: list[PresignedPart]


def safe_filename(filename: str) -> str:
    """Strip a client-supplied filename down to something safe to concatenate."""
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = _SAFE_CHARS.sub("_", base).strip("._") or "upload"
    return cleaned[:MAX_FILENAME]


def object_key(user_id: str, project_id: str, filename: str) -> str:
    """Per-user, per-project prefix. Never interpolate a raw filename."""
    return f"u/{user_id}/p/{project_id}/source/{safe_filename(filename)}"


class ObjectStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = aioboto3.Session()

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "service_name": "s3",
            "endpoint_url": self._settings.s3_endpoint_url,
            "region_name": self._settings.s3_region,
            "aws_access_key_id": self._settings.s3_access_key_id,
            "aws_secret_access_key": self._settings.s3_secret_access_key.get_secret_value(),
            "config": Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if self._settings.s3_force_path_style else "auto"},
            ),
        }

    async def create_multipart_upload(
        self,
        user_id: str,
        project_id: str,
        filename: str,
        content_type: str,
        part_count: int,
    ) -> MultipartUpload:
        key = object_key(user_id, project_id, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            created = await s3.create_multipart_upload(
                Bucket=self._settings.s3_bucket,
                Key=key,
                ContentType=content_type,
            )
            upload_id = str(created["UploadId"])

            parts: list[PresignedPart] = []
            for n in range(1, part_count + 1):
                url = await s3.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": self._settings.s3_bucket,
                        "Key": key,
                        "UploadId": upload_id,
                        "PartNumber": n,
                    },
                    ExpiresIn=self._settings.signed_url_ttl_seconds,
                )
                parts.append(PresignedPart(part_number=n, url=url))

        return MultipartUpload(upload_id=upload_id, key=key, parts=parts)

    async def complete_multipart_upload(
        self, key: str, upload_id: str, parts: list[tuple[int, str]]
    ) -> int:
        """Finish the upload and return the object size in bytes."""
        ordered = sorted(parts, key=lambda p: p[0])
        async with self._session.client(**self._client_kwargs()) as s3:
            try:
                await s3.complete_multipart_upload(
                    Bucket=self._settings.s3_bucket,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={
                        "Parts": [{"PartNumber": n, "ETag": etag} for n, etag in ordered]
                    },
                )
                head = await s3.head_object(Bucket=self._settings.s3_bucket, Key=key)
            except Exception as exc:
                raise StorageError(str(exc)) from exc
        return int(head["ContentLength"])

    async def abort_multipart_upload(self, key: str, upload_id: str) -> None:
        async with self._session.client(**self._client_kwargs()) as s3:
            await s3.abort_multipart_upload(
                Bucket=self._settings.s3_bucket, Key=key, UploadId=upload_id
            )

    async def presign_get(self, key: str) -> str:
        async with self._session.client(**self._client_kwargs()) as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._settings.s3_bucket, "Key": key},
                ExpiresIn=self._settings.signed_url_ttl_seconds,
            )
        return str(url)

    async def ensure_bucket(self) -> None:
        async with self._session.client(**self._client_kwargs()) as s3:
            try:
                await s3.head_bucket(Bucket=self._settings.s3_bucket)
            except Exception:
                await s3.create_bucket(Bucket=self._settings.s3_bucket)
