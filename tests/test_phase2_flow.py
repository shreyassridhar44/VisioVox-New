"""Phase 2 exit criterion, end to end.

    upload -> mocked processing with live progress -> a ready project
    with a valid manifest

Run against the real Compose stack: Postgres, Redis and MinIO. Mocking the
object store here would skip exactly the parts that break — presigned URL
signing, multipart completion, and the ETag round trip.

The Celery task is invoked in-process rather than through a broker. That keeps
the test hermetic while still exercising the real state machine, the real
database writes and the real manifest builder; broker delivery is Celery's
concern, not ours.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from visiovox_api.config import get_settings
from visiovox_api.db import dispose_engine
from visiovox_api.main import app
from visiovox_api.storage import ObjectStore

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "packages" / "contracts" / "schemas" / "manifest.schema.json"
PASSWORD = "correct-horse-battery-staple"

settings = get_settings()


async def _stack_up() -> bool:
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        await engine.dispose()
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{settings.s3_endpoint_url}/minio/health/live")
            if r.status_code != 200:
                return False
    except Exception:
        return False
    return True


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    if not await _stack_up():
        pytest.skip("Compose stack not running; run `make dev`")
    await dispose_engine()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        await dispose_engine()


@pytest_asyncio.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    email = f"flow-{uuid.uuid4().hex[:10]}@example.com"
    resp = await client.post("/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.mark.asyncio
async def test_full_upload_to_ready_flow(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    # 1. create the project
    created = await client.post(
        "/v1/projects",
        json={"title": "Phase 2 flow", "rights_attested": True},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]

    # 2. start a multipart upload
    payload = b"not real media, but a real object" * 400
    init = await client.post(
        f"/v1/projects/{project_id}/upload/init",
        json={
            "filename": "meeting.mp4",
            "content_type": "video/mp4",
            "size_bytes": len(payload),
            "part_count": 1,
        },
        headers=auth_headers,
    )
    assert init.status_code == 200, init.text
    body = init.json()
    assert body["key"].startswith("u/")
    assert len(body["parts"]) == 1

    # 3. upload the part straight to MinIO using the presigned URL
    async with httpx.AsyncClient(timeout=30) as raw:
        put = await raw.put(body["parts"][0]["url"], content=payload)
    assert put.status_code == 200, put.text
    etag = put.headers["ETag"]

    # 4. complete
    done = await client.post(
        f"/v1/projects/{project_id}/upload/complete",
        json={
            "upload_id": body["upload_id"],
            "key": body["key"],
            "parts": [{"part_number": 1, "etag": etag}],
        },
        headers=auth_headers,
    )
    assert done.status_code == 202, done.text
    job_id = done.json()["job_id"]

    # 5. run the pipeline in-process
    from worker_cpu.tasks import run_mock_pipeline

    result = run_mock_pipeline(job_id, speed=0.0)
    assert result["status"] in {"succeeded", "partial"}, result

    # 6. the project is ready and the manifest validates
    project = await client.get(f"/v1/projects/{project_id}", headers=auth_headers)
    assert project.status_code == 200
    assert project.json()["status"] == "ready"

    manifest_resp = await client.get(f"/v1/projects/{project_id}/manifest", headers=auth_headers)
    assert manifest_resp.status_code == 200, manifest_resp.text
    manifest = manifest_resp.json()

    validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text()), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(manifest), key=lambda e: e.json_path)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)
    assert manifest["project_id"] == project_id

    # 7. stage rows were recorded, in order
    job = await client.get(f"/v1/projects/{project_id}/job", headers=auth_headers)
    assert job.status_code == 200
    stages = job.json()["stages"]
    assert len(stages) == 11
    assert [s["ordinal"] for s in stages] == sorted(s["ordinal"] for s in stages)
    assert all(s["status"] == "succeeded" for s in stages)
    assert job.json()["progress"] == 100


@pytest.mark.asyncio
async def test_rerun_skips_completed_stages(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Invariant 7: a resumed job does not redo work already recorded."""
    created = await client.post(
        "/v1/projects",
        json={"title": "Resume", "rights_attested": True},
        headers=auth_headers,
    )
    project_id = created.json()["id"]

    store = ObjectStore(settings)
    await store.ensure_bucket()

    payload = b"x" * 1024
    init = (
        await client.post(
            f"/v1/projects/{project_id}/upload/init",
            json={
                "filename": "a.mp4",
                "content_type": "video/mp4",
                "size_bytes": len(payload),
                "part_count": 1,
            },
            headers=auth_headers,
        )
    ).json()
    async with httpx.AsyncClient(timeout=30) as raw:
        put = await raw.put(init["parts"][0]["url"], content=payload)
    done = await client.post(
        f"/v1/projects/{project_id}/upload/complete",
        json={
            "upload_id": init["upload_id"],
            "key": init["key"],
            "parts": [{"part_number": 1, "etag": put.headers["ETag"]}],
        },
        headers=auth_headers,
    )
    job_id = done.json()["job_id"]

    from worker_cpu.tasks import run_mock_pipeline

    run_mock_pipeline(job_id, speed=0.0)
    first = await client.get(f"/v1/projects/{project_id}/job", headers=auth_headers)
    attempts_before = len(first.json()["stages"])

    run_mock_pipeline(job_id, speed=0.0)
    second = await client.get(f"/v1/projects/{project_id}/job", headers=auth_headers)
    assert len(second.json()["stages"]) == attempts_before, "stages were duplicated"


@pytest.mark.asyncio
async def test_manifest_before_processing_is_a_conflict(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await client.post(
        "/v1/projects",
        json={"title": "Empty", "rights_attested": True},
        headers=auth_headers,
    )
    pid = created.json()["id"]
    resp = await client.get(f"/v1/projects/{pid}/manifest", headers=auth_headers)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await client.post(
        "/v1/projects",
        json={"title": "Huge", "rights_attested": True},
        headers=auth_headers,
    )
    pid = created.json()["id"]
    resp = await client.post(
        f"/v1/projects/{pid}/upload/init",
        json={
            "filename": "huge.mp4",
            "content_type": "video/mp4",
            "size_bytes": settings.max_upload_bytes + 1,
            "part_count": 1,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_upload_init_on_another_users_project_is_404(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Invariant 4 must hold on the media routes too, not just the CRUD ones."""
    created = await client.post(
        "/v1/projects",
        json={"title": "Owned", "rights_attested": True},
        headers=auth_headers,
    )
    pid = created.json()["id"]

    other = await client.post(
        "/v1/auth/register",
        json={"email": f"x-{uuid.uuid4().hex[:8]}@example.com", "password": PASSWORD},
    )
    intruder = {"Authorization": f"Bearer {other.json()['access_token']}"}

    resp = await client.post(
        f"/v1/projects/{pid}/upload/init",
        json={
            "filename": "a.mp4",
            "content_type": "video/mp4",
            "size_bytes": 100,
            "part_count": 1,
        },
        headers=intruder,
    )
    assert resp.status_code == 404
