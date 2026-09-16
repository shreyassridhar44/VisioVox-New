"""Share links (docs/28 §W7).

The properties worth pinning are all security properties, and each one is a way
this feature could quietly leak someone's meeting:

- a revoked link must die immediately and stay dead
- a link scoped to one speaker must not carry the others' audio
- the raw token must never be stored
- missing, revoked and expired must be indistinguishable to a stranger
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from visiovox_api.config import get_settings
from visiovox_api.db import dispose_engine
from visiovox_api.main import app

pytestmark = pytest.mark.integration

DB_URL = get_settings().database_url

MANIFEST = {
    "version": "1.0",
    "speakers": [
        {"ordinal": 1, "label": "Speaker 1", "audio": {"faithful": {"url": "spk_1_f.m4a"}}},
        {"ordinal": 2, "label": "Speaker 2", "audio": {"faithful": {"url": "spk_2_f.m4a"}}},
        {"ordinal": 3, "label": "Speaker 3", "audio": {"faithful": {"url": "spk_3_f.m4a"}}},
    ],
}


async def _reachable() -> bool:
    try:
        engine = create_async_engine(DB_URL)
        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        await engine.dispose()
    except Exception:
        return False
    return True


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    if not await _reachable():
        pytest.skip("Postgres not reachable; run `make dev`")
    await dispose_engine()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c
    finally:
        await dispose_engine()


async def _ready_project(client: AsyncClient) -> tuple[str, dict[str, str]]:
    """A signed-in user with a project marked ready and carrying a manifest."""
    email = f"share-{uuid.uuid4().hex[:12]}@example.com"
    registered = await client.post(
        "/v1/auth/register", json={"email": email, "password": "correct-horse-battery"}
    )
    auth = {"Authorization": f"Bearer {registered.json()['access_token']}"}

    created = await client.post(
        "/v1/projects", json={"title": "Board meeting", "rights_attested": True}, headers=auth
    )
    project_id = created.json()["id"]

    # The share route needs a ready project with a manifest; the pipeline is not
    # the thing under test here.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from visiovox_api.models import Project

    engine = create_async_engine(DB_URL)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        project = await session.get(Project, project_id)
        assert project is not None
        project.status = "ready"
        project.manifest = MANIFEST
        project.speaker_count = 3
        project.duration_ms = 600_000
        await session.commit()
    await engine.dispose()

    return project_id, auth


# --------------------------------------------------------------------------


async def test_a_share_opens_without_an_account(client: AsyncClient) -> None:
    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    token = created.json()["token"]

    # No Authorization header: the token is the whole authorisation.
    response = await client.get(f"/v1/shared/{token}")
    assert response.status_code == 200
    assert response.json()["title"] == "Board meeting"


async def test_the_token_is_returned_only_once(client: AsyncClient) -> None:
    project_id, auth = await _ready_project(client)
    await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)

    listed = await client.get(f"/v1/projects/{project_id}/shares", headers=auth)
    assert listed.json()["items"][0]["token"] is None


async def test_revocation_takes_effect_immediately(client: AsyncClient) -> None:
    """A revoked share that still opens is a privacy incident, not a cache miss."""
    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    token = created.json()["token"]
    share_id = created.json()["id"]

    assert (await client.get(f"/v1/shared/{token}")).status_code == 200

    revoked = await client.delete(f"/v1/projects/{project_id}/shares/{share_id}", headers=auth)
    assert revoked.status_code == 204

    assert (await client.get(f"/v1/shared/{token}")).status_code == 404


async def test_a_scoped_share_carries_only_that_speaker(client: AsyncClient) -> None:
    """The manifest is the authorisation for the media, so scoping has to happen
    server-side rather than in the player."""
    project_id, auth = await _ready_project(client)
    created = await client.post(
        f"/v1/projects/{project_id}/shares", json={"speaker_ordinal": 2}, headers=auth
    )
    token = created.json()["token"]

    body = (await client.get(f"/v1/shared/{token}")).json()
    speakers = body["manifest"]["speakers"]
    assert [s["ordinal"] for s in speakers] == [2]
    assert "spk_1_f.m4a" not in str(body)
    assert "spk_3_f.m4a" not in str(body)


async def test_an_unscoped_share_carries_everyone(client: AsyncClient) -> None:
    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    body = (await client.get(f"/v1/shared/{created.json()['token']}")).json()
    assert len(body["manifest"]["speakers"]) == 3


async def test_a_password_protected_share_refuses_without_it(client: AsyncClient) -> None:
    project_id, auth = await _ready_project(client)
    created = await client.post(
        f"/v1/projects/{project_id}/shares", json={"password": "open-sesame"}, headers=auth
    )
    token = created.json()["token"]

    assert (await client.get(f"/v1/shared/{token}")).status_code == 401
    assert (await client.get(f"/v1/shared/{token}?password=wrong")).status_code == 401
    assert (await client.get(f"/v1/shared/{token}?password=open-sesame")).status_code == 200


async def test_an_unknown_token_is_indistinguishable_from_a_revoked_one(
    client: AsyncClient,
) -> None:
    """Different answers would tell a stranger whether a token ever existed."""
    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    await client.delete(f"/v1/projects/{project_id}/shares/{created.json()['id']}", headers=auth)

    revoked = await client.get(f"/v1/shared/{created.json()['token']}")
    never_existed = await client.get(f"/v1/shared/{'x' * 43}")

    assert revoked.status_code == never_existed.status_code == 404
    assert revoked.json()["detail"] == never_existed.json()["detail"]


async def test_shared_pages_are_not_indexed(client: AsyncClient) -> None:
    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    response = await client.get(f"/v1/shared/{created.json()['token']}")
    assert "noindex" in response.headers["X-Robots-Tag"]


async def test_views_are_counted(client: AsyncClient) -> None:
    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    token = created.json()["token"]

    for _ in range(3):
        await client.get(f"/v1/shared/{token}")

    listed = await client.get(f"/v1/projects/{project_id}/shares", headers=auth)
    assert listed.json()["items"][0]["access_count"] == 3


async def test_another_users_project_cannot_be_shared(client: AsyncClient) -> None:
    """Invariant 4: 404 rather than 403, so the endpoint is not an existence
    oracle for other people's projects."""
    project_id, _ = await _ready_project(client)
    _, other_auth = await _ready_project(client)

    response = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=other_auth)
    assert response.status_code == 404


async def test_the_raw_token_is_never_stored(client: AsyncClient) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from visiovox_api.models import ShareLink

    project_id, auth = await _ready_project(client)
    created = await client.post(f"/v1/projects/{project_id}/shares", json={}, headers=auth)
    token = created.json()["token"]

    engine = create_async_engine(DB_URL)
    maker = async_sessionmaker(engine)
    async with maker() as session:
        rows = (await session.scalars(__import__("sqlalchemy").select(ShareLink))).all()
        assert all(token not in (r.token_hash or "") for r in rows)
    await engine.dispose()
