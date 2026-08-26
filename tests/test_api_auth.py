"""API auth and ownership integration tests, against a real Postgres.

Not mocked, because the things most likely to be wrong here are the parts a
mock would paper over: the unique constraint on email, the refresh-token
family revoke touching more than one row, and the ownership filter actually
reaching the WHERE clause.

Two behaviours carry real security weight and get direct tests:

- **Refresh reuse detection (FR-ACC-04)** — replaying a used token must revoke
  the whole family, not just that token.
- **Ownership (invariant 4)** — another user's project must be indistinguishable
  from a nonexistent one, or the endpoint becomes an existence oracle.
"""

from __future__ import annotations

import os
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


async def _database_reachable() -> bool:
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
    """Fresh engine per test.

    pytest-asyncio gives each test its own event loop, and asyncpg connections
    are bound to the loop that created them. A module-level pool therefore
    hands the second test a connection belonging to a closed loop. Disposing
    around the test is cheaper than reasoning about loop scopes.
    """
    if not await _database_reachable():
        pytest.skip("Postgres not reachable; run `make dev`")
    await dispose_engine()
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        await dispose_engine()


def _email() -> str:
    return f"user-{uuid.uuid4().hex[:12]}@example.com"


PASSWORD = "correct-horse-battery-staple"


async def _register(client: AsyncClient) -> tuple[str, dict[str, str]]:
    email = _email()
    resp = await client.post("/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    return email, body


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthz(client: AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_checks_the_database(client: AsyncClient) -> None:
    resp = await client.get("/readyz")
    assert resp.status_code == 200


# --------------------------------------------------------------------------
# registration and login
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_returns_a_usable_token(client: AsyncClient) -> None:
    _, tokens = await _register(client)
    resp = await client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert resp.status_code == 200
    assert resp.json()["plan"] == "free"


@pytest.mark.asyncio
async def test_privacy_defaults_are_off(client: AsyncClient) -> None:
    """NFR-PRIV-02/05: privacy-preserving behaviour needs no user action."""
    _, tokens = await _register(client)
    resp = await client.get(
        "/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    body = resp.json()
    assert body["allow_training_use"] is False
    assert body["persist_voiceprints"] is False


@pytest.mark.asyncio
async def test_duplicate_registration_is_rejected(client: AsyncClient) -> None:
    email, _ = await _register(client)
    resp = await client.post("/v1/auth/register", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_email_case_does_not_fork_an_identity(client: AsyncClient) -> None:
    email, _ = await _register(client)
    resp = await client.post(
        "/v1/auth/register", json={"email": email.upper(), "password": PASSWORD}
    )
    assert resp.status_code == 409, "uppercase email created a second account"


@pytest.mark.asyncio
async def test_short_password_is_rejected(client: AsyncClient) -> None:
    resp = await client.post("/v1/auth/register", json={"email": _email(), "password": "short"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_unknown_field_is_rejected(client: AsyncClient) -> None:
    """extra='forbid': a typo must fail, not be silently dropped."""
    resp = await client.post(
        "/v1/auth/register",
        json={"email": _email(), "password": PASSWORD, "is_admin": True},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_wrong_password_is_unauthorised(client: AsyncClient) -> None:
    email, _ = await _register(client)
    resp = await client.post(
        "/v1/auth/login", json={"email": email, "password": "wrong-password-entirely"}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_no_token_is_unauthorised(client: AsyncClient) -> None:
    assert (await client.get("/v1/auth/me")).status_code == 401


# --------------------------------------------------------------------------
# refresh rotation and reuse detection
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_rotates_the_token(client: AsyncClient) -> None:
    _, tokens = await _register(client)
    resp = await client.post("/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 200
    rotated = resp.json()
    assert rotated["refresh_token"] != tokens["refresh_token"]


@pytest.mark.asyncio
async def test_reusing_a_refresh_token_revokes_the_family(client: AsyncClient) -> None:
    """FR-ACC-04. Replay means theft or a client bug; we cannot tell, so revoke."""
    _, tokens = await _register(client)
    first = tokens["refresh_token"]

    rotated = (await client.post("/v1/auth/refresh", json={"refresh_token": first})).json()
    second = rotated["refresh_token"]

    # replay the consumed token
    replay = await client.post("/v1/auth/refresh", json={"refresh_token": first})
    assert replay.status_code == 401
    assert "reuse" in replay.json()["detail"].lower()

    # the legitimate successor must also stop working
    after = await client.post("/v1/auth/refresh", json={"refresh_token": second})
    assert after.status_code == 401, "family revoke did not invalidate the successor"


@pytest.mark.asyncio
async def test_revoked_session_rejects_its_access_token(client: AsyncClient) -> None:
    """A revoked session must fail immediately, not when the JWT expires."""
    _, tokens = await _register(client)
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert (await client.get("/v1/auth/me", headers=headers)).status_code == 200

    await client.post("/v1/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert (await client.get("/v1/auth/me", headers=headers)).status_code == 401


@pytest.mark.asyncio
async def test_garbage_refresh_token_is_rejected(client: AsyncClient) -> None:
    resp = await client.post("/v1/auth/refresh", json={"refresh_token": "not-a-token"})
    assert resp.status_code == 401


# --------------------------------------------------------------------------
# projects and ownership
# --------------------------------------------------------------------------


async def _make_project(client: AsyncClient, tokens: dict[str, str], title: str) -> str:
    resp = await client.post(
        "/v1/projects",
        json={"title": title, "rights_attested": True},
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


@pytest.mark.asyncio
async def test_create_and_read_own_project(client: AsyncClient) -> None:
    _, tokens = await _register(client)
    pid = await _make_project(client, tokens, "My recording")
    assert pid.startswith("prj_")

    resp = await client.get(
        f"/v1/projects/{pid}",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "My recording"


@pytest.mark.asyncio
async def test_rights_attestation_is_required(client: AsyncClient) -> None:
    """FR-UPL-08: refuse rather than record a false attestation."""
    _, tokens = await _register(client)
    resp = await client.post(
        "/v1/projects",
        json={"title": "x", "rights_attested": False},
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_another_users_project_is_not_found(client: AsyncClient) -> None:
    """Invariant 4. 404 not 403 — 403 would confirm the id exists."""
    _, owner = await _register(client)
    pid = await _make_project(client, owner, "Private")

    _, intruder = await _register(client)
    resp = await client.get(
        f"/v1/projects/{pid}",
        headers={"Authorization": f"Bearer {intruder['access_token']}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_another_users_project_cannot_be_deleted(client: AsyncClient) -> None:
    _, owner = await _register(client)
    pid = await _make_project(client, owner, "Private")
    _, intruder = await _register(client)

    resp = await client.delete(
        f"/v1/projects/{pid}",
        headers={"Authorization": f"Bearer {intruder['access_token']}"},
    )
    assert resp.status_code == 404

    still_there = await client.get(
        f"/v1/projects/{pid}",
        headers={"Authorization": f"Bearer {owner['access_token']}"},
    )
    assert still_there.status_code == 200, "another user managed to delete it"


@pytest.mark.asyncio
async def test_project_list_is_scoped_to_the_owner(client: AsyncClient) -> None:
    _, a = await _register(client)
    await _make_project(client, a, "A only")
    _, b = await _register(client)

    resp = await client.get(
        "/v1/projects", headers={"Authorization": f"Bearer {b['access_token']}"}
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.asyncio
async def test_malformed_project_id_is_rejected(client: AsyncClient) -> None:
    _, tokens = await _register(client)
    resp = await client.get(
        "/v1/projects/not-a-ulid",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert resp.status_code == 422


_ = os  # keep the import meaningful if env gating is added later
