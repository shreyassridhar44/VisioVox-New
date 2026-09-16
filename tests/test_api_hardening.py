"""Security headers, problem details and correlation ids (docs/11 §1, §11).

Driven through the real ASGI stack against `/healthz`, which needs no database,
so these run everywhere rather than only where Postgres happens to be up. The
point is the middleware order and the error shape, both of which are easy to
break from a distance and invisible until something else goes wrong.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from visiovox_api.main import app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------------------
# security headers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "no-referrer"),
        ("Cross-Origin-Resource-Policy", "same-origin"),
    ],
)
async def test_security_headers_present(client: AsyncClient, header: str, expected: str) -> None:
    response = await client.get("/healthz")
    assert response.headers[header] == expected


async def test_api_csp_forbids_everything(client: AsyncClient) -> None:
    """The API renders nothing, so the strictest policy that exists is correct.

    The web app's CSP is a separate and much more permissive thing, because it
    has to allow WebGL.
    """
    csp = (await client.get("/healthz")).headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp


async def test_hsts_absent_outside_production(client: AsyncClient) -> None:
    """Sending HSTS over plain HTTP is meaningless, and in local development it
    pins localhost to https in the browser - genuinely painful to undo."""
    assert "Strict-Transport-Security" not in (await client.get("/healthz")).headers


# --------------------------------------------------------------------------
# correlation id
# --------------------------------------------------------------------------


async def test_correlation_id_is_issued(client: AsyncClient) -> None:
    value = (await client.get("/healthz")).headers.get("X-Correlation-Id")
    assert value


async def test_correlation_id_is_echoed_when_supplied(client: AsyncClient) -> None:
    """So a trace can span the browser and the API."""
    response = await client.get("/healthz", headers={"X-Correlation-Id": "abc123"})
    assert response.headers["X-Correlation-Id"] == "abc123"


async def test_supplied_correlation_id_is_length_capped(client: AsyncClient) -> None:
    """It lands in log lines; an unbounded header is a log-volume problem."""
    response = await client.get("/healthz", headers={"X-Correlation-Id": "x" * 500})
    assert len(response.headers["X-Correlation-Id"]) == 64


async def test_ids_differ_between_requests(client: AsyncClient) -> None:
    first = (await client.get("/healthz")).headers["X-Correlation-Id"]
    second = (await client.get("/healthz")).headers["X-Correlation-Id"]
    assert first != second


# --------------------------------------------------------------------------
# problem details
# --------------------------------------------------------------------------


async def test_not_found_is_problem_json(client: AsyncClient) -> None:
    response = await client.get("/v1/projects/prj_00000000000000000000000000")
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_problem_body_has_the_documented_shape(client: AsyncClient) -> None:
    response = await client.get("/v1/projects/prj_00000000000000000000000000")
    body = response.json()
    for field in ("type", "title", "status", "detail", "instance", "code", "correlation_id"):
        assert field in body, f"missing {field}"
    assert body["status"] == response.status_code
    assert body["instance"] == "/v1/projects/prj_00000000000000000000000000"


async def test_problem_correlation_id_matches_the_header(client: AsyncClient) -> None:
    """The id in the body is what a user quotes; it has to be the one in the logs."""
    response = await client.get("/v1/projects/prj_00000000000000000000000000")
    assert response.json()["correlation_id"] == response.headers["X-Correlation-Id"]


async def test_validation_failure_reports_fields_without_echoing_input(
    client: AsyncClient,
) -> None:
    response = await client.post("/v1/auth/login", json={"email": "not-an-email"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_FAILED"
    assert any(e["field"] for e in body["errors"])
    # The offending value is deliberately not reflected back.
    assert "not-an-email" not in response.text


async def test_errors_do_not_leak_internals(client: AsyncClient) -> None:
    """detail is user-facing. An error message is an information-disclosure
    channel that is easy to forget, because it only fires when something is
    already wrong."""
    response = await client.get("/v1/projects/prj_00000000000000000000000000")
    text = response.text.lower()
    for leak in ("traceback", "select ", "sqlalchemy", "asyncpg", "/home/", "secret"):
        assert leak not in text, f"leaked {leak!r}"


# --------------------------------------------------------------------------
# rate limit headers
# --------------------------------------------------------------------------


async def test_rate_limit_headers_on_ordinary_responses(client: AsyncClient) -> None:
    """Clients should be able to back off before being refused, not only after."""
    response = await client.get("/v1/projects/prj_00000000000000000000000000")
    assert "RateLimit-Limit" in response.headers
    assert "RateLimit-Remaining" in response.headers
    assert "RateLimit-Reset" in response.headers


async def test_health_checks_are_exempt(client: AsyncClient) -> None:
    """Counted health checks would let a monitoring agent exhaust the limit for
    every user behind the same address."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert "RateLimit-Limit" not in response.headers
