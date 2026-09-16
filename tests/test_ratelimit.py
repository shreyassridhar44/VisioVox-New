"""Rate limiting (docs/11-api-spec.md §10).

Against a fake Redis rather than a real one: the behaviour worth pinning is the
window arithmetic and the failure policy, and both are far easier to drive
deterministically than by sleeping through real windows.

The test that matters most is `test_window_does_not_allow_a_double_burst`. A
fixed-window counter passes every other test here and still lets an attacker
send 2x the limit across a boundary, which for the 5-per-15-minutes login rule
is the difference between a working control and a decorative one.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from visiovox_api.config import Settings
from visiovox_api.ratelimit import RULES, Decision, RateLimiter, Rule, client_ip, enforce


class FakePipeline:
    def __init__(self, store: dict[str, int], fail: bool) -> None:
        self._store = store
        self._fail = fail
        self._ops: list[Any] = []

    def incrby(self, key: str, amount: int) -> None:
        self._ops.append(("incrby", key, amount))

    def expire(self, key: str, ttl: int) -> None:
        self._ops.append(("expire", key, ttl))

    def get(self, key: str) -> None:
        self._ops.append(("get", key))

    async def execute(self) -> list[Any]:
        if self._fail:
            raise ConnectionError("redis is down")
        out: list[Any] = []
        for op in self._ops:
            if op[0] == "incrby":
                self._store[op[1]] = self._store.get(op[1], 0) + op[2]
                out.append(self._store[op[1]])
            elif op[0] == "expire":
                out.append(True)
            else:
                out.append(self._store.get(op[1]))
        return out


class FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, int] = {}
        self.fail = fail

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.store, self.fail)

    async def aclose(self) -> None:
        return None


def _settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Controllable time, so windows can be crossed without sleeping."""
    state = {"now": 1_000_000.0}

    def fake_time() -> float:
        return state["now"]

    monkeypatch.setattr("visiovox_api.ratelimit.time.time", fake_time)
    return state


# --------------------------------------------------------------------------
# basic behaviour
# --------------------------------------------------------------------------


async def test_requests_under_the_limit_are_allowed(clock: Any) -> None:
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=5, window_seconds=60, scope="user")
    for _ in range(5):
        assert (await limiter.check(rule, "u1")).allowed is True


async def test_the_request_past_the_limit_is_refused(clock: Any) -> None:
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=3, window_seconds=60, scope="user")
    for _ in range(3):
        assert (await limiter.check(rule, "u1")).allowed is True
    assert (await limiter.check(rule, "u1")).allowed is False


async def test_keys_are_independent(clock: Any) -> None:
    """One user exhausting a limit must not affect another."""
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=2, window_seconds=60, scope="user")
    for _ in range(3):
        await limiter.check(rule, "noisy")
    assert (await limiter.check(rule, "quiet")).allowed is True


async def test_rules_are_independent(clock: Any) -> None:
    """Exhausting one endpoint's budget must not spend another's."""
    limiter = RateLimiter(FakeRedis(), _settings())
    a = Rule("a", limit=1, window_seconds=60, scope="user")
    b = Rule("b", limit=1, window_seconds=60, scope="user")
    await limiter.check(a, "u1")
    await limiter.check(a, "u1")
    assert (await limiter.check(b, "u1")).allowed is True


# --------------------------------------------------------------------------
# the window itself
# --------------------------------------------------------------------------


async def test_window_does_not_allow_a_double_burst(clock: Any) -> None:
    """The reason this is a sliding window and not a fixed one.

    Spend the whole limit at the very end of a window, then step just past the
    boundary. A fixed window resets and allows the full limit again - 2x in a
    couple of seconds. The weighted counter still sees almost all of the
    previous bucket and refuses.
    """
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=5, window_seconds=60, scope="ip+email")

    clock["now"] = 1_000_000.0 - (1_000_000.0 % 60) + 59.0  # 1s before the boundary
    for _ in range(5):
        assert (await limiter.check(rule, "attacker")).allowed is True

    clock["now"] += 2.0  # now just inside the next bucket
    assert (await limiter.check(rule, "attacker")).allowed is False


async def test_budget_returns_once_the_window_has_passed(clock: Any) -> None:
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=2, window_seconds=60, scope="user")
    for _ in range(3):
        await limiter.check(rule, "u1")
    clock["now"] += 130  # two clear windows later
    assert (await limiter.check(rule, "u1")).allowed is True


# --------------------------------------------------------------------------
# headers
# --------------------------------------------------------------------------


async def test_headers_describe_the_budget(clock: Any) -> None:
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=10, window_seconds=60, scope="user")
    decision = await limiter.check(rule, "u1")
    headers = decision.headers()
    assert headers["RateLimit-Limit"] == "10"
    assert headers["RateLimit-Remaining"] == "9"
    assert 0 < int(headers["RateLimit-Reset"]) <= 60
    assert "Retry-After" not in headers  # only on refusal


async def test_refusal_carries_retry_after(clock: Any) -> None:
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=1, window_seconds=60, scope="user")
    await limiter.check(rule, "u1")
    decision = await limiter.check(rule, "u1")
    assert decision.allowed is False
    assert "Retry-After" in decision.headers()


async def test_remaining_never_reports_negative(clock: Any) -> None:
    """A client reading a negative budget would compute nonsense backoff."""
    limiter = RateLimiter(FakeRedis(), _settings())
    rule = Rule("t", limit=1, window_seconds=60, scope="user")
    for _ in range(6):
        decision = await limiter.check(rule, "u1")
    assert int(decision.headers()["RateLimit-Remaining"]) == 0


def test_enforce_raises_429_with_headers() -> None:
    decision = Decision(False, 5, 0, 42, RULES["auth_login"])
    with pytest.raises(HTTPException) as exc:
        enforce(decision)
    assert exc.value.status_code == 429
    assert exc.value.headers is not None
    assert exc.value.headers["Retry-After"] == "42"


def test_enforce_is_a_noop_when_allowed() -> None:
    enforce(Decision(True, 5, 4, 42, RULES["auth_login"]))


# --------------------------------------------------------------------------
# failure policy
# --------------------------------------------------------------------------


async def test_redis_down_fails_open_by_default(clock: Any) -> None:
    """A dead cache must not take the API down with it."""
    limiter = RateLimiter(FakeRedis(fail=True), _settings(rate_limit_fail_open=True))
    decision = await limiter.check(RULES["reads"], "u1")
    assert decision.allowed is True


async def test_redis_down_can_be_made_to_fail_closed(clock: Any) -> None:
    limiter = RateLimiter(FakeRedis(fail=True), _settings(rate_limit_fail_open=False))
    with pytest.raises(ConnectionError):
        await limiter.check(RULES["reads"], "u1")


# --------------------------------------------------------------------------
# client identity
# --------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, headers: dict[str, str], host: str | None) -> None:
        self.headers = headers
        self.client = type("C", (), {"host": host})() if host else None


def test_client_ip_uses_the_socket_address_by_default() -> None:
    req = _FakeRequest({"cf-connecting-ip": "9.9.9.9"}, "10.0.0.5")
    assert client_ip(req, _settings()) == "10.0.0.5"  # type: ignore[arg-type]


def test_client_ip_honours_a_configured_proxy_header() -> None:
    """Behind a tunnel the socket address is the proxy, so every user would
    otherwise share one bucket."""
    req = _FakeRequest({"cf-connecting-ip": "9.9.9.9"}, "10.0.0.5")
    settings = _settings(trusted_client_ip_header="cf-connecting-ip")
    assert client_ip(req, settings) == "9.9.9.9"  # type: ignore[arg-type]


def test_client_ip_takes_the_first_entry_of_a_forwarded_chain() -> None:
    req = _FakeRequest({"x-forwarded-for": "9.9.9.9, 10.0.0.1, 10.0.0.2"}, "10.0.0.5")
    settings = _settings(trusted_client_ip_header="x-forwarded-for")
    assert client_ip(req, settings) == "9.9.9.9"  # type: ignore[arg-type]


def test_client_ip_ignores_an_unconfigured_header() -> None:
    """The spoofing guard: without configuration the header is not consulted, so
    a caller cannot mint a fresh identity per request."""
    req = _FakeRequest({"x-forwarded-for": "1.2.3.4"}, "10.0.0.5")
    assert client_ip(req, _settings()) == "10.0.0.5"  # type: ignore[arg-type]


def test_client_ip_survives_a_missing_peer() -> None:
    assert client_ip(_FakeRequest({}, None), _settings()) == "unknown"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the documented table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "limit", "window", "scope"),
    [
        ("auth_login", 5, 900, "ip+email"),
        ("auth_refresh", 30, 60, "user"),
        ("upload_init", 10, 3600, "user"),
        ("exports", 20, 3600, "user"),
        ("shared_token", 60, 60, "token+ip"),
        ("reads", 300, 60, "user"),
        ("global_ip", 1000, 60, "ip"),
    ],
)
def test_rules_match_the_api_spec(name: str, limit: int, window: int, scope: str) -> None:
    """docs/11 §10 is the contract; drift here is drift from the spec."""
    rule = RULES[name]
    assert (rule.limit, rule.window_seconds, rule.scope) == (limit, window, scope)
