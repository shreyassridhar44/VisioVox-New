"""Append-only audit log (docs/15-security.md §10).

**Logged:** authn success/failure, authz denial, upload init/complete/reject, job
lifecycle, artifact download, share create/access/revoke, deletion, settings
changes, admin actions.

**Never logged:** passwords, tokens (even hashed), full IPs, media content,
transcript text, speaker embeddings, email bodies.

The second list is the reason this is a module rather than a `logger.info` call
at each site. `record()` takes structured arguments and hashes the IP itself, so
the safe thing is the easy thing and the unsafe thing requires effort.

Writes are best-effort: an audit failure must not fail the request that was
otherwise fine. That is a real trade - a dropped audit row is a gap in the
security record - and it is taken because the alternative is that a full disk or
a schema drift turns into an outage of the whole API.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .models import AuditEvent

logger = logging.getLogger(__name__)

# Actions, named once so queries and dashboards have a stable vocabulary.
LOGIN_SUCCEEDED = "auth.login.succeeded"
LOGIN_FAILED = "auth.login.failed"
LOGOUT = "auth.logout"
REGISTERED = "auth.registered"
TOKEN_REUSE_DETECTED = "auth.token_reuse_detected"  # noqa: S105 - an action name, not a credential
AUTHZ_DENIED = "authz.denied"
RATE_LIMITED = "limit.rate_limited"
QUOTA_EXCEEDED = "limit.quota_exceeded"
UPLOAD_INIT = "upload.init"
UPLOAD_COMPLETED = "upload.completed"
UPLOAD_REJECTED = "upload.rejected"
JOB_QUEUED = "job.queued"
JOB_FINISHED = "job.finished"
PROJECT_DELETED = "project.deleted"
SETTINGS_CHANGED = "account.settings_changed"


def hash_ip(ip: str | None, salt: str) -> str | None:
    """Hash an address before it is stored.

    Salted because the IPv4 space is small enough to exhaust: an unsalted
    SHA-256 of an address is reversible with a rainbow table in minutes, which
    would make "hashed" purely decorative.
    """
    if not ip:
        return None
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()


async def record(
    session: AsyncSession,
    action: str,
    *,
    user_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    outcome: str = "success",
    ip: str | None = None,
    ip_salt: str = "",
    user_agent: str | None = None,
    correlation_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append one event. Never raises.

    `detail` is for small structured facts - a reason code, a byte count, a
    stage name. It must never carry media, transcript text or anything derived
    from a person's voice or face.
    """
    try:
        session.add(
            AuditEvent(
                user_id=user_id,
                action=action,
                target_type=target_type,
                target_id=target_id,
                outcome=outcome,
                ip_hash=hash_ip(ip, ip_salt),
                # Capped: it is attacker-controlled and ends up in a text column.
                user_agent=(user_agent or "")[:400] or None,
                correlation_id=correlation_id,
                detail=detail,
            )
        )
        await session.flush()
    except Exception:
        logger.warning("audit write failed for action=%s", action, exc_info=True)
