# 10 — Data Model

PostgreSQL 16. All IDs are ULIDs stored as `TEXT` (sortable by creation time, safe to expose,
no enumeration risk). All timestamps `TIMESTAMPTZ` in UTC.

---

## 1. ER overview

```
  users ──1:N── sessions
    │
    ├──1:N── refresh_tokens (families, rotation)
    ├──1:N── audit_events
    ├──1:N── api_keys
    └──1:N── projects
                │
                ├──1:1── jobs ──1:N── job_stages
                ├──1:N── speakers ──1:N── caption_segments
                ├──1:N── artifacts
                └──1:N── share_links
```

---

## 2. Schema

### users
```sql
CREATE TABLE users (
  id                TEXT PRIMARY KEY,
  email             CITEXT UNIQUE NOT NULL,
  email_verified_at TIMESTAMPTZ,
  password_hash     TEXT,                       -- NULL for OIDC-only accounts
  display_name      TEXT,
  avatar_url        TEXT,
  plan              TEXT NOT NULL DEFAULT 'free'
                    CHECK (plan IN ('free','pro','team')),
  data_region       TEXT NOT NULL DEFAULT 'eu'
                    CHECK (data_region IN ('eu','us','in')),   -- NFR-PRIV-06
  retention_days    INT  NOT NULL DEFAULT 30
                    CHECK (retention_days BETWEEN 1 AND 365),
  allow_training_use BOOLEAN NOT NULL DEFAULT FALSE,           -- NFR-PRIV-05
  persist_voiceprints BOOLEAN NOT NULL DEFAULT FALSE,          -- NFR-PRIV-02
  totp_secret_enc   BYTEA,
  totp_enabled_at   TIMESTAMPTZ,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at        TIMESTAMPTZ
);
CREATE INDEX ON users (deleted_at) WHERE deleted_at IS NOT NULL;
```

`allow_training_use` and `persist_voiceprints` **default FALSE**. The privacy-preserving behaviour
requires no action from the user (Charter principle 5).

### sessions & refresh_tokens
```sql
CREATE TABLE sessions (
  id          TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  user_agent  TEXT,
  ip_hash     TEXT,                              -- hashed, not raw (privacy)
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at  TIMESTAMPTZ NOT NULL,
  revoked_at  TIMESTAMPTZ
);

CREATE TABLE refresh_tokens (
  id          TEXT PRIMARY KEY,
  family_id   TEXT NOT NULL,                     -- rotation lineage
  user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  token_hash  TEXT NOT NULL,                     -- SHA-256; never store the token
  parent_id   TEXT REFERENCES refresh_tokens(id),
  used_at     TIMESTAMPTZ,
  expires_at  TIMESTAMPTZ NOT NULL,
  revoked_at  TIMESTAMPTZ
);
CREATE INDEX ON refresh_tokens (family_id);
CREATE UNIQUE INDEX ON refresh_tokens (token_hash);
```

**Reuse detection:** presenting a token whose `used_at` is already set means the token was stolen
and replayed. Revoke the entire `family_id`. This is the mechanism behind FR-ACC-04.

### projects
```sql
CREATE TABLE projects (
  id               TEXT PRIMARY KEY,
  user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title            TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','validating','queued','processing',
                                     'ready','failed','cancelled','expired')),
  source_key       TEXT,
  source_bytes     BIGINT,
  source_sha256    TEXT,
  duration_ms      INT,
  width            INT,
  height           INT,
  has_video        BOOLEAN NOT NULL DEFAULT TRUE,
  speaker_count    INT CHECK (speaker_count BETWEEN 0 AND 8),
  overlap_ratio    REAL,
  difficulty       TEXT CHECK (difficulty IN ('easy','moderate','hard')),
  manifest         JSONB,                         -- the artifact manifest
  warnings         TEXT[] NOT NULL DEFAULT '{}',
  rights_attested_at TIMESTAMPTZ,                 -- FR-UPL-08
  rights_attested_ip_hash TEXT,
  expires_at       TIMESTAMPTZ,                   -- retention
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at       TIMESTAMPTZ
);
CREATE INDEX ON projects (user_id, created_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX ON projects (status) WHERE status IN ('queued','processing');
CREATE INDEX ON projects (expires_at) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX ON projects (user_id, source_sha256) WHERE deleted_at IS NULL;  -- FR-UPL-09
```

### jobs & job_stages
```sql
CREATE TABLE jobs (
  id             TEXT PRIMARY KEY,
  project_id     TEXT NOT NULL UNIQUE REFERENCES projects(id) ON DELETE CASCADE,
  state          TEXT NOT NULL DEFAULT 'queued',
  current_stage  TEXT,
  progress_pct   SMALLINT NOT NULL DEFAULT 0 CHECK (progress_pct BETWEEN 0 AND 100),
  pipeline_version TEXT NOT NULL,
  model_versions JSONB NOT NULL DEFAULT '{}',
  attempt        SMALLINT NOT NULL DEFAULT 1,
  error_code     TEXT,
  error_message  TEXT,
  correlation_id TEXT NOT NULL,
  gpu_seconds    REAL,                            -- cost attribution
  queued_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at     TIMESTAMPTZ,
  finished_at    TIMESTAMPTZ
);

CREATE TABLE job_stages (
  id           BIGSERIAL PRIMARY KEY,
  job_id       TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  stage        TEXT NOT NULL,                     -- 'S0_ingest' … 'S9_package'
  state        TEXT NOT NULL,                     -- pending|running|done|failed|skipped
  attempt      SMALLINT NOT NULL DEFAULT 1,
  output_hash  TEXT,                              -- idempotency key
  metrics      JSONB,                             -- duration, peak VRAM, stage-specific
  error        TEXT,
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ,
  UNIQUE (job_id, stage, attempt)
);
CREATE INDEX ON job_stages (job_id, stage);
```

`job_stages` is what makes resume-from-last-completed-stage possible (NFR-REL-03) and supplies the
per-stage metrics for NFR-OBS-04.

### speakers
```sql
CREATE TABLE speakers (
  id                 TEXT PRIMARY KEY,
  project_id         TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  ordinal            SMALLINT NOT NULL,           -- 1..N, stable ordering
  label              TEXT NOT NULL,               -- user-renameable
  color_token        TEXT NOT NULL,
  modality           TEXT NOT NULL CHECK (modality IN ('audiovisual','audio_only','visual_only')),
  thumbnail_key      TEXT,
  speaking_ms        INT NOT NULL DEFAULT 0,
  speaking_ratio     REAL,
  binding_confidence REAL,
  mean_confidence    REAL,
  extraction_ok      BOOLEAN NOT NULL DEFAULT TRUE,
  -- biometric derivative: NULL unless users.persist_voiceprints (NFR-PRIV-01)
  voiceprint         BYTEA,
  voiceprint_expires_at TIMESTAMPTZ,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (project_id, ordinal)
);
```

`voiceprint` is nullable **and normally null**. It is written only when the user has explicitly
opted in, and it carries its own expiry independent of the project's. A nightly job nulls expired
voiceprints regardless of project state.

### artifacts
```sql
CREATE TABLE artifacts (
  id           TEXT PRIMARY KEY,
  project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  speaker_id   TEXT REFERENCES speakers(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL,   -- video|audio_mixed|audio_faithful|audio_natural|
                                -- hls_playlist|captions_vtt|captions_json|thumbnail|peaks|manifest
  storage_key  TEXT NOT NULL,
  bytes        BIGINT,
  content_type TEXT,
  checksum     TEXT,
  class        TEXT NOT NULL DEFAULT 'output'
               CHECK (class IN ('source','work','output','biometric')),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON artifacts (project_id, kind);
CREATE INDEX ON artifacts (class) WHERE class IN ('work','biometric');
CREATE UNIQUE INDEX ON artifacts (storage_key);
```

The `class` column drives the retention sweeper: `work` deleted after 7 days, `biometric` deleted at
job completion unless opted in. It also makes the **delete receipt** (NFR-PRIV-04) a simple query.

### caption_segments
```sql
CREATE TABLE caption_segments (
  id          BIGSERIAL PRIMARY KEY,
  project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  speaker_id  TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  start_ms    INT NOT NULL,
  end_ms      INT NOT NULL,
  text        TEXT NOT NULL,
  words       JSONB,                        -- [{w,s,e,c}]
  confidence  REAL,
  trust       REAL,                         -- from S8
  contested   BOOLEAN NOT NULL DEFAULT FALSE,  -- UNRESOLVED leakage
  CHECK (end_ms > start_ms)
);
CREATE INDEX ON caption_segments (project_id, speaker_id, start_ms);
CREATE INDEX ON caption_segments USING GIN (to_tsvector('english', text));   -- search
```

Captions live in Postgres **and** are exported as static VTT/JSON to storage. The DB copy powers
search and the interactive transcript; the static copy is what the player streams. They are written
in the same transaction as the manifest to keep them consistent.

### share_links
```sql
CREATE TABLE share_links (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  token_hash    TEXT NOT NULL UNIQUE,
  password_hash TEXT,
  allow_download BOOLEAN NOT NULL DEFAULT FALSE,
  view_count    INT NOT NULL DEFAULT 0,
  max_views     INT,
  expires_at    TIMESTAMPTZ NOT NULL,
  revoked_at    TIMESTAMPTZ,
  created_by    TEXT NOT NULL REFERENCES users(id),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Share tokens are **hashed**, like passwords. A leaked database dump must not yield working share
links.

### usage_counters
```sql
CREATE TABLE usage_counters (
  user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  period_start  DATE NOT NULL,
  uploads       INT NOT NULL DEFAULT 0,
  media_seconds INT NOT NULL DEFAULT 0,
  gpu_seconds   REAL NOT NULL DEFAULT 0,
  bytes_stored  BIGINT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, period_start)
);
```

Enforces FR-ACC-09 and NFR-SEC-05. `gpu_seconds` is the one that actually protects the budget.

### audit_events
```sql
CREATE TABLE audit_events (
  id          BIGSERIAL PRIMARY KEY,
  user_id     TEXT REFERENCES users(id) ON DELETE SET NULL,
  action      TEXT NOT NULL,     -- auth.login, auth.fail, upload.init, upload.reject,
                                 -- project.delete, share.create, artifact.download, authz.deny
  resource    TEXT,
  ip_hash     TEXT,
  user_agent  TEXT,
  metadata    JSONB,
  correlation_id TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON audit_events (user_id, created_at DESC);
CREATE INDEX ON audit_events (action, created_at DESC);
```

`ON DELETE SET NULL`, not `CASCADE`: account deletion must not erase the security audit trail. The
row survives with the identity removed — this satisfies both NFR-SEC-08 and GDPR erasure.

### deletion_receipts
```sql
CREATE TABLE deletion_receipts (
  id           TEXT PRIMARY KEY,
  user_id      TEXT,
  project_id   TEXT,
  object_keys  TEXT[] NOT NULL,
  row_counts   JSONB NOT NULL,
  requested_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  verified     BOOLEAN NOT NULL DEFAULT FALSE
);
```

Implements NFR-PRIV-04 — deletion is *provable*, not merely claimed. The `verified` flag is set by a
follow-up job that confirms the keys no longer resolve.

---

## 3. Invariants

| # | Invariant | Enforced by |
|---|---|---|
| I1 | A ready project has ≥ 1 speaker and a non-null manifest | Application + DB trigger |
| I2 | Every speaker of a ready project has an `audio_faithful` artifact (or `extraction_ok = false`) | Application check at S9 |
| I3 | All audio artifacts for a project have identical sample counts | Hard assert in S9 (docs/05 §12) |
| I4 | `voiceprint IS NULL` unless the owner opted in | Trigger + nightly sweeper |
| I5 | No artifact is publicly readable | Bucket policy + IaC test |
| I6 | Caption `start_ms`/`end_ms` fall within `projects.duration_ms` | CHECK + application |
| I7 | A project's storage keys are all under its own prefix | Application; verified by the delete job |

I3 is worth restating: a one-sample length difference between speaker tracks becomes accumulating
A/V drift in the player. It is cheap to assert and expensive to debug from a user report.

---

## 4. Migrations

Alembic. Rules:

- Every migration has a working `downgrade()`. No exceptions.
- **Expand → migrate → contract** for breaking changes: add the new column nullable, backfill in
  batches, switch reads, then drop the old column in a later release. Never in one deploy.
- No blocking DDL on large tables during business hours; use `CREATE INDEX CONCURRENTLY`.
- Migrations are tested against a production-sized seed in CI, not just an empty schema.
- One migration per PR.

---

## 5. Retention jobs

| Job | Schedule | Action |
|---|---|---|
| `sweep_work_artifacts` | hourly | Delete `class='work'` older than 7 days |
| `sweep_biometrics` | every 15 min | Null `voiceprint` past expiry; delete `class='biometric'` |
| `expire_projects` | daily | Mark expired past `expires_at`, delete objects, write receipt |
| `warn_expiring` | daily | Email users 3 days before expiry (FR-PRJ-05) |
| `purge_deleted_users` | daily | Hard-delete accounts soft-deleted > 24 h ago |
| `verify_deletions` | daily | Confirm receipt keys no longer resolve; set `verified` |
| `rotate_audit` | monthly | Archive audit events older than 1 year to cold storage |

---

## 6. Performance notes

- Partial indexes on `deleted_at IS NULL` — the vast majority of queries filter this way.
- `manifest` as JSONB avoids a join-heavy read on the hottest path (project load). It duplicates data
  in `speakers`/`artifacts`; the manifest is the read model, the tables are the write model. Written
  in one transaction.
- Connection pooling via PgBouncer in transaction mode (FastAPI async + many short queries).
- `caption_segments` is the largest table (~500 rows per project-speaker). Partition by
  `project_id` hash only if it exceeds ~50 M rows — unlikely at the target scale, so not done now.
- The GIN index on caption text supports FR-PRJ-04 search without a separate search service.
