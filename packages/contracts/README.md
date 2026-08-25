# contracts

Source of truth for the Web↔API↔Worker boundary (`docs/09-system-design.md` §11).

- `openapi.yaml` — control-plane API
- `schemas/manifest.schema.json` — artifact manifest

Clients in `packages/ts-client` and `packages/py-client` are **generated** from these and are
never hand-edited. A contract change that breaks a client fails CI.

The manifest is frozen in Phase 1 (week 3); the mock and real pipelines both emit it, which is
what keeps the two development tracks honest.
