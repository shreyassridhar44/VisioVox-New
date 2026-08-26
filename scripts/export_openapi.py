"""Export the OpenAPI schema to packages/contracts.

Committed rather than generated on demand, because the TypeScript client is
generated from it and CI diffs the two. A spec change that would break a client
therefore shows up as a failing check on the pull request that caused it,
instead of as a runtime error after deploy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from visiovox_api.main import app

DEST = Path(__file__).resolve().parents[1] / "packages" / "contracts" / "openapi.json"


def main() -> int:
    spec = app.openapi()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = len(spec.get("paths", {}))
    print(f"wrote {DEST.relative_to(Path.cwd())} ({paths} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
