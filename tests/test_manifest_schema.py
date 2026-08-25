"""Artifact manifest contract tests.

The manifest is the frozen boundary between the ML track and the application
track (docs/09 §11). Both the real S9 packager and the mock pipeline emit it,
and a drift between them would silently break the parallel-track plan -- so
these tests assert not only that a good manifest validates, but that the
mistakes we actually expect are rejected.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "packages" / "contracts" / "schemas" / "manifest.schema.json"
EXAMPLE_PATH = REPO_ROOT / "packages" / "contracts" / "examples" / "manifest.example.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.fixture
def manifest() -> dict[str, Any]:
    return copy.deepcopy(load(EXAMPLE_PATH))


def test_schema_is_itself_valid() -> None:
    """check_schema raises if the schema itself is malformed."""
    schema: dict[str, Any] = load(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    assert schema["$id"].endswith("/manifest/1.0.json")


def test_example_validates(validator: Draft202012Validator, manifest: dict[str, Any]) -> None:
    errors = sorted(validator.iter_errors(manifest), key=lambda e: e.json_path)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)


def test_version_is_pinned(manifest: dict[str, Any]) -> None:
    """1.0 is frozen in Phase 1; bumping it is a deliberate migration, not an edit."""
    assert manifest["manifest_version"] == "1.0"


# --------------------------------------------------------------------------
# rejections — each is a mistake we expect someone to actually make
# --------------------------------------------------------------------------


def invalid(validator: Draft202012Validator, doc: dict[str, Any]) -> bool:
    return bool(list(validator.iter_errors(doc)))


def test_rejects_float_duration(validator: Draft202012Validator, manifest: dict[str, Any]) -> None:
    """Invariant 2: integer milliseconds, never float seconds."""
    manifest["duration_ms"] = 612.48
    assert invalid(validator, manifest)


def test_rejects_unprefixed_ids(validator: Draft202012Validator, manifest: dict[str, Any]) -> None:
    manifest["project_id"] = "01HX8ZQ3M7N4P5R6S7T8V9W0XY"
    assert invalid(validator, manifest)


def test_rejects_unknown_property(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    """extra='forbid' at every boundary: a typo must fail, not be silently dropped."""
    manifest["playbck_hint"] = "webaudio"
    assert invalid(validator, manifest)


def test_rejects_video_flag_without_video_track(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    del manifest["video"]
    assert manifest["has_video"] is True
    assert invalid(validator, manifest)


def test_rejects_hls_hint_without_master_playlist(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    manifest["playback_hint"] = "hls"
    assert "master_playlist" not in manifest
    assert invalid(validator, manifest)


def test_rejects_speaker_without_faithful_track(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    """Invariant 6: the Faithful track is the default and the transcribed one."""
    del manifest["speakers"][0]["audio"]["faithful"]
    assert invalid(validator, manifest)


def test_rejects_confidence_out_of_range(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    manifest["speakers"][0]["mean_confidence"] = 1.4
    assert invalid(validator, manifest)


def test_rejects_freeform_warning_text(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    """Warnings are machine-readable tokens the UI maps to copy, not prose."""
    manifest["warnings"] = ["Speaker 2 had no face track!"]
    assert invalid(validator, manifest)


def test_accepts_partial_failure(validator: Draft202012Validator, manifest: dict[str, Any]) -> None:
    """Invariant 8: a failed speaker is a valid manifest, not a failed job."""
    manifest["speakers"][1]["extraction_ok"] = False
    manifest["warnings"].append("speaker_2_extraction_failed")
    assert not invalid(validator, manifest)


def test_accepts_audio_only_project(
    validator: Draft202012Validator, manifest: dict[str, Any]
) -> None:
    manifest["has_video"] = False
    del manifest["video"]
    for spk in manifest["speakers"]:
        spk["modality"] = "audio_only"
        spk.pop("thumbnail_url", None)
    assert not invalid(validator, manifest)
