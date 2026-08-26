"""Mock pipeline and storage-key tests.

The mock exists to take the application off the ML critical path, and that only
holds if what it emits is shape-identical to the real S9 output. So it is
validated against the *same* frozen schema — if the two drift, this fails in CI
rather than surfacing during integration months later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from visiovox_api.storage import object_key, safe_filename
from worker_cpu.mock_pipeline import STAGES, build_manifest, plan_stages

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "packages" / "contracts" / "schemas" / "manifest.schema.json"

PROJECT_ID = "prj_01HX8ZQ3M7N4P5R6S7T8V9W0XY"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text()), format_checker=FormatChecker())


# --------------------------------------------------------------------------
# manifest conformance — the contract that keeps both tracks aligned
# --------------------------------------------------------------------------


@pytest.mark.parametrize("duration_ms", [30_000, 600_000, 3_600_000])
def test_mock_manifest_matches_the_frozen_schema(
    validator: Draft202012Validator, duration_ms: int
) -> None:
    manifest = build_manifest(PROJECT_ID, duration_ms)
    errors = sorted(validator.iter_errors(manifest), key=lambda e: e.json_path)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)


def test_audio_only_mock_is_valid(validator: Draft202012Validator) -> None:
    manifest = build_manifest(PROJECT_ID, 60_000, has_video=False)
    assert "video" not in manifest
    assert not list(validator.iter_errors(manifest))


def test_three_speaker_mock_exercises_mixed_modality(
    validator: Draft202012Validator,
) -> None:
    """The UI should meet a project where one speaker has no face, not just the easy case."""
    manifest = build_manifest(PROJECT_ID, 60_000, speaker_count=3)
    speakers = manifest["speakers"]
    assert isinstance(speakers, list)
    modalities = {s["modality"] for s in speakers}
    assert "audio_only" in modalities
    warnings = manifest["warnings"]
    assert isinstance(warnings, list)
    assert any("no_face_track" in w for w in warnings)
    assert not list(validator.iter_errors(manifest))


def test_mock_is_deterministic_per_project() -> None:
    """A retry must look identical, or UI bugs become unreproducible."""
    a = build_manifest(PROJECT_ID, 120_000)
    b = build_manifest(PROJECT_ID, 120_000)
    assert a["speakers"] == b["speakers"]
    assert a["overlap_ratio"] == b["overlap_ratio"]


def test_different_projects_differ() -> None:
    a = build_manifest("prj_01HX8ZQ3M7N4P5R6S7T8V9W0X1", 120_000)
    b = build_manifest("prj_01HX8ZQ3M7N4P5R6S7T8V9W0X2", 120_000)
    assert a["speakers"] != b["speakers"]


def test_duration_is_passed_through_as_integer_ms() -> None:
    manifest = build_manifest(PROJECT_ID, 612_480)
    assert manifest["duration_ms"] == 612_480
    assert isinstance(manifest["duration_ms"], int)


# --------------------------------------------------------------------------
# stage timing
# --------------------------------------------------------------------------


def test_stage_shares_sum_to_one() -> None:
    assert sum(share for _, _, share in STAGES) == pytest.approx(1.0, abs=1e-6)


def test_progress_is_monotonic_and_ends_at_100() -> None:
    plans = plan_stages(600_000)
    values = [p.progress_after for p in plans]
    assert values == sorted(values), "progress went backwards"
    assert values[-1] == 100, "progress did not reach 100"
    assert all(0 <= v <= 100 for v in values)


def test_video_analysis_dominates_runtime() -> None:
    """docs/05 §13: video analysis is 28% of runtime, extraction 17%.

    Pinned because a UI tuned against a uniform fake pipeline feels wrong the
    first time it meets a real job.
    """
    plans = {p.stage: p.duration_ms for p in plan_stages(600_000)}
    assert plans["S2B_video"] > plans["S5_extract"]
    assert plans["S2B_video"] > plans["S7_transcribe"]


def test_every_stage_gets_nonzero_time_even_for_tiny_media() -> None:
    plans = plan_stages(1_000)
    assert all(p.duration_ms >= 1 for p in plans)
    assert len(plans) == len(STAGES)


# --------------------------------------------------------------------------
# storage keys
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("holiday.mp4", "holiday.mp4"),
        ("../../etc/passwd", "passwd"),  # path segments are dropped, not escaped
        ("/absolute/path/x.mp4", "x.mp4"),
        ("C:\\Users\\me\\clip.mov", "clip.mov"),
        ("weird name (1).mkv", "weird_name__1_.mkv"),
        ("", "upload"),
        ("...", "upload"),
    ],
)
def test_filenames_are_sanitised(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


def test_filename_length_is_capped() -> None:
    assert len(safe_filename("a" * 500 + ".mp4")) <= 120


def test_object_key_is_namespaced_by_user() -> None:
    """A key bug must not be able to address another user's object."""
    key = object_key("usr_A", "prj_B", "../../../other.mp4")
    assert key.startswith("u/usr_A/p/prj_B/source/")
    assert ".." not in key


def test_traversal_cannot_escape_the_prefix() -> None:
    for hostile in ("../../x", "..\\..\\x", "a/../../b", "%2e%2e/x"):
        key = object_key("usr_A", "prj_B", hostile)
        assert key.count("u/usr_A/p/prj_B/source/") == 1
        assert ".." not in key
