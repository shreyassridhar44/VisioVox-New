"""Workspace smoke tests.

Phase 0 has no application code yet. These assert that the workspace itself is
wired correctly, so a broken layout fails here rather than inside CI for some
unrelated change.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

WORKSPACE_PACKAGES = [
    "visiovox_api",
    "worker_cpu",
    "worker_gpu",
    "visiovox_client",
]


@pytest.mark.parametrize("module_name", WORKSPACE_PACKAGES)
def test_workspace_package_imports(module_name: str) -> None:
    """Every uv workspace member is importable and reports a version."""
    module = importlib.import_module(module_name)
    assert module.__version__ == "0.1.0"


def test_python_pin_matches_conventions() -> None:
    """The repo targets 3.12; a drift here silently changes ruff and mypy behaviour."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"
    assert pyproject["tool"]["ruff"]["target-version"] == "py312"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.12"


def test_mypy_runs_strict() -> None:
    """Strict mode is a stated convention, not a preference."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["mypy"]["strict"] is True
