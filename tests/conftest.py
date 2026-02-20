"""
Shared pytest fixtures and configuration for NVIDIA RAG pipeline test suite.
Supports unit, integration, regression, and property tests.
"""

from pathlib import Path

import pytest


@pytest.fixture
def worker_module():
    """Load the worker module with nvidia_rag mocked (for tests that need it)."""
    # Import is delayed to avoid polluting sys.modules for tests that don't need it
    from tests.test_worker import _load_worker_module
    return _load_worker_module()


@pytest.fixture
def pipeline_module():
    """Load the pipeline module."""
    import importlib.util
    mod_path = Path(__file__).resolve().parents[1] / "docker" / "pipelines" / "nvidia_ingest_bridge_pipe.py"
    spec = importlib.util.spec_from_file_location("nvidia_ingest_bridge_pipe", str(mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module
