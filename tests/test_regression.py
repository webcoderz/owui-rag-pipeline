"""
Regression tests for NVIDIA RAG pipeline and worker.
Documents fixed behaviors that must not regress in production (regulated systems).
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient


def _load_pipeline_module():
    import importlib.util
    from pathlib import Path
    mod_path = Path(__file__).resolve().parents[1] / "docker" / "pipelines" / "nvidia_ingest_bridge_pipe.py"
    spec = importlib.util.spec_from_file_location("nvidia_ingest_bridge_pipe", str(mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_worker_module():
    from tests.test_worker import _load_worker_module
    return _load_worker_module()


def _collect_streaming_text(response):
    """Synchronously collect streaming text from pipeline response (sync generator or dict)."""
    if isinstance(response, dict):
        return json.dumps(response)
    if hasattr(response, "__iter__") and not hasattr(response, "__aiter__"):
        chunks = []
        for i, chunk in enumerate(response):
            chunks.append(chunk)
            if "data: [DONE]" in chunk:
                break
            if i > 500:
                break
        return "".join(chunks)
    return ""


class TestPipeReturnTypeRegression:
    """REGRESSION: pipe() must return sync generator when stream=True (not async/coroutine).
    Open WebUI Pipelines expects sync iteration; returning async generator caused failures.
    """

    def test_pipe_returns_sync_generator_not_coroutine(self):
        """pipe(stream=True) returns a synchronous iterator, not an awaitable."""
        module = _load_pipeline_module()
        p = module.Pipeline()
        p.valves.OPENWEBUI_API_KEY = "token"
        body = {"messages": [{"role": "user", "content": "/commands"}], "stream": True}
        resp = p.pipe(body=body, __user__={"id": "u"})
        # Must be iterable synchronously (no await)
        assert hasattr(resp, "__iter__")
        assert not hasattr(resp, "__aiter__") or not asyncio.iscoroutine(resp)
        # Must be able to consume without asyncio.run
        chunks = list(resp)
        assert len(chunks) > 0
        full = "".join(chunks)
        assert "Commands:" in full
        assert "data: [DONE]" in full


class TestWorkerDeleteBodyFormatRegression:
    """REGRESSION: DELETE /v1/documents accepts JSON array body (NVIDIA Ingestor compatibility)."""

    def test_delete_accepts_json_array_body(self):
        """Worker must accept body as raw JSON array ['a.pdf','b.pdf']."""
        worker = _load_worker_module()
        client = TestClient(worker.app)
        resp = client.request(
            "DELETE",
            "/v1/documents",
            params={"collection_name": "multimodal_data", "vdb_endpoint": "http://milvus:19530"},
            json=["doc1.pdf", "doc2.pdf"],
        )
        assert resp.status_code == 200
        js = resp.json()
        assert js.get("message") == "deleted"
        assert js.get("total_documents") == 2


class TestWorkerReadyEndpointRegression:
    """REGRESSION: /ready endpoint must exist and validate generation backend config."""

    def test_ready_endpoint_exists(self):
        """Worker exposes /ready for production health checks."""
        worker = _load_worker_module()
        client = TestClient(worker.app)
        resp = client.get("/ready")
        assert resp.status_code in (200, 500)  # 500 if OPENAI config missing when backend=openai
        js = resp.json()
        assert "status" in js
        assert "generation_backend" in js


class TestWorkerCollectionsEndpointRegression:
    """REGRESSION: GET /collections returns list of Milvus collection names."""

    def test_collections_endpoint_returns_json(self):
        """Worker /collections returns {collections: [...]}."""
        worker = _load_worker_module()
        client = TestClient(worker.app)
        resp = client.get("/collections", params={"vdb_endpoint": "http://milvus:19530"})
        assert resp.status_code == 200
        js = resp.json()
        assert "collections" in js
        assert isinstance(js["collections"], list)


class TestPipelineCommandOutputRegression:
    """REGRESSION: Slash command outputs must contain expected help text."""

    def test_commands_contains_forget(self):
        """ /commands must mention /forget."""
        module = _load_pipeline_module()
        p = module.Pipeline()
        p.valves.OPENWEBUI_API_KEY = "token"
        body = {"messages": [{"role": "user", "content": "/commands"}], "stream": True}
        resp = p.pipe(body=body, __user__={"id": "u"})
        text = _collect_streaming_text(resp)
        assert "/forget" in text

    def test_commands_contains_ingest(self):
        """ /commands must mention /ingest."""
        module = _load_pipeline_module()
        p = module.Pipeline()
        p.valves.OPENWEBUI_API_KEY = "token"
        body = {"messages": [{"role": "user", "content": "/commands"}], "stream": True}
        resp = p.pipe(body=body, __user__={"id": "u"})
        text = _collect_streaming_text(resp)
        assert "/ingest" in text
