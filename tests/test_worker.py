import sys
import types
import io
import json
from pathlib import Path
import importlib.util

from fastapi.testclient import TestClient


class _DummyRagResponse:
    def __init__(self):
        self.status_code = 200
        # minimal stream that looks like OpenAI-style SSE chunks
        self.generator = iter(
            [
                'data: {"choices":[{"delta":{"content":"hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
            ]
        )


class _DummyRag:
    def generate(self, messages=None, use_knowledge_base=False, collection_names=None, **kwargs):
        return _DummyRagResponse()


class _DummyIngestor:
    def create_collection(self, collection_name: str, vdb_endpoint: str):
        return {"ok": True, "collection_name": collection_name, "vdb_endpoint": vdb_endpoint}

    async def upload_documents(self, collection_name, vdb_endpoint, blocking, split_options, filepaths, generate_summary):
        return {"task_id": "dummy_task", "collection_name": collection_name, "uploaded": len(filepaths)}


def _load_worker_module():
    # Inject a dummy nvidia_rag module before import
    dummy_mod = types.ModuleType("nvidia_rag")
    dummy_mod.NvidiaRAG = _DummyRag
    dummy_mod.NvidiaRAGIngestor = _DummyIngestor
    sys.modules["nvidia_rag"] = dummy_mod

    worker_path = Path(__file__).resolve().parents[1] / "docker" / "worker" / "nvidia_rag_worker.py"
    spec = importlib.util.spec_from_file_location("nvidia_rag_worker", str(worker_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_health():
    worker = _load_worker_module()
    client = TestClient(worker.app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_generate_streaming():
    worker = _load_worker_module()
    client = TestClient(worker.app)

    payload = {"messages": [{"role": "user", "content": "hi"}], "collection_names": [], "use_knowledge_base": False}
    with client.stream("POST", "/generate", json=payload) as resp:
        assert resp.status_code == 200
        body = "".join([chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else chunk for chunk in resp.iter_text()])
        # ensure our dummy chunks are present
        assert "hello" in body
        assert "world" in body
        assert "data: [DONE]" in body


def test_ingest_upload(tmp_path):
    worker = _load_worker_module()
    client = TestClient(worker.app)

    f = tmp_path / "doc.txt"
    f.write_text("dummy content")

    files = {"file": ("doc.txt", f.read_bytes(), "text/plain")}
    data = {
        "collection_name": "test_library",
        "vdb_endpoint": "http://milvus:19530",
        "blocking": "true",
        "chunk_size": "256",
        "chunk_overlap": "32",
        "generate_summary": "false",
    }

    resp = client.post("/ingest", data=data, files=files)
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("task_id") == "dummy_task"
    assert js.get("collection_name") == "test_library"
    assert js.get("uploaded") == 1

