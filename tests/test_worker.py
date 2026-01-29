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
        self.last_filepaths = list(filepaths)
        return {"task_id": "dummy_task", "collection_name": collection_name, "uploaded": len(filepaths)}

    async def get_documents(self, collection_name: str = "multimodal_data", vdb_endpoint: str = "http://milvus:19530"):
        return {
            "message": "ok",
            "total_documents": 2,
            "documents": [{"document_name": "a.pdf", "metadata": {}}, {"document_name": "b.pdf", "metadata": {}}],
        }

    async def delete_documents(
        self,
        collection_name: str = "multimodal_data",
        vdb_endpoint: str = "http://milvus:19530",
        document_names=None,
        file_names=None,
        documents=None,
    ):
        # Accept any of the common parameter names and return what was asked for.
        names = document_names or file_names or documents or []
        return {
            "message": "deleted",
            "total_documents": len(names),
            "documents": [{"document_name": n, "metadata": {}} for n in names],
        }

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

    # Ensure the worker preserved the original filename in the temp filepath passed to the ingestor.
    tmp_paths = getattr(worker.ingestor, "last_filepaths", [])
    assert len(tmp_paths) == 1
    assert tmp_paths[0].replace("\\", "/").endswith("/doc.txt")
    # And it should have cleaned up the temp file afterward.
    assert not Path(tmp_paths[0]).exists()


def test_documents_list_and_delete():
    worker = _load_worker_module()
    client = TestClient(worker.app)

    r1 = client.get(
        "/v1/documents",
        params={"collection_name": "multimodal_data", "vdb_endpoint": "http://milvus:19530"},
    )
    assert r1.status_code == 200
    js1 = r1.json()
    assert js1.get("total_documents") == 2
    assert any(d.get("document_name") == "a.pdf" for d in js1.get("documents", []))

    r2 = client.request(
        "DELETE",
        "/v1/documents",
        params={"collection_name": "multimodal_data", "vdb_endpoint": "http://milvus:19530"},
        json=["a.pdf", "b.pdf"],
    )
    assert r2.status_code == 200
    js2 = r2.json()
    assert js2.get("message") == "deleted"
    assert js2.get("total_documents") == 2

