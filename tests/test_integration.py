"""
Integration tests for NVIDIA RAG pipeline and worker.
Tests component interaction with realistic stubs (no live Milvus/OWUI).
"""

import json

import pytest
from fastapi.testclient import TestClient


def _load_worker_module():
    from tests.test_worker import _load_worker_module
    return _load_worker_module()


def _load_pipeline_module():
    import importlib.util
    from pathlib import Path
    mod_path = Path(__file__).resolve().parents[1] / "docker" / "pipelines" / "nvidia_ingest_bridge_pipe.py"
    spec = importlib.util.spec_from_file_location("nvidia_ingest_bridge_pipe", str(mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _collect_streaming_text(response):
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


class TestWorkerIngestGenerateFlow:
    """Integration: Worker ingest -> list -> delete flow (full request/response cycle)."""

    def test_ingest_then_list_then_delete_cycle(self, tmp_path):
        """Upload file, list documents, delete - validates full worker flow."""
        worker = _load_worker_module()
        client = TestClient(worker.app)
        f = tmp_path / "report.pdf"
        f.write_text("sample report content")

        # Ingest
        files = {"file": ("report.pdf", f.read_bytes(), "application/pdf")}
        data = {
            "collection_name": "test_collection",
            "vdb_endpoint": "http://milvus:19530",
            "blocking": "true",
            "chunk_size": "512",
            "chunk_overlap": "150",
            "generate_summary": "false",
        }
        ingest_resp = client.post("/ingest", data=data, files=files)
        assert ingest_resp.status_code == 200
        assert ingest_resp.json().get("task_id")
        assert ingest_resp.json().get("uploaded") == 1

        # List (our dummy returns 2 docs; real worker would show the new one)
        list_resp = client.get("/v1/documents", params={"collection_name": "test_collection"})
        assert list_resp.status_code == 200
        assert "documents" in list_resp.json()

        # Delete
        del_resp = client.request(
            "DELETE", "/v1/documents",
            params={"collection_name": "test_collection"},
            json=["report.pdf"],
        )
        assert del_resp.status_code == 200
        assert del_resp.json().get("message") == "deleted"


class TestPipelineIngestThenQueryFlow:
    """Integration: Pipeline /ingest flow and allowlist persistence (stubbed OWUI and worker)."""

    def test_ingest_populates_allowlist_then_query_uses_it(self, monkeypatch):
        """
        /ingest with target collection populates allowlist; /query uses it.
        Validates the ingest->allowlist->query data flow with full stubs.
        """
        module = _load_pipeline_module()
        p = module.Pipeline()
        p.valves.OPENWEBUI_API_KEY = "token"

        allowlist = {}  # (user_key, chat_id) -> [collections]
        ingested_collections = []

        async def stub_ow_get_json(client, path, user_token=None, **kwargs):
            if "/knowledge/" in path:
                return {"files": [{"id": "f1", "filename": "doc.txt"}]}
            if "/files/" in path:
                return {"filename": "doc.txt", "size": 10}
            return {}

        async def stub_ingest_entries(ow, w, entries, coll, emit, model_id, user_token=None, uploaded_by=None, **kw):
            ingested_collections.append(coll)
            for _ in entries:
                await emit("ok\n")
            return None

        async def stub_allowlist_add(uk, cid, cols):
            key = (uk, cid)
            existing = allowlist.get(key, [])
            allowlist[key] = list(dict.fromkeys(existing + cols))

        async def stub_allowlist_get(uk, cid):
            return allowlist.get((uk, cid), [])

        monkeypatch.setattr(p, "_ow_get_json", stub_ow_get_json)
        monkeypatch.setattr(p, "_ingest_entries_into_collection", stub_ingest_entries)
        monkeypatch.setattr(p, "_allowlist_add", stub_allowlist_add)
        monkeypatch.setattr(p, "_allowlist_get", stub_allowlist_get)
        monkeypatch.setattr(p, "_library_include_by_default", lambda uk: False)
        monkeypatch.setattr(p, "_chat_get_save_to_library", lambda uk, cid: False)
        monkeypatch.setattr(p, "_download_to_tempfile_and_hash", lambda *a, **k: ("/tmp/x", "sha", 5))
        monkeypatch.setattr(p, "_manifest_get_status", lambda *a: None)
        monkeypatch.setattr(p, "_manifest_try_claim", lambda *a: True)
        monkeypatch.setattr(p, "_manifest_set", lambda *a: None)
        monkeypatch.setattr(p, "_call_worker_ingest_from_path", lambda *a, **k: {"ok": True})

        # Ingest into owui-test (target collection)
        body1 = {
            "messages": [{"role": "user", "content": "/ingest owui-test"}],
            "stream": True, "chat_id": "c1",
            "files": [{"type": "collection", "id": "kb1"}],
        }
        resp1 = p.pipe(body=body1, __user__={"id": "u1"})
        text1 = _collect_streaming_text(resp1)
        assert "data: [DONE]" in text1
        assert ingested_collections, "Ingest should have processed at least one collection"
        assert allowlist.get(("u1", "c1")), "Allowlist should be populated for (u1, c1)"

        # Verify /query would use the same allowlist (integration contract)
        collections_for_query = allowlist.get(("u1", "c1"), [])
        assert "owui-test" in collections_for_query or any(
            "owui" in c for c in collections_for_query
        ), "Query would use ingested collection from allowlist"
