import sys
import types
import json
import asyncio
import importlib.util
from pathlib import Path

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient


class DummyConn:
    def __init__(self):
        self._store = {
            "ingest_manifest": {},
            "chat_allowlist": set(),
            "user_library_settings": {},
            "chat_settings": {},
        }

    async def execute(self, *args, **kwargs):
        return "OK"

    async def fetchrow(self, query, *params):
        if "FROM ingest_manifest" in query:
            file_id, collection, sha = params
            key = (file_id, collection, sha)
            status = self._store["ingest_manifest"].get(key)
            return {"status": status} if status else None
        if "FROM user_library_settings" in query:
            user_key = params[0]
            row = self._store["user_library_settings"].get(user_key)
            if row:
                return {"enabled": row["enabled"], "include_by_default": row["include_by_default"]}
            return None
        if "FROM chat_settings" in query:
            user_key, chat_id = params
            val = self._store["chat_settings"].get((user_key, chat_id))
            return {"save_to_library": val} if val is not None else None
        return None

    async def fetch(self, query, *params):
        if "FROM chat_allowlist" in query:
            user_key, chat_id = params
            return [{"collection_name": c} for (u, ch, c) in self._store["chat_allowlist"] if u == user_key and ch == chat_id]
        return []

    async def executemany(self, query, values):
        if "INSERT INTO chat_allowlist" in query:
            for (user_key, chat_id, c, _now) in values:
                self._store["chat_allowlist"].add((user_key, chat_id, c))
        return "OK"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyPool:
    def __init__(self):
        self.conn = DummyConn()

    class _AcquireCtx:
        def __init__(self, conn):
            self._conn = conn
        async def __aenter__(self):
            return self._conn
        async def __aexit__(self, exc_type, exc, tb):
            return False

    def acquire(self):
        return self._AcquireCtx(self.conn)


class DummyStreamResponse:
    def __init__(self, body_bytes: bytes):
        self._body = body_bytes
        self.status_code = 200
        self.headers = {"content-length": str(len(body_bytes))}
    def raise_for_status(self):
        return None
    async def aiter_lines(self):
        for line in self._body.splitlines():
            yield line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else line

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        yield self._body


class DummyHTTPXClient:
    def __init__(self, timeout=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, headers=None, timeout=None, params=None):
        class R:
            def __init__(self, json_data=None, content=None, status_code=200):
                self._json = json_data
                self.content = content
                self.status_code = status_code
            def json(self):
                return self._json or {}
            def raise_for_status(self):
                return None
        # KB meta and file meta
        if "/api/v1/knowledge/" in url:
            return R(json_data={"files": [{"id": "fid1", "filename": "doc.txt"}]})
        if "/api/v1/files/" in url and url.endswith("/content"):
            # should be streamed via .stream(), not here
            return R(content=b"content")
        if "/api/v1/files/" in url:
            return R(json_data={"filename": "upload.txt", "size": 10})
        return R(json_data={})

    async def post(self, url, data=None, files=None, json=None, timeout=None):
        if url.endswith("/ingest"):
            return type("R", (), {"status_code": 200, "json": lambda: {"task_id": "t1"}, "raise_for_status": lambda: None})()
        return type("R", (), {"status_code": 200, "json": lambda: {}, "raise_for_status": lambda: None})()

    def stream(self, method, url, json=None, headers=None, data=None, files=None, timeout=None, params=None):
        # OWUI file download
        if method == "GET" and "/api/v1/files/" in url and url.endswith("/content"):
            return DummyStreamResponse(b"hello world")
        # Worker generate stream
        if method == "POST" and url.endswith("/generate"):
            lines = [
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            body = b"".join(lines)
            return DummyStreamResponse(body)
        return DummyStreamResponse(b"")


def _load_pipeline_module():
    # Load pipeline module by path
    mod_path = Path(__file__).resolve().parents[1] / "docker" / "pipelines" / "nvidia_ingest_bridge_pipe.py"
    spec = importlib.util.spec_from_file_location("nvidia_ingest_bridge_pipe", str(mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


async def _collect_streaming_text(response):
    # The pipeline `pipe()` may return:
    # - a dict (non-stream)
    # - a sync generator yielding SSE lines (stream)
    # - a StreamingResponse (older behavior)
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

    # Fallback: wrap StreamingResponse inside a tiny ASGI app to collect output
    app = FastAPI()

    @app.get("/pipe")
    async def serve():
        return response

    with TestClient(app) as client:
        with client.stream("GET", "/pipe", timeout=5.0) as r:
            chunks = []
            for i, chunk in enumerate(r.iter_text()):
                chunks.append(chunk)
                if "data: [DONE]" in chunk:
                    break
                if i > 200:
                    break
            return "".join(chunks)


def test_pipe_no_api_key(monkeypatch):
    module = _load_pipeline_module()
    p = module.Pipeline()
    p.valves.OPENWEBUI_API_KEY = ""

    resp = anyio.run(lambda: p.pipe(body={}, __user__={}))
    text = anyio.run(lambda: _collect_streaming_text(resp))
    assert "OPENWEBUI_API_KEY is not set" in text
    assert "data: [DONE]" in text


def test_pipe_commands(monkeypatch):
    module = _load_pipeline_module()
    p = module.Pipeline()
    p.valves.OPENWEBUI_API_KEY = "token"

    body = {"messages": [{"role": "user", "content": "/commands"}], "stream": True}
    resp = anyio.run(lambda: p.pipe(body=body, __user__={"id": "u"}))
    text = anyio.run(lambda: _collect_streaming_text(resp))
    assert "Commands:" in text
    assert "data: [DONE]" in text


def test_pipe_library_toggle(monkeypatch):
    module = _load_pipeline_module()
    p = module.Pipeline()
    p.valves.OPENWEBUI_API_KEY = "token"

    # Stub chat settings to keep test isolated
    state = {"val": False}

    async def stub_get(user_key: str, chat_id: str):
        return state["val"]

    async def stub_set(user_key: str, chat_id: str, value: bool):
        state["val"] = bool(value)

    monkeypatch.setattr(p, "_chat_get_save_to_library", stub_get)
    monkeypatch.setattr(p, "_chat_set_save_to_library", stub_set)

    body_on = {"messages": [{"role": "user", "content": "/library on"}], "stream": True, "chat_id": "c1"}
    resp_on = anyio.run(lambda: p.pipe(body=body_on, __user__={"id": "u"}))
    text_on = anyio.run(lambda: _collect_streaming_text(resp_on))
    assert "save new ingests" in text_on.lower()

    body_status = {"messages": [{"role": "user", "content": "/library"}], "stream": True, "chat_id": "c1"}
    resp_st = anyio.run(lambda: p.pipe(body=body_status, __user__={"id": "u"}))
    text_st = anyio.run(lambda: _collect_streaming_text(resp_st))
    assert "library setting" in text_st.lower()


def test_pipe_ingest_and_generate(monkeypatch, tmp_path):
    module = _load_pipeline_module()
    p = module.Pipeline()

    # Bypass DB usage and networked helpers with simple stubs
    async def stub_allowlist_get(user_key: str, chat_id: str):
        return []
    async def stub_allowlist_add(user_key: str, chat_id: str, collections):
        return None
    async def stub_library_include_by_default(user_key: str):
        return False
    async def stub_chat_get_save_to_library(user_key: str, chat_id: str):
        return False
    async def stub_ow_get_json(client, path: str):
        if path.startswith("/api/v1/knowledge/"):
            return {"files": [{"id": "fid1", "filename": "doc.txt"}]}
        if path.startswith("/api/v1/files/"):
            return {"filename": "upload.txt", "size": 10}
        return {}
    async def stub_download_to_tempfile_and_hash(client, file_id: str, filename: str, emit, model_id: str):
        f = tmp_path / filename
        f.write_text("dummy")
        return str(f), "sha123", 5
    async def stub_call_worker_ingest_from_path(client, collection_name: str, filename: str, tmp_path_str: str):
        return {"ok": True, "collection_name": collection_name}
    async def stub_manifest_get_status(file_id: str, collection: str, sha: str):
        return None
    async def stub_manifest_try_claim(file_id: str, collection: str, sha: str):
        return True
    async def stub_manifest_set(file_id: str, collection: str, sha: str, status: str):
        return None
    async def stub_stream_worker_generate(client, messages, collection_names):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(p, "_allowlist_get", stub_allowlist_get)
    monkeypatch.setattr(p, "_allowlist_add", stub_allowlist_add)
    monkeypatch.setattr(p, "_library_include_by_default", stub_library_include_by_default)
    monkeypatch.setattr(p, "_chat_get_save_to_library", stub_chat_get_save_to_library)
    monkeypatch.setattr(p, "_ow_get_json", stub_ow_get_json)
    monkeypatch.setattr(p, "_download_to_tempfile_and_hash", stub_download_to_tempfile_and_hash)
    monkeypatch.setattr(p, "_call_worker_ingest_from_path", stub_call_worker_ingest_from_path)
    monkeypatch.setattr(p, "_manifest_get_status", stub_manifest_get_status)
    monkeypatch.setattr(p, "_manifest_try_claim", stub_manifest_try_claim)
    monkeypatch.setattr(p, "_manifest_set", stub_manifest_set)
    monkeypatch.setattr(p, "_stream_worker_generate", stub_stream_worker_generate)

    p.valves.OPENWEBUI_API_KEY = "token"
    p.valves.OPENWEBUI_BASE_URL = "http://open-webui:8080"
    p.valves.NVIDIA_WORKER_URL = "http://nvidia-rag-worker:8123"
    p.valves.VDB_ENDPOINT = "http://milvus:19530"
    p.valves.MAX_PARALLEL_FILE_INGEST = 2

    body = {
        "model": "nvidia-rag-auto-ingest",
        "files": [
            {"type": "collection", "id": "kb1"},
            {"type": "file", "id": "file-123"},
        ],
        "messages": [{"role": "user", "content": "hello"}],
        "chat_id": "chat-1",
    }

    resp = anyio.run(lambda: p.pipe(body=body, __user__={"id": "user@example.com"}))
    text = anyio.run(lambda: _collect_streaming_text(resp))

    # status emissions
    assert "Attachments detected" in text
    assert "Knowledge `kb1`" in text
    assert "Ingestion complete" in text
    # streaming output
    assert "data: [DONE]" in text

