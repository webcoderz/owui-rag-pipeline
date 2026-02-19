import asyncio
import inspect
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import Body, FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# ----
# IMPORTANT (NVIDIA RAG SDK dependency):
# The `nvidia_rag` package may initialize MinIO-related internals at import time.
# In Docker, we expect MINIO_* env vars to be set (see `docker-compose.yaml`).
#
# For unit tests / CI (and for lightweight local imports), do NOT hard-fail at import
# time if MINIO_* isn't set. Instead, apply safe Docker-friendly defaults.
# ----

def _set_default_env(name: str, default: str) -> None:
    v = (os.getenv(name) or "").strip()
    if not v and default:
        os.environ[name] = default

# Docker-friendly defaults (avoid localhost inside containers).
_set_default_env("MINIO_ENDPOINT", "minio:9010")
_set_default_env("MINIO_ACCESSKEY", "minioadmin")
_set_default_env("MINIO_SECRETKEY", "minioadmin")
_set_default_env("VDB_ENDPOINT", "http://milvus:19530")
# NVIDIA RAG Blueprint uses APP_VECTORSTORE_URL for Milvus; set from VDB_ENDPOINT so SDK sees it at init.
_set_default_env("APP_VECTORSTORE_URL", os.getenv("VDB_ENDPOINT", "http://milvus:19530"))

# NVIDIA client SDK (py3.12+)
from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

app = FastAPI(title="nvidia-rag-worker", version="1.0.0")

rag = NvidiaRAG()
ingestor = NvidiaRAGIngestor()


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs down to only what `fn` accepts.
    This keeps us compatible across NVIDIA RAG SDK versions.
    """
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_ingestor_first(method_names: List[str], **kwargs):
    """
    Try a list of method names on `ingestor`, returning the first successful call result.
    Raises the last exception if all candidate methods fail.
    """
    last_exc: Optional[Exception] = None
    for name in method_names:
        fn = getattr(ingestor, name, None)
        if not fn:
            continue
        try:
            filtered = _filter_kwargs_for_callable(fn, kwargs)
            return await _maybe_await(fn(**filtered))
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Delete/list documents is not supported by the installed nvidia_rag SDK.")


def _safe_upload_filename(name: Optional[str]) -> str:
    """
    Sanitize an upload filename for use as a temp filepath.
    We keep the original basename so document deletion by name is possible later.
    """
    raw = (name or "").strip()
    raw = raw.replace("\\", "/")
    base = os.path.basename(raw) if raw else ""
    base = base.strip() or "upload"

    # Allow only a conservative set of characters.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ")
    cleaned = "".join(ch if ch in allowed else "_" for ch in base)
    cleaned = " ".join(cleaned.split()).strip()  # normalize whitespace
    cleaned = cleaned.replace(" ", "_")
    if not cleaned:
        cleaned = "upload"

    # Keep filename bounded (avoid Windows/path issues).
    if len(cleaned) > 180:
        root, ext = os.path.splitext(cleaned)
        cleaned = root[:160] + ext[:20]

    return cleaned


def _get_last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _normalize_openai_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    # Accept either ".../v1" or bare host and normalize to ".../v1"
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _openai_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key or 'dummy'}"}


def _openai_models_url(base_url: str) -> str:
    base = _normalize_openai_base_url(base_url)
    return f"{base}/models" if base else ""


def _extract_search_results_text(citations_obj: Any, max_chars: int = 12000) -> str:
    """
    Best-effort extraction of plain-text context from nvidia_rag citations/search results.
    We only include textual content and truncate to keep prompts bounded.
    """
    results = None
    if citations_obj is None:
        return ""
    if isinstance(citations_obj, dict):
        results = citations_obj.get("results") or citations_obj.get("citations", {}).get("results")
    else:
        results = getattr(citations_obj, "results", None)
        if results is None:
            try:
                results = citations_obj.get("results")  # type: ignore[attr-defined]
            except Exception:
                results = None

    if not results:
        return ""

    chunks: List[str] = []
    for r in results:
        try:
            if isinstance(r, dict):
                doc_name = r.get("document_name") or r.get("source") or "doc"
                content = r.get("content") or ""
            else:
                doc_name = getattr(r, "document_name", None) or getattr(r, "source", None) or "doc"
                content = getattr(r, "content", "") or ""
            # Heuristic: skip very long base64-ish blobs (images)
            if isinstance(content, str) and len(content) > 2000 and all(
                c.isalnum() or c in "+/=\n\r" for c in content[:200]
            ):
                continue
            if content:
                chunks.append(f"[{doc_name}]\n{content}".strip())
        except Exception:
            continue

    text = "\n\n---\n\n".join(chunks).strip()
    return text[:max_chars]


def _stream_openai_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
):
    """
    Stream OpenAI-compatible SSE response from a vLLM/OpenAI endpoint and yield SSE lines.
    """
    url = _normalize_openai_base_url(base_url)
    if not url:
        raise RuntimeError("OPENAI_BASE_URL is not set (required when GENERATION_BACKEND=openai).")
    if not model:
        raise RuntimeError("OPENAI_MODEL is not set (required when GENERATION_BACKEND=openai).")

    endpoint = f"{url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key or 'dummy'}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": True}
    with requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            if not line.startswith("data:"):
                line = "data: " + line
            yield line + "\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """
    Readiness check for production:
    - Confirms required env is present for the selected generation backend
    - Optionally checks OpenAI/vLLM endpoint connectivity
    """
    backend = (os.getenv("GENERATION_BACKEND") or "nvidia").strip().lower()
    status: Dict[str, Any] = {"status": "ok", "generation_backend": backend}

    if backend in ("openai", "vllm"):
        base_url = os.getenv("OPENAI_BASE_URL", "")
        model = os.getenv("OPENAI_MODEL", "")
        api_key = os.getenv("OPENAI_API_KEY", "dummy")

        if not base_url:
            return JSONResponse({"status": "error", "error": "OPENAI_BASE_URL is not set"}, status_code=500)
        if not model:
            return JSONResponse({"status": "error", "error": "OPENAI_MODEL is not set"}, status_code=500)

        try:
            url = _openai_models_url(base_url)
            r = requests.get(url, headers=_openai_headers(api_key), timeout=5)
            status["openai_models_status_code"] = r.status_code
        except Exception as e:
            return JSONResponse(
                {"status": "error", "error": f"OpenAI/vLLM connectivity check failed: {e}"},
                status_code=500,
            )

    return status


@app.post("/ingest")
async def ingest(
    collection_name: str = Form(...),
    vdb_endpoint: str = Form(...),
    blocking: bool = Form(True),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(150),
    generate_summary: bool = Form(False),
    file: UploadFile = File(...),
):
    """
    Ingest one document into a collection using NvidiaRAGIngestor.
    Accepts a single file (multipart upload). Pipelines calls this concurrently (bounded).
    """
    # ensure collection exists (ignore "already exists" errors)
    try:
        ingestor.create_collection(collection_name=collection_name, vdb_endpoint=vdb_endpoint)
    except Exception:
        pass

    # write to temp file because upload_documents expects filepaths
    tmp_path = None
    tmp_dir = None
    try:
        # Create a per-request directory and write the file using the *original* filename
        # (sanitized). This allows later delete-by-document-name to work consistently.
        tmp_dir = tempfile.mkdtemp(prefix="owui_upload_")
        tmp_path = os.path.join(tmp_dir, _safe_upload_filename(file.filename))
        with open(tmp_path, "wb") as tmp:
            content = await file.read()
            tmp.write(content)

        resp = await ingestor.upload_documents(
            collection_name=collection_name,
            vdb_endpoint=vdb_endpoint,
            blocking=blocking,
            split_options={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            filepaths=[tmp_path],
            generate_summary=generate_summary,
        )
        return JSONResponse(resp)

    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if tmp_dir:
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass


@app.get("/v1/documents")
async def list_documents(
    collection_name: str = Query(default="multimodal_data"),
    vdb_endpoint: Optional[str] = Query(default=None),
):
    """
    Compatibility endpoint mirroring the NVIDIA RAG Ingestor Server:
      GET /v1/documents?collection_name=...

    Returns a list of documents known to the ingestion layer for that collection.
    """
    try:
        vdb = vdb_endpoint or (os.getenv("VDB_ENDPOINT") or "http://milvus:19530")
        # Support multiple possible SDK method names.
        resp = await _call_ingestor_first(
            ["get_documents", "list_documents"],
            collection_name=collection_name,
            vdb_endpoint=vdb,
        )
        return JSONResponse(resp if isinstance(resp, dict) else {"result": resp})
    except Exception as e:
        return JSONResponse(
            {
                "error": "list_documents_not_supported",
                "detail": str(e),
                "hint": "This repo runs a lightweight worker. The full NVIDIA Ingestor Server exposes GET/DELETE /v1/documents.",
            },
            status_code=501,
        )


@app.delete("/v1/documents")
async def delete_documents(
    collection_name: str = Query(default="multimodal_data"),
    vdb_endpoint: Optional[str] = Query(default=None),
    file_names: Optional[List[str]] = Body(default=None),
):
    """
    Compatibility endpoint mirroring the NVIDIA RAG Ingestor Server:
      DELETE /v1/documents?collection_name=...
      body: ["file1.pdf", "file2.pdf"]

    Deletes documents (by document name / filename) from the specified collection.
    """
    try:
        if file_names is not None and not isinstance(file_names, list):
            return JSONResponse({"error": "invalid_body", "detail": "Expected JSON array of strings."}, status_code=422)

        vdb = vdb_endpoint or (os.getenv("VDB_ENDPOINT") or "http://milvus:19530")

        # Provide multiple possible parameter names (filtered by signature).
        kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "vdb_endpoint": vdb,
            "file_names": file_names,
            "document_names": file_names,
            "documents": file_names,
        }

        resp = await _call_ingestor_first(
            ["delete_documents", "delete_document", "remove_documents", "remove_document"],
            **kwargs,
        )
        return JSONResponse(resp if isinstance(resp, dict) else {"result": resp})
    except Exception as e:
        return JSONResponse(
            {
                "error": "delete_documents_not_supported",
                "detail": str(e),
                "hint": "If you deployed the NVIDIA Ingestor Server, it supports DELETE /v1/documents as shown in their notebooks.",
            },
            status_code=501,
        )


@app.post("/generate")
async def generate(payload: Dict[str, Any]):
    """
    Streams OpenAI-style SSE lines back to caller.
    payload example:
      {
        "messages":[...],
        "collection_names":["..."],
        "use_knowledge_base":true,
        "vdb_endpoint":"http://milvus:19530"  (optional; else VDB_ENDPOINT env or default)
      }
    """
    messages = payload.get("messages") or []
    collection_names = payload.get("collection_names") or []
    use_knowledge_base = bool(payload.get("use_knowledge_base", bool(collection_names)))
    vdb_endpoint = (payload.get("vdb_endpoint") or os.getenv("VDB_ENDPOINT") or "http://milvus:19530").strip()

    generation_backend = (os.getenv("GENERATION_BACKEND") or "nvidia").strip().lower()

    # Backend A: use NVIDIA RAG SDK generation (defaults to NVIDIA model stack)
    if generation_backend in ("nvidia", "nvidia_rag", "sdk"):
        gen_kwargs: Dict[str, Any] = {
            "messages": messages,
            "use_knowledge_base": use_knowledge_base,
            "collection_names": collection_names,
            "vdb_endpoint": vdb_endpoint,
        }
        gen_kwargs = _filter_kwargs_for_callable(rag.generate, gen_kwargs)
        rag_resp = rag.generate(**gen_kwargs)

        if getattr(rag_resp, "status_code", 200) != 200:
            return JSONResponse(
                {"error": "rag.generate failed", "status_code": getattr(rag_resp, "status_code", None)},
                status_code=500,
            )

        def sse_stream():
            for chunk in rag_resp.generator:
                if not chunk:
                    continue
                chunk = chunk.strip()
                if not chunk.startswith("data:"):
                    chunk = "data: " + chunk
                yield chunk + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    # Backend B: retrieval via NVIDIA RAG SDK, generation via OpenAI-compatible endpoint (vLLM)
    if generation_backend in ("openai", "vllm"):
        context_text = ""
        if use_knowledge_base and collection_names:
            try:
                query = _get_last_user_text(messages)
                if query:
                    search_kwargs = {"query": query, "collection_names": collection_names, "vdb_endpoint": vdb_endpoint}
                    search_kwargs = _filter_kwargs_for_callable(rag.search, search_kwargs)
                    citations = rag.search(**search_kwargs)
                    context_text = _extract_search_results_text(citations)
            except Exception:
                context_text = ""

        out_messages: List[Dict[str, Any]] = []
        if context_text:
            out_messages.append(
                {"role": "system", "content": "Use the following retrieved context when relevant.\n\n" + context_text}
            )
        out_messages.extend(messages)

        base_url = os.getenv("OPENAI_BASE_URL", "")
        api_key = os.getenv("OPENAI_API_KEY", "dummy")
        model = os.getenv("OPENAI_MODEL", "")

        def sse_stream():
            for chunk in _stream_openai_chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=out_messages,
            ):
                yield chunk
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    return JSONResponse(
        {"error": f"Unknown GENERATION_BACKEND={generation_backend}. Use 'nvidia' or 'openai'."},
        status_code=400,
    )
