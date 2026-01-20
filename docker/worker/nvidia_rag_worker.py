import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# ----
# IMPORTANT (NVIDIA RAG SDK dependency):
# The `nvidia_rag` package imports/initializes its ingestor server and MinIO operator at import time.
# If MINIO_* env vars are missing, it may default to localhost:9010 inside the container and fail.
# ----

def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(
            f"{name} is not set. The NVIDIA RAG SDK requires MinIO configuration inside Docker. "
            f"Set {name} (and MINIO_ACCESSKEY/MINIO_SECRETKEY) to a hostname reachable from the container "
            f"(e.g. MINIO_ENDPOINT=minio:9010), not localhost."
        )
    return v

_require_env("MINIO_ENDPOINT")
_require_env("MINIO_ACCESSKEY")
_require_env("MINIO_SECRETKEY")

# NVIDIA client SDK (py3.12+)
from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

app = FastAPI(title="nvidia-rag-worker", version="1.0.0")

rag = NvidiaRAG()
ingestor = NvidiaRAGIngestor()

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
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.split(".")[-1]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
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


@app.post("/generate")
async def generate(payload: Dict[str, Any]):
    """
    Streams OpenAI-style SSE lines back to caller.
    payload example:
      {
        "messages":[...],
        "collection_names":["..."],
        "use_knowledge_base":true
      }
    """
    messages = payload.get("messages") or []
    collection_names = payload.get("collection_names") or []
    use_knowledge_base = bool(payload.get("use_knowledge_base", bool(collection_names)))

    generation_backend = (os.getenv("GENERATION_BACKEND") or "nvidia").strip().lower()

    # Backend A: use NVIDIA RAG SDK generation (defaults to NVIDIA model stack)
    if generation_backend in ("nvidia", "nvidia_rag", "sdk"):
        rag_resp = rag.generate(
            messages=messages,
            use_knowledge_base=use_knowledge_base,
            collection_names=collection_names,
        )

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
                    citations = rag.search(query=query, collection_names=collection_names)
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
