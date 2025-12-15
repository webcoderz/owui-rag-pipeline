import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# NVIDIA client SDK (py3.12+)
from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

app = FastAPI(title="nvidia-rag-worker", version="1.0.0")

rag = NvidiaRAG()
ingestor = NvidiaRAGIngestor()


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

    # NvidiaRAG.generate returns an object with .status_code and .generator (per your demo)
    rag_resp = rag.generate(
        messages=messages,
        use_knowledge_base=use_knowledge_base,
        collection_names=collection_names,
        # Optional passthroughs if you want later:
        # embedding_endpoint=payload.get("embedding_endpoint"),
        # filter_expr=payload.get("filter_expr"),
    )

    if getattr(rag_resp, "status_code", 200) != 200:
        return JSONResponse({"error": "rag.generate failed", "status_code": getattr(rag_resp, "status_code", None)}, status_code=500)

    def sse_stream():
        # pass-through NVIDIA “data: {...}” chunks; ensure SSE framing
        for chunk in rag_resp.generator:
            if not chunk:
                continue
            chunk = chunk.strip()
            if not chunk.startswith("data:"):
                chunk = "data: " + chunk
            yield chunk + "\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")
