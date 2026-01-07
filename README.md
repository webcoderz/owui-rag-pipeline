## OWUI RAG Pipeline - Operations Guide

This repo wires Open WebUI Pipelines to an NVIDIA RAG worker (Python SDK) and a Postgres-backed ingest manifest, with Milvus as the VDB.

### What you get
- NVIDIA RAG worker (`docker/worker`) exposing:
  - `POST /ingest` for document ingestion (multipart)
  - `POST /generate` for streaming chat completions (SSE)
- Open WebUI Pipelines container loading `docker/pipelines/nvidia_ingest_bridge_pipe.py`, which:
  - Fetches files/KBs from Open WebUI
  - Streams to the worker for ingestion and generation
  - Tracks ingest status in Postgres
  - Supports per-user library and chat allowlist

## Prerequisites
- Docker and Docker Compose
- External services reachable from this compose:
  - Open WebUI (Admin) at `OPENWEBUI_BASE_URL` (e.g., `http://open-webui:8080`)
  - Milvus at `VDB_ENDPOINT` (e.g., `http://milvus:19530`)
- Two external Docker networks must already exist (or adjust `docker-compose.yaml`):
  - `default` (external: true) shared with your Open WebUI stack
  - `nvidia-rag` (external: true) for the worker + pipelines (and other NVIDIA services, if any)

Create networks if needed:

```bash
docker network create default || true
docker network create nvidia-rag || true
```

## Environment
Create a `.env` next to `docker-compose.yaml` with at least:

```bash
OPENWEBUI_API_KEY=your-openwebui-service-token
HTTP_PROXY=
HTTPS_PROXY=
NO_PROXY=localhost,127.0.0.1,open-webui,pipelines,nvidia-rag-worker,owui-postgres,milvus

# NVIDIA RAG SDK (MinIO is required for ingestion internals)
# IMPORTANT: inside Docker, localhost points at the container itself.
# Use the MinIO service hostname reachable on your Docker network.
MINIO_ENDPOINT=minio:9010
MINIO_ACCESSKEY=minioadmin
MINIO_SECRETKEY=minioadmin
# Optional
NVINGEST_MINIO_BUCKET=nv-ingest
```

Notes:
- If behind a proxy, set `HTTP_PROXY`/`HTTPS_PROXY`. `NO_PROXY` should include internal hostnames above.
- Make sure your Open WebUI instance has a service token (Admin → Settings → API Keys).

## Bring up the stack

```bash
docker compose up -d --build
```

What starts:
- `nvidia-rag-worker` at `http://nvidia-rag-worker:8123` (exposed on host `8123`)
- `pipelines` (Open WebUI Pipelines) with the custom pipe mounted
- `owui-postgres` (stores ingest manifest and per-user settings)

Quick health check (worker):

```bash
curl http://localhost:8123/health
```

## Connect Pipelines to Open WebUI
In Open WebUI (Admin → Pipelines):
- Add pipeline URL: `http://pipelines:9099`
- Set the API key to match `PIPELINES_API_KEY` in `docker-compose.yaml` (defaults to `0p3n-w3bu!`)
- Save and verify the pipeline is listed as:
  - `NVIDIA RAG (Auto-Ingest • Library • Persistent)` with ID `nvidia-rag-auto-ingest`

Requirements for connectivity:
- The Open WebUI container must be on the same external `default` network so it can resolve `pipelines`.
- The pipelines container can reach `open-webui` at `OPENWEBUI_BASE_URL`.
- The worker and pipelines can reach `milvus` at `VDB_ENDPOINT`.

## Using it in chats
In Open WebUI:
1. Start a chat and select the pipeline tool “NVIDIA RAG (Auto-Ingest • Library • Persistent)”.
2. Attach files or select Knowledge Bases; the pipeline will:
   - Fetch the content from Open WebUI
   - Ingest into collections (named by convention)
   - Stream model responses via the NVIDIA worker
3. Commands supported:
   - `/library on` — future ingests save to your per-user library automatically
   - `/library off` — disable auto-saving to your library for this chat

Collection naming conventions (prefix defaults to `owui`):
- Knowledge base: `owui-kb-<kb_id>` or `owui-kb-public-<kb_id>`
- Chat uploads: `owui-u-<user>-chat-<chat_id>` (user-scoped by default)
- Library: `owui-u-<user>-library`

## Troubleshooting
- Proxies:
  - Set `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` in `.env`
  - Ensure `NO_PROXY` includes internal names: `open-webui,pipelines,nvidia-rag-worker,owui-postgres,milvus,localhost,127.0.0.1`
- Networking:
  - Ensure Open WebUI, pipelines, and worker share an external network and can resolve each other by the names in `docker-compose.yaml`
- Postgres:
  - Data is stored in the `owui_pg` volume; remove with care if you need a reset
- Milvus:
  - Verify it’s reachable at `VDB_ENDPOINT` and the collection prefix has permissions to create collections

## Direct worker calls (optional)
You can test the worker directly.

Generate:
```bash
curl -N -X POST http://localhost:8123/generate \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}], "collection_names": [], "use_knowledge_base": false}'
```

Ingest (single file):
```bash
curl -X POST http://localhost:8123/ingest \
  -F collection_name="test_library" \
  -F vdb_endpoint="http://milvus:19530" \
  -F blocking=true \
  -F chunk_size=512 \
  -F chunk_overlap=150 \
  -F generate_summary=false \
  -F file=@/path/to/your.pdf
```

## Clean up
```bash
docker compose down
```
# owui-rag-pipeline