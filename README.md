# OWUI RAG Pipeline
[![CI](https://github.com/webcoderz/owui-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/webcoderz/owui-rag-pipeline/actions/workflows/ci.yml)


## Operations Guide

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

**Ingest metadata:** The worker creates collections with a metadata schema and stores **filename** (original upload name), **uploaded_at** (ISO 8601 timestamp), and **uploaded_by** (OWUI user id, for audit/attribution) per document. Access control remains at the OWUI/pipeline level (collection naming and which collections are queried); `uploaded_by` is for display and filtering only. Collections created before this feature have no schema; uploads to them skip metadata (backward compatible).

### Access control (OWUI–Milvus linking)

Collections are scoped so Milvus usage respects Open WebUI permissions:

| Collection pattern | OWUI concept | Access rule |
|-------------------|---------------|-------------|
| `owui-u-{user}-library` | User's library | Only that user |
| `owui-u-{user}-chat-{id}` / `owui-chat-{id}` | Chat uploads | User-scoped or global (see `USER_SCOPED_CHAT_COLLECTIONS`) |
| `owui-kb-public-{id}` / `owui-kb-{id}` | Knowledge base | OWUI KB permissions (user/group); checked via `GET /api/v1/knowledge/{id}` |

When the pipeline receives the requesting user's Bearer token (e.g. from `__request__`), it uses that token for OWUI API calls (files, knowledge). That way Open WebUI enforces access and we only query/ingest collections the user is allowed to see. If only a service key is used, KB access is not user-scoped.

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

# Optional: use your own OpenAI-compatible (vLLM) model for final generation
# (retrieval still happens via NVIDIA RAG + Milvus)
GENERATION_BACKEND=openai
OPENAI_BASE_URL=http://<HOST>:<PORT>
OPENAI_MODEL=openai/gpt-oss-120b
OPENAI_API_KEY=dummy
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
- `proxy` (nginx) at host port `PROXY_HTTP_PORT` (default `80`), proxying **NVIDIA RAG stack** at subpaths and Open WebUI as default:
  - `http://localhost/ingestor/` → ingestor (8082); `http://localhost/rag/` → RAG (8081); `/` → Open WebUI (8080). Swagger at `/rag/docs` and `/ingestor/docs` work via sub_filter. Backends on same Docker network (e.g. `nvidia-rag`, `open-webui_default`).

Quick health check (worker):

```bash
curl http://localhost:8123/health
```

Readiness check (validates OpenAI/vLLM settings when enabled):

```bash
curl http://localhost:8123/ready
```

## Pipeline vs. “your OpenAI model” in Open WebUI
When you **select the pipeline** as the model in a chat (e.g. “NVIDIA RAG (Auto-Ingest • Library • Persistent)”), Open WebUI sends the request to the **Pipelines** server. The pipeline then calls the **NVIDIA RAG worker** at `POST /generate` — it does **not** call Open WebUI’s chat API or the model you use in other chats (e.g. your OpenAI model).

So the model that actually generates the RAG reply is controlled by the **worker** environment, not by the model dropdown in Open WebUI:

- **`GENERATION_BACKEND=nvidia`** (default): the worker uses the NVIDIA RAG SDK’s built-in generation (NVIDIA NIM/catalog). No OpenAI config is used for the reply.
- **`GENERATION_BACKEND=openai`**: the worker does RAG retrieval (NVIDIA + Milvus) and sends the augmented messages to **`OPENAI_BASE_URL`** with **`OPENAI_MODEL`** to produce the reply. That can be your usual OpenAI-compatible endpoint (same one you use in OWUI) or a different one.

**To use “your” OpenAI model for RAG replies:** set the worker env to the same API and model you use in Open WebUI, for example:

```bash
GENERATION_BACKEND=openai
OPENAI_BASE_URL=https://api.openai.com/v1   # or your proxy/OWUI URL if the model is exposed there
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

**Model id:** Use the **exact** value the API expects. For Open WebUI’s API use the id shown in the UI (e.g. `openai/gpt-oss-120b`); for vLLM/direct backends use the raw model name (e.g. `gpt-oss-120b`).

If your model is only available through Open WebUI (e.g. you added it in Admin → Connections), use Open WebUI’s API URL and the model id Open WebUI uses (e.g. `OPENAI_BASE_URL=http://open-webui:8080/v1` and `OPENAI_MODEL=openai/gpt-oss-120b` (use the exact model id the endpoint expects: with prefix for Open WebUI API, raw name for vLLM/direct)), and ensure the worker can reach that URL on the Docker network.

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

## (Optional) Add Tools (recommended for production UX)
This repo includes an OpenAPI tools server (`owui-rag-tools`) so you can promote “commands” into real Open WebUI tools (discoverable + schema’d).

In Open WebUI (Admin → Settings → Tools / OpenAPI Servers):
- Add OpenAPI server URL: `http://owui-rag-tools:8181/openapi.json`
- Enable the tools for the model(s) you want (e.g. your main chat model)

Available tools:
- `POST /library/set` — toggle per-chat “save to library”
- `GET /library/status` — read current setting
- `POST /allowlist/add` — add collection(s) to the chat allowlist
- `GET /allowlist/list` — list allowlisted collections

## Using it in chats
In Open WebUI:
1. Start a chat and select the pipeline tool “NVIDIA RAG (Auto-Ingest • Library • Persistent)”.
2. Attach files or select Knowledge Bases; the pipeline will:
   - Fetch the content from Open WebUI
   - Ingest into collections (named by convention)
   - Stream model responses via the NVIDIA worker
3. Slash commands (handled by the pipeline):
   - `/commands` or `/help` — show available commands
   - `/collection list` — show this chat’s remembered collections + derived chat/library collection names
   - `/library` — show this chat’s current “save to library” setting
   - `/library on` — future ingests in this chat also save to your per-user library
   - `/library off` — disable auto-saving to your library for this chat
   - `/ingest` — ingest the currently attached files / selected Knowledge Bases and stop (no response generation)
   - `/ingest <collection>` — same, but ingest into a specific target collection (also remembers it for this chat)
   - `/ingest chat` — shorthand for ingesting into this chat’s derived uploads collection
   - `/ingest library` — shorthand for ingesting into your derived library collection
   - `/query <question>` — ask a question using the current chat’s remembered collections (and optionally your library if enabled by default)
   - `/query <collection> <question>` — ask a question using a specific collection only
   - `/query chat <question>` / `/query library <question>` — shorthand for derived collections
   - `/delete <collection> <filename>` — delete a document from a collection (uses worker `DELETE /v1/documents`)
   - `/delete chat <filename>` / `/delete library <filename>` — shorthand for derived collections
  - `/forget` — clear this chat's remembered collections; next message won't use RAG until you attach again or use `/query <collection>`

Notes:
- **/ingest only uses files attached to the message you send.** If you remove the PDFs from the chat and then send `/ingest library`, the request has no files, so nothing is ingested and the worker is never called (no "Ingest request" / "Ingest complete (Milvus)" logs). Attach the files again and send `/ingest` in the same message.
- **Why does it always query against the doc I attached?** After you attach a document, the pipeline adds that collection to this chat's "remembered" list so every follow-up message uses it for RAG. To stop using the doc in this chat, use **`/forget`**; the next message will then be answered without RAG unless you attach files again or use `/query <collection> <question>`.

- These are not Open WebUI “tool” slash commands, so they won’t autocomplete; they work only when you’re chatting with the pipeline model.
- For a UI-native tools experience, use the OpenAPI tools server (`owui-rag-tools`) above.

**How to verify RAG is working:** (1) **Content check:** Ask a question only your ingested docs can answer; if the answer reflects that content, retrieval is working. (2) **Worker/Milvus logs:** Tail the worker to see that search and Milvus are used:  
`docker compose logs -f nvidia-rag-worker`  
You should see: **RAG querying** — `RAG query (generate):` or `RAG query (search):` with collections and (for search) the user query; then SDK lines `RAG PIPELINE START - search() method invoked`, `Search Parameters:` with query and collection names (e.g. `owui_u_anon_library`). **Collections:** `GET /collections?...` (200). **Ingest:** `Ingest request:` and `Ingest complete (Milvus):` per file. The worker sets the RAG pipeline logger to INFO so these lines are visible by default.

Collection naming conventions (prefix defaults to `owui`):
- Knowledge base: `owui-kb-<kb_id>` or `owui-kb-public-<kb_id>`
- Chat uploads: `owui-u-<user>-chat-<chat_id>` (user-scoped by default)
- Library: `owui-u-<user>-library` (one collection per user; all library ingest appends to it)
- Names are **unique per user/chat/kb**, not per ingest: e.g. every `/ingest library` (same user) uses the same collection so documents accumulate. To get a **unique collection per ingest**, set `UNIQUE_LIBRARY_COLLECTION_PER_INGEST=true` (then each `/ingest library` creates e.g. `owui-u-<user>-library-<8-char-id>`), or use an explicit name: `/ingest my-batch-20260220`.
- **Service account / API key:** When you call the pipeline with a **service account or API key** (no logged-in user), Open WebUI often does not pass a user id/email in the body, so the pipeline would use **`anon`**. You can (1) set **`ENABLE_FORWARD_USER_INFO_HEADERS=True`** in Open WebUI so it sends user info in request headers; the pipeline then reads **`X-User-Id`** / **`X-User-Email`** (or **`X-OpenWebUI-User-Id`** / **`X-OpenWebUI-User-Email`**) and uses them for collection naming when the body has no user. (2) Or set **`COLLECTION_USER_KEY`** in the pipelines environment (e.g. `COLLECTION_USER_KEY=service-acct`) so API-key requests use that label instead of `anon`.

### Ensuring user info is forwarded (critical when using API key)

For **per-user** library and chat collections (e.g. `owui-u-<user>-library` instead of `owui-u-anon-library`), the pipeline must receive the requesting user. When using a **service account or API key**, the request body often has no user, so the pipeline relies on **headers** from Open WebUI.

**Official Open WebUI docs:** [Environment Variable Configuration](https://docs.openwebui.com/getting-started/env-configuration/) → **Security Variables** → **ENABLE_FORWARD_USER_INFO_HEADERS**. The docs state that no other step is required: set the env var to `True` in the Open WebUI deployment. There is no admin UI toggle for this; it is environment-only. When enabled, Open WebUI forwards these headers to downstream APIs: **`X-OpenWebUI-User-Name`**, **`X-OpenWebUI-User-Id`**, **`X-OpenWebUI-User-Email`**, **`X-OpenWebUI-User-Role`**, **`X-OpenWebUI-Chat-Id`**. The pipeline uses Id and Email for collection naming. The docs describe forwarding to "OpenAI API and Ollama API"; Pipelines are connected as an **OpenAI connection** in Admin → Settings → Connections, so the same setting applies to requests from Open WebUI to the Pipelines server.

1. **Enable in Open WebUI** (required for header-based user):
   - In your **Open WebUI** deployment (not this repo), set:
     - **`ENABLE_FORWARD_USER_INFO_HEADERS=True`**
   - Restart Open WebUI after changing env so the setting takes effect. Open WebUI then adds the headers above to its outbound request when calling the pipeline (and other OpenAI-compatible connections).
2. **Verify from pipeline logs** (optional):
   - In this repo’s `.env`, set **`PIPE_DEBUG=true`** and ensure **`PIPE_DEBUG`** is passed to the pipelines service (see `docker-compose.yaml`).
   - Restart pipelines: `docker compose restart pipelines`.
   - Trigger a request (e.g. send a message in a chat using the pipeline).
   - In **pipelines** logs (`docker compose logs pipelines`) look for:
     - `PIPE_DEBUG user: ... header_X-OpenWebUI-User-Id=present header_X-OpenWebUI-User-Email=present ... user_key_source=headers`
   - If you see `header_X-OpenWebUI-User-Id=missing` or `no_headers`, then: (a) confirm **`ENABLE_FORWARD_USER_INFO_HEADERS=True`** is set in the **Open WebUI** container/env and that you **restarted Open WebUI** after setting it; (b) ensure the request goes **through Open WebUI** to the pipeline (e.g. you select the pipeline as the model in the chat UI, or your app calls Open WebUI’s API and the pipeline is the target model)—calling the Pipelines server URL directly from outside Open WebUI will not include these headers; (c) ensure no proxy between Open WebUI and the Pipelines service strips or overwrites the `X-OpenWebUI-*` headers.
3. **Fallback if you cannot enable headers:** Set **`COLLECTION_USER_KEY`** in the pipelines env (e.g. `COLLECTION_USER_KEY=service-acct`) so all API-key requests use that label instead of `anon` (single shared identity).

## Backfilling existing OWUI knowledge into Milvus

If you had knowledge bases in Open WebUI **before** integrating this pipeline, their documents live in OWUI but not in Milvus. Use the backfill script to ingest those into the same collection names the pipeline uses, so RAG queries can search that content.

**Safety and idempotency:**
- **--confirm** is required for real ingest (without `--dry-run`). Avoids accidental runs.
- **Skip existing:** By default the script asks the worker which documents are already in each collection and skips those, so you can run it multiple times without duplicating documents.
- **Preflight:** Before any work, the script validates OWUI and the worker; use `--no-validate` only if you know they are up.

**From the host** (Python 3, no extra deps):

```bash
# Required env (or set in .env and source / export)
export OPENWEBUI_BASE_URL=http://localhost:8080
export OPENWEBUI_API_KEY=your-openwebui-service-token
export NVIDIA_WORKER_URL=http://localhost:8123
export VDB_ENDPOINT=http://milvus:19530          # optional

# Dry run (no ingest, no --confirm needed)
python scripts/backfill_owui_knowledge_to_milvus.py --kb-ids=abc123,def456 --dry-run

# Real run (requires --confirm)
python scripts/backfill_owui_knowledge_to_milvus.py --kb-ids=abc123,def456 --confirm
```

If your OWUI version supports listing knowledge bases at `GET /api/v1/knowledge`, you can omit `--kb-ids` and the script will try to backfill all. Otherwise pass `--kb-ids=id1,id2` (IDs from the Knowledge UI or API).

- To re-upload documents that are already in the collection (not recommended): use `--no-skip-existing`.
- Collection names and Milvus-safe conversion match the pipeline; the script does not create extra or “false” collections—each KB maps to one collection name.

## Troubleshooting
- **Code changes not taking effect / "create the collection first" / collection doesn't exist**
  - **Worker:** The worker image **bakes in** `nvidia_rag_worker.py` at build time. `docker compose up --force-recreate` does **not** rebuild the image, so you still run old code. To pick up worker changes:  
    `docker compose build nvidia-rag-worker --no-cache && docker compose up -d nvidia-rag-worker`
  - **Pipeline:** The pipeline script is **mounted** from the host, but the Pipelines server **caches** the Python module in memory after first import. Restart the container so it reloads the file:  
    `docker compose restart pipelines`
- **"coroutine Pipeline.pipe was never awaited"**  
  The pipeline exposes a synchronous `pipe()` so Open WebUI can call it without `await`. If you still see this, ensure you’re on the latest pipeline image and that the Pipelines service has been restarted after an update.
- **/commands, /help, or /library off does nothing**  
  Usually the same cause as above: the runtime wasn’t awaiting `pipe()`. With the sync wrapper, slash commands should return immediately. Send the command as the only content in the message (e.g. type `/library off` and send).
- **Pipeline keeps re-ingesting or “retrieving” the same document**  
  If the chat request still includes the same attachments on every turn, the pipeline will ingest again each time. Send a message *without* new attachments for plain chat or slash commands (e.g. `/library off`, `/help`, or a question). Clear or don’t re-attach the file for the next message.
- **Blank or empty reply when querying a collection**  
  The pipeline does **not** use the model you select elsewhere in Open WebUI (e.g. your OpenAI model). It calls the **NVIDIA RAG worker** for generation. So: (1) If you want the same OpenAI model for RAG, set the **worker** env: `GENERATION_BACKEND=openai`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY` (see [Pipeline vs. “your OpenAI model”](#pipeline-vs-your-openai-model-in-open-webui)). (2) If the worker uses the NVIDIA backend, ensure the NVIDIA RAG SDK can reach its generation endpoint (e.g. API catalog). (3) To see what the worker is actually streaming, use the pipeline debug log (file-only, no network — see **Airgapped / offline** below).
- **Airgapped / offline:** Pipeline debug logging writes only to a file (no external server). The pipelines service uses the writable mount `./pipeline_debug:/app/pipeline_debug` (see `docker-compose.yaml`). After reproducing the issue, the log is at **`pipeline_debug/debug.log`** in the project directory. Open that file and copy-paste its contents to share. Each line is one NDJSON record (timestamp, message, data, hypothesisId).
- **Streaming response not showing in UI (logs show completion)**  
  The pipeline yields SSE event lines in the format the Open WebUI Pipelines server expects (`data: {...}` per line; the server adds `\n\n`). Chunk payloads include `logprobs` and match the OpenAI-style `chat.completion.chunk` shape. If the UI still doesn't update, check that the OpenAI API URL in Open WebUI points at the Pipelines service and that no proxy is buffering or altering the stream.
- **Log shows `stream:true:<generator object Pipeline._pipe_async...>`**  
  That log line is from the Pipelines server (it logs the return value of `pipe()`). A **generator** is correct for streaming — it means the pipeline returned a stream. The server then iterates that generator and forwards each chunk to the client. If you previously saw `<coroutine object ...>`, that was the bug (unawaited); seeing `<generator ...>` means the response is set up correctly.
- **Worker response is HTML / blank UI:** The pipeline’s request to the worker was sent via an HTTP proxy that returned a block page instead of the SSE stream. The pipeline now uses a dedicated HTTP client for worker requests that ignores `HTTP_PROXY` (`trust_env=False`), so worker traffic goes direct. Also set `NO_PROXY` (and `no_proxy`) in the pipelines service so internal hostnames bypass the proxy (see `docker-compose.yaml` default). If the proxy error persists, the **worker** may be the one hitting the proxy when it calls the NVIDIA SDK or OPENAI_BASE_URL. The worker now defaults to `NO_PROXY=*` and disables proxy in code for the OpenAI path; **rebuild the worker** after pulling: `docker compose build nvidia-rag-worker --no-cache && docker compose up -d nvidia-rag-worker`.
- **Failed to connect to Milvus / "HTTP proxy handshake ... 503" in worker:** The Milvus client (gRPC) and some HTTP stacks do not respect `NO_PROXY`; if the worker has `HTTP_PROXY` set, it sends Milvus and OPENAI_BASE_URL traffic via the proxy and fails. Compose **overrides** proxy to empty for the worker by default. If you must pass a proxy to the worker, set `NO_PROXY` and `no_proxy` so Milvus and the LLM host bypass it, e.g. `no_proxy: "milvus-standalone,<OPENAI_BASE_URL host or IP>"` (use the hostname or IP of your gpt-oss / OPENAI_BASE_URL). Recreate the worker after changing env.
- Proxies:
  - Set `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` in `.env` for pipelines and build. **Worker does not receive proxy vars** so it can reach Milvus/MinIO directly.
  - Pipelines: `NO_PROXY` should include internal hostnames (compose default).
  - Ensure `NO_PROXY` includes internal names when not using *: `open-webui,pipelines,nvidia-rag-worker,owui-postgres,milvus,milvus-standalone,localhost,127.0.0.1`
- Networking:
  - Ensure Open WebUI, pipelines, and worker share an external network and can resolve each other by the names in `docker-compose.yaml`
- Postgres:
  - Data is stored in the `owui_pg` volume; remove with care if you need a reset
- Milvus:
  - Verify it’s reachable at `VDB_ENDPOINT` and the collection prefix has permissions to create collections
  - **Warnings "no entities found in collection metadata_scheme/document_info for filter"**: The NVIDIA RAG SDK may query internal Milvus collections (`metadata_scheme`, `document_info`) for catalog/metadata. If you haven't ingested catalog-style data, those collections are empty and the SDK logs a warning. These are **benign** and can be ignored; RAG search uses your document collections, not these.
- **No ingest logs / "doesn't recall the documents" after deleting PDFs:** The pipeline only ingests files that Open WebUI sends in **that request** (`body.files` = attachments on the current message). If you remove the PDFs from the chat and then send `/ingest library`, there are no files in the request, so you get "📎 No attachments/knowledge selected" and the worker is never called—no "Ingest request" or "Ingest complete (Milvus)" in the worker logs. **Fix:** Attach the files again and send `/ingest <collection>` in the same message (or keep the attachments on the message). After a successful ingest, the worker logs one "Ingest request" and one "Ingest complete (Milvus)" per file.
- **Milvus console shows no collections or only empty `document_info`:** The worker must talk to the **same** Milvus instance you open in the console. Set **`VDB_ENDPOINT`** to that instance’s URL using the **Docker hostname** the worker can resolve (e.g. if your stack runs Milvus as `milvus-standalone`, use `VDB_ENDPOINT=http://milvus-standalone:19530` in `.env` for both pipelines and worker). Ensure the worker container is on the same Docker network as Milvus. After fixing, restart worker and pipelines, run `/ingest` with files attached, then check worker logs for `Ingest create_collection ok`, `Ingest complete (Milvus)` and the `response=` payload; any failure will be logged as `create_collection failed` or `upload_documents failed`. If you see no `Ingest request` lines, the pipeline didn't call ingest (usually because no files were attached).
- **"Collection owui_u_anon_library does not exist" in worker logs:** Search was called for a collection that doesn't exist yet. The pipeline now only passes collection names that exist in Milvus to the worker; if none exist, you get a normal reply without RAG instead of this error. To create and fill the collection, attach one or more files to your message and send `/ingest library` (or the target collection name).
- **Ingest → query flow and the 200 on POST /generate**: When you attach files and send a message, the pipeline (1) ingests into collections and streams status ("✅ Ingestion complete.", "📂 Ingested into: `collection-name`"), (2) then runs the RAG query and streams "💬 Querying…" followed by the model reply. The **200** on `POST /generate` is the worker's successful completion of that RAG request. So: ingest messages first, then "Querying…", then the response; the 200 is expected and indicates the answer was generated successfully.

## Production checklist (high-signal)
- **User info when using API key:** If the pipeline is called with a service account/API key and you want per-user library/chat collections, set **`ENABLE_FORWARD_USER_INFO_HEADERS=True`** in your **Open WebUI** deployment so OWUI sends user headers to the pipeline. See [Ensuring user info is forwarded](#ensuring-user-info-is-forwarded-critical-when-using-api-key).
- **MinIO**: `MINIO_ENDPOINT` must be a Docker-reachable hostname (never `localhost` inside containers). The SDK requires **`MINIO_BUCKET`** for “retrieve collections” and ingest; set it to a bucket that exists in MinIO (e.g. `nv-ingest`, or `default-bucket` / `a-bucket` if that’s what you have). Create the bucket in MinIO if needed.
- **Generation backend**:
  - If using vLLM/OpenAI: set `GENERATION_BACKEND=openai`, plus `OPENAI_BASE_URL` + `OPENAI_MODEL`.
  - Validate with `curl http://localhost:8123/ready` (should return `status: ok`).
- **Networking**: ensure `NO_PROXY` includes your internal service names so proxy settings don’t break container-to-container traffic.
- **Restarts**: after editing `docker/pipelines/nvidia_ingest_bridge_pipe.py`, restart the `pipelines` container to reload it.

## Proxy (nginx, optional)
A stock **nginx** container proxies ingestor and RAG at subpaths and sends everything else to Open WebUI. No custom build (works behind a proxy). `sub_filter` rewrites responses so Swagger at `/rag/docs` and `/ingestor/docs` load the spec and "Try it out" uses the correct base path.

- **Ingestor (8082)**: `http://localhost/ingestor/...` → `ingestor-server:8082`
- **RAG (8081)**: `http://localhost/rag/...` → `rag-server:8081`
- **Open WebUI**: `http://localhost/` → `open-webui:8080`

Set `PROXY_HTTP_PORT` in `.env` to change the host port (default `80`). Edit `docker/nginx/default.conf` if your stack uses different backend hostnames.

## API reference (Ingestor + RAG)
For **all HTTP requests** to the NVIDIA Ingestor and RAG servers (upload documents, create/list/delete collections, RAG chat completions, health checks), see **[docs/NVIDIA_RAG_API_REQUESTS.md](docs/NVIDIA_RAG_API_REQUESTS.md)**. That doc includes method, path, headers, body, and curl examples for both services.

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
