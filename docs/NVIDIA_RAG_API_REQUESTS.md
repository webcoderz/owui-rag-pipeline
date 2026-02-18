# NVIDIA RAG Blueprint — API Requests Reference

This document lists **all HTTP requests** to interact with the two NVIDIA RAG Blueprint services:

1. **Ingestor Server** — upload documents, manage collections, check status  
2. **RAG Server** — run RAG queries (chat completions over your documents)

In this repo, when using the full NVIDIA stack behind the proxy:

- **Ingestor** → `http://localhost/ingestor/` (proxied to ingestor-server:8082) or direct `http://localhost:8082`
- **RAG** → `http://localhost/rag/` (proxied to rag-server:8081) or direct `http://localhost:8081`

Replace the base URL if your host/port differ. OpenAPI (Swagger) docs: `/ingestor/docs` and `/rag/docs` via the proxy.

---

## 1. Ingestor Server (Document Upload & Collection Management)

Default port: **8082**. Base URL below: `http://<INGESTOR_HOST>` (e.g. `http://localhost:8082` or `http://localhost/ingestor` with proxy).

### 1.1 Health check

```http
GET http://<INGESTOR_HOST>/v1/health?check_dependencies=true
```

Optional query: `check_dependencies=true` to validate dependent services.

---

### 1.2 Upload documents

Upload one or more files to a collection. Uses **multipart/form-data**.

**Request**

```http
POST http://<INGESTOR_HOST>/v1/documents
Content-Type: multipart/form-data; boundary=<boundary>
```

Do not set the header manually when using curl: use `-F` and curl will set `Content-Type` and generate a boundary automatically.

**Form fields**

| Field        | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `documents` | file   | Yes      | One or more files (repeat for multiple). Supported: PDF, DOCX, etc. (see NeMo Retriever extraction docs). |
| `data`      | string | Yes      | JSON string with ingestion options (see below). |

**`data` JSON fields**

| Field             | Type    | Required | Description |
|------------------|---------|----------|-------------|
| `collection_name`| string  | Yes      | Target collection name. |
| `blocking`       | boolean | No       | If `true`, wait until processing completes. Default often `false`. |
| `split_options` | object  | No       | Chunking, e.g. `{"chunk_size": 512, "chunk_overlap": 150}`. |
| `generate_summary` | boolean | No    | Enable document summarization. |
| `custom_metadata`  | array  | No       | Optional metadata entries. |

**Example (curl)**

```bash
curl -X POST "http://localhost:8082/v1/documents" \
  -F "documents=@/path/to/report.pdf" \
  -F "documents=@/path/to/notes.docx" \
  -F 'data={"collection_name":"my_library","blocking":false,"split_options":{"chunk_size":512,"chunk_overlap":150},"generate_summary":false}'
```

**Example (via proxy)**

```bash
curl -X POST "http://localhost/ingestor/v1/documents" \
  -F "documents=@report.pdf" \
  -F 'data={"collection_name":"my_library","blocking":false,"split_options":{"chunk_size":512,"chunk_overlap":150},"generate_summary":false}'
```

Response typically includes a `task_id` for non-blocking uploads; use the status endpoint to poll.

---

### 1.3 Check upload / task status

After a non-blocking upload, poll for completion.

**Request**

```http
GET http://<INGESTOR_HOST>/v1/documents/status/<task_id>
```

Replace `<task_id>` with the ID returned from the upload response.

**Example (curl)**

```bash
curl "http://localhost:8082/v1/documents/status/YOUR_TASK_ID"
```

---

### 1.4 Create collection

Create a new collection in the vector database. (Exact path may be `/v1/collection` or `/v1/collections` depending on blueprint version; check `/ingestor/docs`.)

**Request**

```http
POST http://<INGESTOR_HOST>/v1/collection
Content-Type: application/json
```

**Body (JSON)**

```json
{
  "collection_name": "my_library",
  "vdb_endpoint": "http://milvus:19530"
}
```

Adjust `vdb_endpoint` to your Milvus (or other VDB) URL.

**Example (curl)**

```bash
curl -X POST "http://localhost:8082/v1/collection" \
  -H "Content-Type: application/json" \
  -d '{"collection_name":"my_library","vdb_endpoint":"http://milvus:19530"}'
```

---

### 1.5 List collections

**Request**

```http
GET http://<INGESTOR_HOST>/v1/collections?vdb_endpoint=http://milvus:19530
```

Query param `vdb_endpoint` is required (your vector DB URL).

**Example (curl)**

```bash
curl "http://localhost:8082/v1/collections?vdb_endpoint=http://milvus:19530"
```

---

### 1.6 Get documents in a collection

**Request**

```http
GET http://<INGESTOR_HOST>/v1/documents?collection_name=my_library&vdb_endpoint=http://milvus:19530
```

Query params: `collection_name`, `vdb_endpoint` (required).

**Example (curl)**

```bash
curl "http://localhost:8082/v1/documents?collection_name=my_library&vdb_endpoint=http://milvus:19530"
```

---

### 1.7 Update documents

Replace/update existing documents in a collection. Same multipart pattern as upload; endpoint may be `PUT /v1/documents` or a dedicated update path — check OpenAPI at `/ingestor/docs`.

**Request (conceptual)**

```http
PUT http://<INGESTOR_HOST>/v1/documents
Content-Type: multipart/form-data
```

Form: `documents` (files) + `data` (JSON with `collection_name`, `blocking`, `split_options`, `generate_summary`, etc.).

---

### 1.8 Delete documents from a collection

**Request**

```http
DELETE http://<INGESTOR_HOST>/v1/documents
Content-Type: application/json
```

**Body (JSON)**

```json
{
  "collection_name": "my_library",
  "document_names": ["report.pdf", "notes.docx"],
  "vdb_endpoint": "http://milvus:19530"
}
```

**Example (curl)**

```bash
curl -X DELETE "http://localhost:8082/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{"collection_name":"my_library","document_names":["report.pdf"],"vdb_endpoint":"http://milvus:19530"}'
```

---

### 1.9 Delete collection

Removes the collection and its documents from the vector DB.

**Request**

```http
DELETE http://<INGESTOR_HOST>/v1/collections
Content-Type: application/json
```

**Body (JSON)**

```json
{
  "vdb_endpoint": "http://milvus:19530",
  "collection_names": ["my_library"]
}
```

**Example (curl)**

```bash
curl -X DELETE "http://localhost:8082/v1/collections" \
  -H "Content-Type: application/json" \
  -d '{"vdb_endpoint":"http://milvus:19530","collection_names":["my_library"]}'
```

*(Exact paths for create/list/delete may vary by blueprint version; confirm in `/ingestor/docs`.)*

---

## 2. RAG Server (Query / Chat Completions)

Default port: **8081**. Base URL: `http://<RAG_HOST>` (e.g. `http://localhost:8081` or `http://localhost/rag` with proxy).

### 2.1 Health check

```http
GET http://<RAG_HOST>/v1/health?check_dependencies=true
```

**Example (curl)**

```bash
curl "http://localhost:8081/v1/health?check_dependencies=true"
```

---

### 2.2 RAG chat completions (query over your documents)

OpenAI-compatible endpoint: send a user message and get a model response grounded in retrieved documents.

**Request**

```http
POST http://<RAG_HOST>/v1/chat/completions
Content-Type: application/json
Accept: application/json
```

**Body (JSON)**

| Field                 | Type    | Required | Description |
|----------------------|---------|----------|-------------|
| `messages`           | array   | Yes      | Chat messages, e.g. `[{"role": "user", "content": "Your question here"}]`. |
| `use_knowledge_base` | boolean | Yes*     | Set `true` to use RAG (retrieve from collections). |
| `collection_names`   | array   | No*      | Collections to search, e.g. `["my_library"]`. Omit or empty to use server default. |
| `model`              | string  | No       | Model name (if the server supports multiple). |
| `max_tokens`         | number  | No       | Max tokens to generate (e.g. 150). |
| `temperature`        | number  | No       | 0.0–1.0, default often 0.0. |
| `top_p`              | number  | No       | 0.1–1.0, default often 0.1. |
| `stop`               | array   | No       | Stop sequences, e.g. `["\n"]`. |
| `stream`             | boolean | No       | If `true`, response is SSE stream. |

*For RAG you typically set `use_knowledge_base: true` and optionally `collection_names`.

**Example (curl) — single question, RAG on one collection**

```bash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the price of a hammer?"}],
    "use_knowledge_base": true,
    "collection_names": ["my_library"],
    "max_tokens": 150
  }'
```

**Example (curl) — via proxy**

```bash
curl -X POST "http://localhost/rag/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize the main points from the documents."}],
    "use_knowledge_base": true,
    "collection_names": ["my_library"],
    "max_tokens": 256,
    "temperature": 0.3
  }'
```

**Example (curl) — streaming (SSE)**

```bash
curl -N -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "messages": [{"role": "user", "content": "What is the price of a hammer?"}],
    "use_knowledge_base": true,
    "collection_names": ["my_library"],
    "max_tokens": 150,
    "stream": true
  }'
```

Response format is standard chat completion (e.g. `choices[].message.content`). The blueprint may add a `citations` field (e.g. in the first chunk when streaming) with retrieved document snippets.

---

## Quick reference table

| Action              | Server   | Method | Path (typical)              |
|---------------------|----------|--------|-----------------------------|
| Ingestor health     | Ingestor | GET    | `/v1/health`                |
| Upload documents    | Ingestor | POST   | `/v1/documents` (multipart) |
| Task status         | Ingestor | GET    | `/v1/documents/status/<id>` |
| Create collection   | Ingestor | POST   | `/v1/collection` (or similar) |
| List collections   | Ingestor | GET    | `/v1/collections`            |
| Get documents      | Ingestor | GET    | `/v1/documents`             |
| Update documents   | Ingestor | PUT    | `/v1/documents` (or similar) |
| Delete documents   | Ingestor | DELETE | `/v1/documents`             |
| Delete collection  | Ingestor | DELETE | `/v1/collections`           |
| RAG health         | RAG      | GET    | `/v1/health`                |
| RAG query          | RAG      | POST   | `/v1/chat/completions`      |

---

## Relation to this repo

- **NVIDIA Ingestor + RAG** are the upstream services; this doc describes their HTTP APIs.
- This repo’s **worker** (`nvidia-rag-worker`) wraps ingestion and generation:
  - `POST http://localhost:8123/ingest` — multipart ingest (see README “Direct worker calls”).
  - `POST http://localhost:8123/generate` — streaming chat with `messages`, `collection_names`, `use_knowledge_base`.
- When the full NVIDIA stack is used, the **proxy** exposes:
  - **Ingestor** at `http://localhost/ingestor/` → ingestor-server:8082
  - **RAG** at `http://localhost/rag/` → rag-server:8081  

So you can call either the **upstream APIs** (this doc) at `/ingestor/` and `/rag/`, or the **worker** at port 8123 (README).

---

## Official docs

- Ingestor OpenAPI: [API - Ingestor Server Schema](https://docs.nvidia.com/rag/2.3.0/api-ingestor.html)
- RAG OpenAPI: [API - RAG Server Schema](https://docs.nvidia.com/rag/2.3.0/api-rag.html)
- Blueprint overview: [NVIDIA RAG Blueprint](https://docs.nvidia.com/rag/2.3.0/index.html)
