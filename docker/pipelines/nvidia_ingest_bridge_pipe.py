"""
title: NVIDIA RAG (Worker • Postgres • Chat Allowlist • User Library • SSE)
author: Cody Webb
version: 1.1.0
requirements: httpx, asyncpg
"""

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


def _now() -> int:
    return int(time.time())


def _sse_chunk(model: str, content: str = "", role: Optional[str] = None) -> str:
    """
    OpenAI chat.completions streaming chunk (SSE).
    """
    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model or "nvidia-rag-auto-ingest",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


class Pipeline:
    """
    Open WebUI Pipelines Pipe.

    Responsibilities:
      - Resolve OWUI attachments + knowledge collections
      - Enforce collection scoping server-side
      - Persist chat allowlist + user library preferences in Postgres
      - Persist ingestion manifest in Postgres (dedupe + concurrency claims)
      - Call NVIDIA worker (py3.12) for ingest + generate
      - Stream status + model output (SSE) back to Open WebUI
    """

    class Valves(BaseModel):
        # Open WebUI
        OPENWEBUI_BASE_URL: str = Field(default="http://open-webui:8080")
        OPENWEBUI_API_KEY: str = Field(default="")  # service token (Bearer)

        # NVIDIA worker (py3.12) + Milvus endpoint
        NVIDIA_WORKER_URL: str = Field(default="http://nvidia-rag-worker:8123")
        VDB_ENDPOINT: str = Field(default="http://milvus:19530")

        # Postgres (persistent)
        DATABASE_URL: str = Field(default="postgresql://owui:owui_pw@postgres:5432/owui_bridge")

        # Naming/policy
        COLLECTION_PREFIX: str = Field(default="owui")
        USER_SCOPED_CHAT_COLLECTIONS: bool = Field(default=True)

        # Library behavior
        LIBRARY_ENABLED_DEFAULT: bool = Field(default=True)
        LIBRARY_INCLUDE_BY_DEFAULT: bool = Field(default=True)

        # Concurrency knobs
        MAX_PARALLEL_FILE_INGEST: int = Field(default=3)

        # Manifest "pending" wait (if another request claimed the same work)
        PENDING_WAIT_SECONDS: int = Field(default=180)
        PENDING_POLL_INTERVAL: float = Field(default=1.0)

        # Chunking defaults passed to worker
        CHUNK_SIZE: int = Field(default=512)
        CHUNK_OVERLAP: int = Field(default=150)

        # Timeouts
        OWUI_FILE_TIMEOUT_S: int = Field(default=600)
        WORKER_INGEST_TIMEOUT_S: int = Field(default=1800)
        WORKER_GENERATE_TIMEOUT_S: int = Field(default=600)

    def __init__(self):
        self.valves = self.Valves()
        self._pool: Optional[asyncpg.Pool] = None

    def pipes(self):
        return [{"id": "nvidia-rag-auto-ingest", "name": "NVIDIA RAG (Auto-Ingest • Library • Persistent)"}]

    # -----------------------
    # DB init + helpers
    # -----------------------

    async def _db(self) -> asyncpg.Pool:
        if not self._pool:
            self._pool = await asyncpg.create_pool(self.valves.DATABASE_URL, min_size=1, max_size=10)
            await self._init_db()
        return self._pool

    async def _init_db(self) -> None:
        """
        Creates required tables if missing (safe to call multiple times).
        """
        pool = self._pool
        if not pool:
            return
        async with pool.acquire() as conn:
            # Ingestion manifest for idempotency/dedupe + cross-request concurrency claims
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_manifest (
                  file_id TEXT NOT NULL,
                  collection_name TEXT NOT NULL,
                  sha256 TEXT NOT NULL,
                  status TEXT NOT NULL,  -- pending|success|failed
                  updated_at BIGINT NOT NULL,
                  PRIMARY KEY (file_id, collection_name, sha256)
                );
                """
            )

            # Per-chat allowlist of collections ("chat remembers what was attached/used")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_allowlist (
                  user_key TEXT NOT NULL,
                  chat_id TEXT NOT NULL,
                  collection_name TEXT NOT NULL,
                  added_at BIGINT NOT NULL,
                  PRIMARY KEY (user_key, chat_id, collection_name)
                );
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chat_allowlist_lookup ON chat_allowlist (user_key, chat_id);"
            )

            # User library settings
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_library_settings (
                  user_key TEXT PRIMARY KEY,
                  enabled BOOLEAN NOT NULL,
                  include_by_default BOOLEAN NOT NULL,
                  created_at BIGINT NOT NULL,
                  updated_at BIGINT NOT NULL
                );
                """
            )

    def _user_key(self, user: dict) -> str:
        return str(user.get("id") or user.get("email") or "anon")

    def _safe_user_key(self, user_key: str) -> str:
        return user_key.replace("@", "_").replace(":", "_").replace("/", "_")

    # -----------------------
    # OWUI naming policies
    # -----------------------

    def _kb_collection_name(self, kb_id: str, kb_meta: dict) -> str:
        # stable KB naming
        is_public = kb_meta.get("is_public")
        visibility = kb_meta.get("visibility")
        publicish = str(is_public).lower() in ("true", "1") or str(visibility).lower() == "public"
        suffix = "kb-public" if publicish else "kb"
        return f"{self.valves.COLLECTION_PREFIX}-{suffix}-{kb_id}"

    def _chat_collection_name(self, chat_id: str, user: dict) -> str:
        if not self.valves.USER_SCOPED_CHAT_COLLECTIONS:
            return f"{self.valves.COLLECTION_PREFIX}-chat-{chat_id}"
        user_key = self._safe_user_key(self._user_key(user))
        return f"{self.valves.COLLECTION_PREFIX}-u-{user_key}-chat-{chat_id}"

    def _library_collection_name(self, user_key: str) -> str:
        safe = self._safe_user_key(user_key)
        return f"{self.valves.COLLECTION_PREFIX}-u-{safe}-library"

    # -----------------------
    # Allowlist + library settings
    # -----------------------

    async def _allowlist_get(self, user_key: str, chat_id: str) -> List[str]:
        pool = await self._db()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT collection_name FROM chat_allowlist WHERE user_key=$1 AND chat_id=$2",
                user_key, chat_id
            )
        return [r["collection_name"] for r in rows]

    async def _allowlist_add(self, user_key: str, chat_id: str, collections: List[str]) -> None:
        if not collections:
            return
        pool = await self._db()
        now = _now()
        values = [(user_key, chat_id, c, now) for c in collections]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chat_allowlist (user_key, chat_id, collection_name, added_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                values
            )

    async def _library_include_by_default(self, user_key: str) -> bool:
        """
        Returns whether we should auto-include the user's library collection on every chat request.
        Creates a default row if missing.
        """
        pool = await self._db()
        now = _now()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT enabled, include_by_default FROM user_library_settings WHERE user_key=$1",
                user_key
            )
            if row:
                return bool(row["enabled"]) and bool(row["include_by_default"])

            # Create defaults
            await conn.execute(
                """
                INSERT INTO user_library_settings (user_key, enabled, include_by_default, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $4)
                """,
                user_key,
                bool(self.valves.LIBRARY_ENABLED_DEFAULT),
                bool(self.valves.LIBRARY_INCLUDE_BY_DEFAULT),
                now
            )
            return bool(self.valves.LIBRARY_ENABLED_DEFAULT) and bool(self.valves.LIBRARY_INCLUDE_BY_DEFAULT)

    # -----------------------
    # Ingest manifest (dedupe + claims)
    # -----------------------

    async def _manifest_get_status(self, file_id: str, collection: str, sha: str) -> Optional[str]:
        pool = await self._db()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM ingest_manifest WHERE file_id=$1 AND collection_name=$2 AND sha256=$3",
                file_id, collection, sha
            )
        return row["status"] if row else None

    async def _manifest_try_claim(self, file_id: str, collection: str, sha: str) -> bool:
        """
        Insert a 'pending' row if absent. True means we claimed and should ingest.
        False means someone else already has pending/success/failed for that sha.
        """
        pool = await self._db()
        async with pool.acquire() as conn:
            res = await conn.execute(
                """
                INSERT INTO ingest_manifest (file_id, collection_name, sha256, status, updated_at)
                VALUES ($1, $2, $3, 'pending', $4)
                ON CONFLICT (file_id, collection_name, sha256) DO NOTHING
                """,
                file_id, collection, sha, _now()
            )
        return res.endswith("1")  # "INSERT 0 1" or "INSERT 0 0"

    async def _manifest_set(self, file_id: str, collection: str, sha: str, status: str) -> None:
        pool = await self._db()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE ingest_manifest
                SET status=$4, updated_at=$5
                WHERE file_id=$1 AND collection_name=$2 AND sha256=$3
                """,
                file_id, collection, sha, status, _now()
            )

    async def _wait_for_terminal(self, file_id: str, collection: str, sha: str) -> Optional[str]:
        deadline = time.time() + self.valves.PENDING_WAIT_SECONDS
        while time.time() < deadline:
            st = await self._manifest_get_status(file_id, collection, sha)
            if st in ("success", "failed"):
                return st
            await asyncio.sleep(self.valves.PENDING_POLL_INTERVAL)
        return await self._manifest_get_status(file_id, collection, sha)

    # -----------------------
    # OWUI API helpers
    # -----------------------

    def _ow_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}"} if self.valves.OPENWEBUI_API_KEY else {}

    async def _ow_get_json(self, client: httpx.AsyncClient, path: str) -> dict:
        r = await client.get(
            f"{self.valves.OPENWEBUI_BASE_URL}{path}",
            headers=self._ow_headers(),
            timeout=60
        )
        r.raise_for_status()
        return r.json()

    async def _ow_get_bytes(self, client: httpx.AsyncClient, file_id: str) -> bytes:
        r = await client.get(
            f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/files/{file_id}/content",
            headers=self._ow_headers(),
            params={"attachment": "false"},
            timeout=self.valves.OWUI_FILE_TIMEOUT_S
        )
        r.raise_for_status()
        return r.content

    def _split_refs(self, files: List[dict]) -> Tuple[List[str], List[str]]:
        adhoc, kb_ids = [], []
        for f in files:
            t, fid = f.get("type"), f.get("id")
            if not fid:
                continue
            if t == "file":
                adhoc.append(fid)
            elif t == "collection":
                kb_ids.append(fid)
        return adhoc, kb_ids

    # -----------------------
    # Worker calls
    # -----------------------

    async def _call_worker_ingest(
        self,
        client: httpx.AsyncClient,
        collection_name: str,
        filename: str,
        data: bytes,
    ) -> dict:
        files = {"file": (filename, data)}
        form = {
            "collection_name": collection_name,
            "vdb_endpoint": self.valves.VDB_ENDPOINT,
            "blocking": "true",
            "chunk_size": str(self.valves.CHUNK_SIZE),
            "chunk_overlap": str(self.valves.CHUNK_OVERLAP),
            "generate_summary": "false",
        }
        r = await client.post(
            f"{self.valves.NVIDIA_WORKER_URL}/ingest",
            data=form,
            files=files,
            timeout=self.valves.WORKER_INGEST_TIMEOUT_S
        )
        r.raise_for_status()
        return r.json()

    async def _stream_worker_generate(
        self,
        client: httpx.AsyncClient,
        messages: List[dict],
        collection_names: List[str],
    ):
        payload = {
            "messages": messages,
            "collection_names": collection_names,
            "use_knowledge_base": bool(collection_names),
        }

        async with client.stream(
            "POST",
            f"{self.valves.NVIDIA_WORKER_URL}/generate",
            json=payload,
            timeout=self.valves.WORKER_GENERATE_TIMEOUT_S
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                yield line

    # -----------------------
    # Ingest orchestration
    # -----------------------

    async def _ingest_entries_into_collection(
        self,
        ow_client: httpx.AsyncClient,
        worker_client: httpx.AsyncClient,
        entries: List[Tuple[str, str]],  # (file_id, filename)
        collection_name: str,
        emit,
    ) -> None:
        sem = asyncio.Semaphore(self.valves.MAX_PARALLEL_FILE_INGEST)

        async def one(file_id: str, filename: str):
            async with sem:
                await emit(f"🔎 Processing `{filename}`…\n")
                data = await self._ow_get_bytes(ow_client, file_id)
                sha = hashlib.sha256(data).hexdigest()

                st = await self._manifest_get_status(file_id, collection_name, sha)
                if st == "success":
                    await emit(f"↩️ Skip (already indexed): `{filename}`\n")
                    return

                claimed = await self._manifest_try_claim(file_id, collection_name, sha)
                if not claimed:
                    await emit(f"⏳ Another request is indexing `{filename}`…\n")
                    terminal = await self._wait_for_terminal(file_id, collection_name, sha)
                    if terminal == "success":
                        await emit(f"✅ Ready: `{filename}`\n")
                    else:
                        await emit(f"❌ Indexing failed (other request): `{filename}`\n")
                    return

                try:
                    await emit(f"📤 Uploading `{filename}` to NVIDIA…\n")
                    await self._call_worker_ingest(worker_client, collection_name, filename, data)
                    await self._manifest_set(file_id, collection_name, sha, "success")
                    await emit(f"✅ Indexed: `{filename}`\n")
                except Exception:
                    await self._manifest_set(file_id, collection_name, sha, "failed")
                    raise

        await asyncio.gather(*(one(fid, fname) for fid, fname in entries))

    # -----------------------
    # Main pipe
    # -----------------------

    async def pipe(self, body: dict, __user__: Optional[dict] = None):
        user = __user__ or {}
        user_key = self._user_key(user)
        model_id = body.get("model") or "nvidia-rag-auto-ingest"

        # Stable chat_id is important for per-chat memory
        chat_id = body.get("chat_id")
        if not chat_id:
            chat_id = body.get("conversation_id") or body.get("id") or str(_now())

        # Optional flag: if true, also ingest attachments into per-user library collection
        # (You can later wire a UI toggle or action that sets this.)
        save_to_library = bool(body.get("save_to_library", False))


        async def runner():

            yield _sse_chunk(model_id, role="assistant")
            if not self.valves.OPENWEBUI_API_KEY:
                yield _sse_chunk(model_id, "❌ OPENWEBUI_API_KEY is not set. Pipeline cannot access OWUI files/knowledge.\n")
                yield "data: [DONE]\n\n"
                return

            # Status queue so we can interleave status with worker stream cleanly
            status_q: asyncio.Queue[str] = asyncio.Queue()

            async def emit(msg: str):
                await status_q.put(_sse_chunk(model_id, msg))

            async with httpx.AsyncClient() as ow_client, httpx.AsyncClient() as worker_client:
                # 1) Start with per-chat remembered collections
                remembered = await self._allowlist_get(user_key, chat_id)
                collection_names: List[str] = list(dict.fromkeys(remembered))

                # 2) Optionally auto-include user library across chats
                if await self._library_include_by_default(user_key):
                    lib_collection = self._library_collection_name(user_key)
                    collection_names.append(lib_collection)
                    collection_names = list(dict.fromkeys(collection_names))

                # 3) Handle attachments (ingest + update allowlist)
                files = body.get("files") or []
                newly_used: List[str] = []

                if files:
                    await emit("📎 Attachments detected. Preparing ingestion…\n")

                    adhoc_file_ids, kb_ids = self._split_refs(files)

                    # Knowledge collections -> per-KB collections
                    for kb_id in kb_ids:
                        kb_meta = await self._ow_get_json(ow_client, f"/api/v1/knowledge/{kb_id}")
                        kb_collection = self._kb_collection_name(kb_id, kb_meta)
                        await emit(f"📚 Knowledge `{kb_id}` → `{kb_collection}`\n")

                        kb_files = kb_meta.get("files") or []
                        entries: List[Tuple[str, str]] = []
                        for f in kb_files:
                            fid = f.get("id") or f.get("file_id")
                            if not fid:
                                continue
                            fname = f.get("filename") or f.get("name") or fid
                            entries.append((fid, fname))

                        await emit(f"📥 KB files: {len(entries)}\n")
                        await self._ingest_entries_into_collection(ow_client, worker_client, entries, kb_collection, emit)

                        newly_used.append(kb_collection)

                        # Optional: also save KB docs to user library
                        if save_to_library:
                            lib_collection = self._library_collection_name(user_key)
                            await emit(f"📚 Saving KB docs to your library → `{lib_collection}`\n")
                            await self._ingest_entries_into_collection(ow_client, worker_client, entries, lib_collection, emit)

                    # Ad-hoc uploads -> per-chat collection
                    if adhoc_file_ids:
                        chat_collection = self._chat_collection_name(chat_id, user)
                        await emit(f"📥 Chat uploads → `{chat_collection}`\n")

                        entries: List[Tuple[str, str]] = []
                        for fid in adhoc_file_ids:
                            try:
                                meta = await self._ow_get_json(ow_client, f"/api/v1/files/{fid}")
                                fname = meta.get("filename") or meta.get("name") or fid

                                size = meta.get("size") or meta.get("file_size")
                                if size and int(size) > 200 * 1024 * 1024:
                                    await emit(f"❌ File too large (>200MB): `{fname}`\n")
                                    return

                            except Exception:
                                fname = fid
                            entries.append((fid, fname))

                        await emit(f"📥 Uploads: {len(entries)}\n")
                        await self._ingest_entries_into_collection(ow_client, worker_client, entries, chat_collection, emit)

                        newly_used.append(chat_collection)

                        if save_to_library:
                            lib_collection = self._library_collection_name(user_key)
                            await emit(f"📚 Saving uploads to your library → `{lib_collection}`\n")
                            await self._ingest_entries_into_collection(ow_client, worker_client, entries, lib_collection, emit)

                    await emit("✅ Ingestion complete.\n")

                    # Persist newly used collections for this chat
                    if newly_used:
                        await self._allowlist_add(user_key, chat_id, list(dict.fromkeys(newly_used)))
                        collection_names = list(dict.fromkeys(collection_names + newly_used))

                # 4) Query phase: always compute final allowlisted collection_names server-side
                # If you want “no files selected => no private retrieval”, you can add a policy here.
                await emit("💬 Querying…\n")

                messages = body.get("messages") or []

                # Drain any queued status before we start streaming tokens
                while not status_q.empty():
                    yield await status_q.get()

                async for line in self._stream_worker_generate(worker_client, messages, list(dict.fromkeys(collection_names))):
                    # keep status responsive
                    while not status_q.empty():
                        yield await status_q.get()

                    line = line.strip()
                    if not line:
                        continue
                    # Pass through SSE lines
                    if line.startswith("data:"):
                        yield line + "\n\n"
                    else:
                        yield "data: " + line + "\n\n"

                # Final drain
                while not status_q.empty():
                    yield await status_q.get()

                yield "data: [DONE]\n\n"

        return StreamingResponse(runner(), media_type="text/event-stream")
