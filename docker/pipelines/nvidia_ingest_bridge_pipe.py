"""
title: NVIDIA RAG (Worker • Postgres • Chat Allowlist • User Library • SSE)
author: Cody Webb
version: 1.2.0
requirements: httpx, asyncpg
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _now() -> int:
    return int(time.time())


def _sse_chunk(model: str, content: str = "", role: Optional[str] = None) -> dict:
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
    return payload


def _sse_done(model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model or "nvidia-rag-auto-ingest",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


def _parse_worker_sse_line(model: str, line: str) -> Optional[dict]:
    """
    Worker streams OpenAI-style SSE lines. The Pipelines runtime usually handles SSE framing,
    so we convert lines to chunk dicts here.
    """
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    if line.startswith("data:"):
        line = line[len("data:") :].strip()
    if not line or line == "[DONE]":
        return _sse_done(model)
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: treat as plain text content
    return _sse_chunk(model, content=line)


def _json_completion(model: str, content: str) -> dict:
    """Non-stream OpenAI-style response payload."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": _now(),
        "model": model or "nvidia-rag-auto-ingest",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class Pipeline:
    class Valves(BaseModel):
        # Open WebUI
        OPENWEBUI_BASE_URL: str = Field(default="http://open-webui:8080")
        OPENWEBUI_API_KEY: str = Field(default="")  # service token (Bearer)

        # NVIDIA worker + Milvus
        NVIDIA_WORKER_URL: str = Field(default="http://nvidia-rag-worker:8123")
        VDB_ENDPOINT: str = Field(default="http://milvus:19530")

        # Postgres
        DATABASE_URL: str = Field(default="postgresql://owui:owui@owui-postgres:5432/owui_bridge")

        # Naming/policy
        COLLECTION_PREFIX: str = Field(default="owui")
        USER_SCOPED_CHAT_COLLECTIONS: bool = Field(default=True)

        # Library defaults
        LIBRARY_ENABLED_DEFAULT: bool = Field(default=True)
        LIBRARY_INCLUDE_BY_DEFAULT: bool = Field(default=True)

        # Concurrency knobs
        MAX_PARALLEL_FILE_INGEST: int = Field(default=3)

        # Manifest pending wait
        PENDING_WAIT_SECONDS: int = Field(default=180)
        PENDING_POLL_INTERVAL: float = Field(default=1.0)

        # Chunking defaults
        CHUNK_SIZE: int = Field(default=512)
        CHUNK_OVERLAP: int = Field(default=150)

        # Timeouts
        OWUI_JSON_TIMEOUT_S: int = Field(default=60)
        OWUI_STREAM_TIMEOUT_S: int = Field(default=600)
        WORKER_INGEST_TIMEOUT_S: int = Field(default=1800)
        WORKER_GENERATE_TIMEOUT_S: int = Field(default=600)

        # Limits
        MAX_FILE_BYTES: int = Field(default=200 * 1024 * 1024)  # 200MB

        # Progress throttling
        PROGRESS_EMIT_EVERY_PCT: int = Field(default=10)

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
        pool = self._pool
        if not pool:
            return
        async with pool.acquire() as conn:
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
            await conn.execute("CREATE INDEX IF NOT EXISTS chat_allowlist_lookup ON chat_allowlist (user_key, chat_id);")
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
            # per-chat settings (save_to_library toggle etc.)
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_settings (
                  user_key TEXT NOT NULL,
                  chat_id TEXT NOT NULL,
                  save_to_library BOOLEAN NOT NULL DEFAULT FALSE,
                  updated_at BIGINT NOT NULL,
                  PRIMARY KEY (user_key, chat_id)
                );
                """
            )

    def _user_key(self, user: dict) -> str:
        return str(user.get("id") or user.get("email") or "anon")

    def _safe_user_key(self, user_key: str) -> str:
        return user_key.replace("@", "_").replace(":", "_").replace("/", "_")

    # -----------------------
    # Collection names
    # -----------------------

    def _kb_collection_name(self, kb_id: str, kb_meta: dict) -> str:
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
        pool = await self._db()
        now = _now()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT enabled, include_by_default FROM user_library_settings WHERE user_key=$1",
                user_key
            )
            if row:
                return bool(row["enabled"]) and bool(row["include_by_default"])
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

    async def _chat_get_save_to_library(self, user_key: str, chat_id: str) -> bool:
        pool = await self._db()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT save_to_library FROM chat_settings WHERE user_key=$1 AND chat_id=$2",
                user_key, chat_id
            )
        return bool(row["save_to_library"]) if row else False

    async def _chat_set_save_to_library(self, user_key: str, chat_id: str, value: bool) -> None:
        pool = await self._db()
        now = _now()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_settings (user_key, chat_id, save_to_library, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_key, chat_id)
                DO UPDATE SET save_to_library=EXCLUDED.save_to_library, updated_at=EXCLUDED.updated_at
                """,
                user_key, chat_id, value, now
            )

    # -----------------------
    # Ingest manifest
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
        return res.endswith("1")

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
            timeout=self.valves.OWUI_JSON_TIMEOUT_S
        )
        r.raise_for_status()
        return r.json()

    async def _download_to_tempfile_and_hash(
        self,
        client: httpx.AsyncClient,
        file_id: str,
        filename: str,
        emit,
        model_id: str,
    ) -> Tuple[str, str, int]:
        """
        Streams OWUI file content to a temp file to avoid RAM spikes.
        Returns: (tmp_path, sha256_hex, byte_count)
        """
        url = f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/files/{file_id}/content"
        h = hashlib.sha256()
        byte_count = 0
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1] or "")
        tmp_path = tmp.name
        tmp.close()

        last_emit_pct = -1

        async with client.stream(
            "GET",
            url,
            headers=self._ow_headers(),
            params={"attachment": "false"},
            timeout=self.valves.OWUI_STREAM_TIMEOUT_S
        ) as r:
            r.raise_for_status()
            cl = r.headers.get("content-length")
            total = int(cl) if cl and cl.isdigit() else None
            if total and total > self.valves.MAX_FILE_BYTES:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise ValueError(f"File too large (> {self.valves.MAX_FILE_BYTES} bytes)")

            with open(tmp_path, "wb") as f:
                async for chunk in r.aiter_bytes():
                    if not chunk:
                        continue
                    f.write(chunk)
                    h.update(chunk)
                    byte_count += len(chunk)

                    if byte_count > self.valves.MAX_FILE_BYTES:
                        raise ValueError(f"File too large (> {self.valves.MAX_FILE_BYTES} bytes)")

                    if total:
                        pct = int((byte_count / total) * 100)
                        step = self.valves.PROGRESS_EMIT_EVERY_PCT
                        pct_bucket = (pct // step) * step
                        if pct_bucket != last_emit_pct and pct_bucket in (0, step, 2*step, 3*step, 4*step, 5*step, 6*step, 7*step, 8*step, 9*step, 100):
                            await emit(f"⬇️ Download `{filename}`… {pct_bucket}%\n")
                            last_emit_pct = pct_bucket

        return tmp_path, h.hexdigest(), byte_count

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

    async def _call_worker_ingest_from_path(
        self,
        client: httpx.AsyncClient,
        collection_name: str,
        filename: str,
        tmp_path: str,
    ) -> dict:
        form = {
            "collection_name": collection_name,
            "vdb_endpoint": self.valves.VDB_ENDPOINT,
            "blocking": "true",
            "chunk_size": str(self.valves.CHUNK_SIZE),
            "chunk_overlap": str(self.valves.CHUNK_OVERLAP),
            "generate_summary": "false",
        }
        with open(tmp_path, "rb") as f:
            files = {"file": (filename, f)}
            r = await client.post(
                f"{self.valves.NVIDIA_WORKER_URL}/ingest",
                data=form,
                files=files,
                timeout=self.valves.WORKER_INGEST_TIMEOUT_S
            )
        r.raise_for_status()
        return r.json()

    async def _stream_worker_generate(self, client: httpx.AsyncClient, messages: List[dict], collection_names: List[str]):
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
                if line:
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
        model_id: str,
    ) -> None:
        sem = asyncio.Semaphore(self.valves.MAX_PARALLEL_FILE_INGEST)
        total = len(entries)
        done = 0
        done_lock = asyncio.Lock()

        async def one(file_id: str, filename: str):
            nonlocal done
            async with sem:
                await emit(f"🔎 Processing `{filename}`…\n")

                tmp_path = None
                sha = None
                try:
                    tmp_path, sha, _ = await self._download_to_tempfile_and_hash(
                        ow_client, file_id, filename, emit, model_id
                    )

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

                    await emit(f"📤 Uploading `{filename}` to RAG…\n")
                    await self._call_worker_ingest_from_path(worker_client, collection_name, filename, tmp_path)
                    await self._manifest_set(file_id, collection_name, sha, "success")
                    await emit(f"✅ Indexed: `{filename}`\n")

                except ValueError as ve:
                    # size limit etc.
                    await emit(f"❌ {ve} — `{filename}`\n")
                    if sha:
                        await self._manifest_set(file_id, collection_name, sha, "failed")
                except Exception:
                    if sha:
                        await self._manifest_set(file_id, collection_name, sha, "failed")
                    raise
                finally:
                    if tmp_path:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                    async with done_lock:
                        done += 1
                        pct = int((done / max(total, 1)) * 100)
                        await emit(f"📦 Ingest progress: {done}/{total} ({pct}%)\n")

        await asyncio.gather(*(one(fid, fname) for fid, fname in entries))

    # -----------------------
    # Main pipe
    # -----------------------

    def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        # Open WebUI Pipelines may pass extra kwargs depending on version.
        # Accept them to avoid runtime crashes like:
        #   TypeError: Pipeline.pipe() got unexpected keyword argument 'user_message'
        user_message: Optional[str] = None,
        **kwargs,
    ):
        user = __user__ or {}
        user_key = self._user_key(user)
        model_id = body.get("model") or "nvidia-rag-auto-ingest"

        chat_id = body.get("chat_id") or body.get("conversation_id") or body.get("id") or str(_now())

        # Prefer runtime-provided `user_message` if present (newer Pipelines versions),
        # otherwise fall back to the last user message in the chat payload.
        messages = body.get("messages") or []
        last_user_text = ""
        if messages and (messages[-1].get("role") == "user"):
            last_user_text = (messages[-1].get("content") or "")
        text = ((user_message or last_user_text) or "").strip().lower()
        stream = bool(body.get("stream", True))

        # Minimal debug to confirm OWUI payload shape without logging secrets/content.
        # Enable by setting PIPE_DEBUG=true in the pipelines container env.
        if (os.getenv("PIPE_DEBUG", "").lower() in ("1", "true", "yes")):
            logger.warning(
                "pipe called stream=%s text=%r body_keys=%s",
                stream,
                text,
                sorted(list(body.keys())),
            )

        commands_text = (
            "Commands:\n"
            "- /library on — save future ingests to your library\n"
            "- /library off — do not save future ingests to your library\n"
            "- /library — show this chat's current save-to-library setting\n"
            "- /commands — show this help\n"
        )

        # For slash-commands, match the response type to the request:
        # - stream=true  -> return SSE generator (so OWUI renders it as streaming output)
        # - stream=false -> return a normal JSON completion
        if text in ("/commands", "/help", "/?"):
            if stream:
                async def cmd_stream():
                    yield _sse_chunk(model_id, role="assistant")
                    yield _sse_chunk(model_id, commands_text)
                    yield _sse_done(model_id)
                return cmd_stream()
            return _json_completion(model_id, commands_text)

        if text.startswith("/"):
            msg = f"Unknown command: {text}. Try /commands."
            if stream:
                async def unknown_stream():
                    yield _sse_chunk(model_id, role="assistant")
                    yield _sse_chunk(model_id, msg + "\n")
                    yield _sse_done(model_id)
                return unknown_stream()
            return _json_completion(model_id, msg)

        async def runner():
            yield _sse_chunk(model_id, role="assistant")

            # Command handling (these won't show as UI autocomplete; we respond explicitly)
            # (slash commands handled above via non-stream JSON return)

            if not self.valves.OPENWEBUI_API_KEY:
                yield _sse_chunk(model_id, "❌ OPENWEBUI_API_KEY is not set. Cannot access OWUI files/knowledge.\n")
                yield _sse_done(model_id)
                return

            status_q: asyncio.Queue[dict] = asyncio.Queue()

            async def emit(msg: str):
                await status_q.put(_sse_chunk(model_id, msg))

            async with httpx.AsyncClient() as ow_client, httpx.AsyncClient() as worker_client:
                # Save-to-library decision:
                # - explicit request field wins
                # - else use per-chat setting
                save_to_library = bool(body.get("save_to_library", await self._chat_get_save_to_library(user_key, chat_id)))

                # 1) Seed collections from remembered allowlist
                remembered = await self._allowlist_get(user_key, chat_id)
                collection_names: List[str] = list(dict.fromkeys(remembered))

                # 2) Auto-include per-user library across chats (if enabled)
                lib_collection = self._library_collection_name(user_key)
                if await self._library_include_by_default(user_key):
                    collection_names.append(lib_collection)
                    collection_names = list(dict.fromkeys(collection_names))

                # 3) Attachments
                files = body.get("files") or []
                newly_used: List[str] = []

                if files:
                    await emit("📎 Attachments detected. Preparing ingestion…\n")
                    adhoc_file_ids, kb_ids = self._split_refs(files)

                    # Knowledge collections
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
                        await self._ingest_entries_into_collection(
                            ow_client, worker_client, entries, kb_collection, emit, model_id
                        )
                        newly_used.append(kb_collection)

                        if save_to_library:
                            await emit(f"📚 Saving KB docs to your library → `{lib_collection}`\n")
                            await self._ingest_entries_into_collection(
                                ow_client, worker_client, entries, lib_collection, emit, model_id
                            )

                    # Ad-hoc uploads
                    if adhoc_file_ids:
                        chat_collection = self._chat_collection_name(chat_id, user)
                        await emit(f"📥 Chat uploads → `{chat_collection}`\n")

                        entries: List[Tuple[str, str]] = []
                        for fid in adhoc_file_ids:
                            try:
                                meta = await self._ow_get_json(ow_client, f"/api/v1/files/{fid}")
                                fname = meta.get("filename") or meta.get("name") or fid
                                size = meta.get("size") or meta.get("file_size")
                                if size and int(size) > self.valves.MAX_FILE_BYTES:
                                    await emit(f"❌ File too large (>200MB): `{fname}`\n")
                                    continue
                            except Exception:
                                fname = fid
                            entries.append((fid, fname))

                        await emit(f"📥 Uploads: {len(entries)}\n")
                        await self._ingest_entries_into_collection(
                            ow_client, worker_client, entries, chat_collection, emit, model_id
                        )
                        newly_used.append(chat_collection)

                        if save_to_library:
                            await emit(f"📚 Saving uploads to your library → `{lib_collection}`\n")
                            await self._ingest_entries_into_collection(
                                ow_client, worker_client, entries, lib_collection, emit, model_id
                            )

                    await emit("✅ Ingestion complete.\n")

                    # Persist chat remember-set
                    if newly_used:
                        await self._allowlist_add(user_key, chat_id, list(dict.fromkeys(newly_used)))
                        collection_names = list(dict.fromkeys(collection_names + newly_used))

                # 4) Query
                await emit("💬 Querying…\n")

                # Drain any queued status before streaming tokens
                while not status_q.empty():
                    yield await status_q.get()

                final_collections = list(dict.fromkeys(collection_names))
                async for line in self._stream_worker_generate(worker_client, messages, final_collections):
                    while not status_q.empty():
                        yield await status_q.get()
                    line = line.strip()
                    if not line:
                        continue
                    parsed = _parse_worker_sse_line(model_id, line)
                    if parsed is not None:
                        yield parsed

                while not status_q.empty():
                    yield await status_q.get()

                yield _sse_done(model_id)

        # IMPORTANT: Open WebUI Pipelines expects the Pipeline.pipe() method to return either:
        # - a plain dict for non-stream responses, OR
        # - a (async) generator for stream responses.
        # Returning a Starlette/FastAPI Response object here can be ignored by the Pipelines runtime.
        if not stream:
            if text in ("/commands", "/help", "/?"):
                return _json_completion(
                    model_id,
                    "Commands:\n"
                    "- /library on — save future ingests to your library\n"
                    "- /library off — do not save future ingests to your library\n"
                    "- /library — show this chat's current save-to-library setting\n"
                    "- /commands — show this help\n",
                )
            if text.startswith("/"):
                return _json_completion(model_id, f"Unknown command: {text}. Try /commands.")
            # Default non-stream fallback: tell OWUI we stream by default.
            return _json_completion(model_id, "This pipeline is configured for streaming responses.")

        # Stream path: return the async generator directly (Pipelines runtime will stream it).
        return runner()
