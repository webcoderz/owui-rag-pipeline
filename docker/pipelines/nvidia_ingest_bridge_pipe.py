"""
title: NVIDIA RAG (Worker • Postgres • Chat Allowlist • User Library • SSE)
author: Cody Webb
version: 1.2.0
requirements: httpx, asyncpg
"""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import tempfile
import threading
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


def _sse_chunk(model: str, content: str = "", role: Optional[str] = None) -> str:
    """Open WebUI Pipelines server expects lines starting with 'data:'; it adds '\\n\\n' itself."""
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
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}"


def _sse_done(_: str) -> str:
    """Pipelines server adds '\\n\\n'; yield only the event line."""
    return "data: [DONE]"


def _parse_worker_sse_line(model: str, line: str) -> Optional[str]:
    """
    Worker streams OpenAI-style SSE lines. Return event lines without trailing \\n\\n
    so the Pipelines server can add them (stream_content yields f"{line}\\n\\n").
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
    if line.startswith("{") and line.endswith("}"):
        return "data: " + line
    if line.startswith("data:"):
        return line.rstrip()
    return "data: " + line


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


def _extract_content_from_sse_chunk(line: str) -> Optional[str]:
    """Parse a 'data: {...}' SSE line and return choices[0].delta.content if present."""
    if not line or not line.startswith("data:"):
        return None
    rest = line[len("data:") :].strip()
    if not rest or rest == "[DONE]":
        return None
    try:
        obj = json.loads(rest)
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        c = delta.get("content")
        return c if isinstance(c, str) else None
    except Exception:
        return None


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
        # Open WebUI Pipelines persists "valves" to a JSON file and may not hydrate from
        # container env vars automatically. In production we want env vars to win.
        self._apply_env_valves_overrides()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_loop: Optional[asyncio.AbstractEventLoop] = None

    def _apply_env_valves_overrides(self) -> None:
        def set_if_env(name: str, caster=lambda x: x):
            v = os.getenv(name)
            if v is None or v == "":
                return
            try:
                setattr(self.valves, name, caster(v))
            except Exception:
                # best-effort; ignore bad env values
                return

        set_if_env("OPENWEBUI_BASE_URL", str)
        set_if_env("OPENWEBUI_API_KEY", str)
        set_if_env("NVIDIA_WORKER_URL", str)
        set_if_env("VDB_ENDPOINT", str)
        set_if_env("DATABASE_URL", str)
        set_if_env("MAX_PARALLEL_FILE_INGEST", int)
        set_if_env("PENDING_WAIT_SECONDS", int)
        set_if_env("CHUNK_SIZE", int)
        set_if_env("CHUNK_OVERLAP", int)

    def pipes(self):
        return [{"id": "nvidia-rag-auto-ingest", "name": "NVIDIA RAG (Auto-Ingest • Library • Persistent)"}]

    # -----------------------
    # Command parsing/helpers
    # -----------------------

    def _last_user_message_text(self, messages: List[dict]) -> str:
        """
        Extract plain text from the last user message. Handles both string content
        and Open WebUI multimodal list content ([{"type": "text", "text": "..."}, ...])
        so slash commands are always detected and never fall through.
        """
        if not messages or messages[-1].get("role") != "user":
            return ""
        content = messages[-1].get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]).strip())
            return " ".join(parts).strip()
        return str(content).strip()

    def _sanitize_collection_name(self, name: str) -> str:
        """
        Best-effort collection name sanitizer (for user input / display).
        Allows: letters, digits, '-', '_', '.', ':'.
        Also replaces whitespace with '-'.
        """
        s = (name or "").strip()
        if not s:
            return ""
        s = "-".join(s.split())
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
        s = "".join(ch for ch in s if ch in allowed)
        return s

    def _to_milvus_safe_collection_name(self, name: str) -> str:
        """
        Convert a collection name to Milvus-valid form.
        Milvus allows only letters, digits, underscore; must start with letter or underscore.
        We replace '-', '.', ':' with '_' and strip any other invalid chars.
        """
        if not (name or "").strip():
            return ""
        s = (name or "").strip()
        out = []
        for ch in s:
            if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_":
                out.append(ch)
            elif ch in "-.:":
                out.append("_")
            # else skip other chars
        result = "".join(out).strip("_") or ""
        if result and result[0].isdigit():
            result = "_" + result
        return result[:255] if result else ""

    def _looks_like_collection_token(self, token: str) -> bool:
        """
        Heuristic to avoid interpreting natural-language questions as `/query <collection> <question>`.
        We consider it a collection token if it contains typical collection punctuation or uses our prefix.
        """
        t = (token or "").strip()
        if not t:
            return False
        if t.startswith(f"{self.valves.COLLECTION_PREFIX}-"):
            return True
        return any(ch in t for ch in "-_:.")

    # -----------------------
    # DB init + helpers
    # -----------------------

    async def _db(self) -> asyncpg.Pool:
        # Pools are bound to the event loop they were created in. We run in different loops
        # per request (daemon thread's asyncio.run), so we must not reuse a pool whose loop
        # is closed — that causes "another operation in progress" and can segfault.
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if self._pool is not None and self._pool_loop is not None:
            if self._pool_loop.is_closed() or current_loop is not self._pool_loop:
                old = self._pool
                self._pool = None
                self._pool_loop = None
                try:
                    await old.close()
                except Exception as e:
                    logger.debug("Error closing stale asyncpg pool: %s", e)
        if not self._pool:
            self._pool = await asyncpg.create_pool(self.valves.DATABASE_URL, min_size=1, max_size=10)
            self._pool_loop = asyncio.get_running_loop()
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
    # Collection names (OWUI–Milvus linking and access)
    # -----------------------
    #
    # Collections are scoped so Milvus access aligns with Open WebUI access:
    # - owui-u-{user}-library  → user's personal library (only that user)
    # - owui-u-{user}-chat-{id} / owui-chat-{id}  → chat uploads (user-scoped or global)
    # - owui-kb-public-{kb_id} / owui-kb-{kb_id}  → OWUI knowledge base (access via OWUI KB permissions)
    # We use __request__'s Bearer token for OWUI API calls when present so OWUI enforces user/group
    # access; before query we filter collections by _filter_collections_by_owui_access (library/chat
    # must belong to current user, KB must be accessible via GET /api/v1/knowledge/{id}).

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

    def _parse_collection_name(self, name: str) -> Tuple[str, Any]:
        """
        Parse our collection name into type and payload for access checks.
        Returns (type, payload): type in ('library','chat','kb'), payload is type-specific.
        - library: payload = user_key (safe segment)
        - chat: payload = user_key or None (None if not user-scoped)
        - kb: payload = kb_id
        """
        if not name or not name.startswith(self.valves.COLLECTION_PREFIX):
            return ("unknown", None)
        rest = name[len(self.valves.COLLECTION_PREFIX) :].lstrip("-")
        if rest.startswith("u-") and rest.endswith("-library"):
            # owui-u-{safe}-library
            segment = rest[2 : -len("-library")].strip()
            return ("library", segment)
        if rest.startswith("u-") and "-chat-" in rest:
            # owui-u-{safe}-chat-{chat_id}
            segment = rest[2:].split("-chat-", 1)[0]
            return ("chat", segment)
        if rest.startswith("chat-"):
            return ("chat", None)
        if rest.startswith("kb-public-"):
            return ("kb", rest[len("kb-public-") :].strip())
        if rest.startswith("kb-"):
            return ("kb", rest[len("kb-") :].strip())
        return ("unknown", None)

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

    def _ow_headers(self, user_token: Optional[str] = None) -> Dict[str, str]:
        """Use user's Bearer token when provided (for OWUI access control); else service key."""
        if user_token and user_token.strip():
            return {"Authorization": user_token.strip()}
        if self.valves.OPENWEBUI_API_KEY:
            return {"Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}"}
        return {}

    async def _ow_get_json(self, client: httpx.AsyncClient, path: str, user_token: Optional[str] = None) -> dict:
        r = await client.get(
            f"{self.valves.OPENWEBUI_BASE_URL}{path}",
            headers=self._ow_headers(user_token),
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
        user_token: Optional[str] = None,
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
            headers=self._ow_headers(user_token),
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
        milvus_name = self._to_milvus_safe_collection_name(collection_name) or collection_name
        form = {
            "collection_name": milvus_name,
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

    async def _worker_get_existing_collections(self, client: httpx.AsyncClient) -> set:
        """Return set of Milvus collection names that exist (Milvus-safe names as in the DB)."""
        try:
            r = await client.get(
                f"{self.valves.NVIDIA_WORKER_URL}/collections",
                params={"vdb_endpoint": self.valves.VDB_ENDPOINT},
                timeout=10.0,
            )
            r.raise_for_status()
            data = r.json()
            names = data.get("collections") or []
            return set(n for n in names if n)
        except Exception:
            return set()

    async def _filter_to_existing_collections(
        self, client: httpx.AsyncClient, collection_names: List[str]
    ) -> List[str]:
        """Keep only collection names that exist in Milvus to avoid 'collection does not exist'."""
        if not collection_names:
            return collection_names
        existing = await self._worker_get_existing_collections(client)
        if not existing:
            return collection_names
        out = []
        for c in collection_names:
            milvus_name = self._to_milvus_safe_collection_name(c) or c
            if milvus_name in existing:
                out.append(c)
        return out

    async def _filter_collections_by_owui_access(
        self,
        ow_client: httpx.AsyncClient,
        collection_names: List[str],
        user_key: str,
        user_token: Optional[str] = None,
    ) -> List[str]:
        """
        Keep only collections the current user is allowed to use (OWUI–Milvus access alignment).
        - Library/chat: allow only if they belong to this user (user_key).
        - KB: allow only if GET /api/v1/knowledge/{kb_id} succeeds (OWUI enforces user/group access).
        """
        if not collection_names:
            return collection_names
        safe_user = self._safe_user_key(user_key)
        out = []
        for name in collection_names:
            ctype, payload = self._parse_collection_name(name)
            if ctype == "library":
                if payload == safe_user:
                    out.append(name)
                # else: different user's library, drop
                continue
            if ctype == "chat":
                if payload is None:
                    out.append(name)
                elif payload == safe_user:
                    out.append(name)
                continue
            if ctype == "kb" and payload:
                try:
                    await self._ow_get_json(ow_client, f"/api/v1/knowledge/{payload}", user_token=user_token)
                    out.append(name)
                except Exception:
                    # 403/404 or network: user has no access or KB gone
                    continue
                continue
            # unknown or no payload: allow (e.g. custom names); or drop for safety — allow to avoid breaking
            out.append(name)
        return out

    async def _stream_worker_generate(self, client: httpx.AsyncClient, messages: List[dict], collection_names: List[str]):
        milvus_names = [self._to_milvus_safe_collection_name(c) or c for c in collection_names]
        payload = {
            "messages": messages,
            "collection_names": milvus_names,
            "use_knowledge_base": bool(collection_names),
            "vdb_endpoint": self.valves.VDB_ENDPOINT,
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

    async def _call_worker_delete_documents(
        self,
        client: httpx.AsyncClient,
        collection_name: str,
        document_names: List[str],
    ) -> dict:
        """
        Delete documents from a collection via the worker's compatibility endpoint:
          DELETE /v1/documents?collection_name=...&vdb_endpoint=...
          body: ["doc1.pdf", "doc2.pdf"]
        """
        milvus_name = self._to_milvus_safe_collection_name(collection_name) or collection_name
        r = await client.request(
            "DELETE",
            f"{self.valves.NVIDIA_WORKER_URL}/v1/documents",
            params={"collection_name": milvus_name, "vdb_endpoint": self.valves.VDB_ENDPOINT},
            json=document_names,
            timeout=self.valves.OWUI_JSON_TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()

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
        user_token: Optional[str] = None,
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
                        ow_client, file_id, filename, emit, model_id, user_token=user_token
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
    # Open WebUI Pipelines may call pipe() without awaiting (sync call). We expose a sync pipe()
    # that runs the async implementation so both "await pipeline.pipe()" and "pipeline.pipe()" work.
    # When stream=True, we return a sync generator. The Pipelines server logs this as
    # "stream:true:<generator object ...>" — that is expected; the server then iterates it
    # and forwards SSE to the client.

    def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Any] = None,
        user_message: Optional[str] = None,
        **kwargs,
    ):
        req = __request__ if __request__ is not None else kwargs.get("__request__")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return asyncio.run(self._pipe_async(body, __user__=__user__, __request__=req, user_message=user_message, **kwargs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                asyncio.run,
                self._pipe_async(body, __user__=__user__, __request__=req, user_message=user_message, **kwargs),
            )
            return future.result()

    async def _pipe_async(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Any] = None,
        # Open WebUI Pipelines may pass extra kwargs depending on version.
        user_message: Optional[str] = None,
        **kwargs,
    ):
        user = __user__ or {}
        user_key = self._user_key(user)
        # Use requesting user's Bearer token for OWUI API when present (enforces KB/user access).
        user_token: Optional[str] = None
        if __request__ is not None and getattr(__request__, "headers", None):
            user_token = (__request__.headers.get("Authorization") or "").strip() or None
        model_id = body.get("model") or "nvidia-rag-auto-ingest"

        chat_id = body.get("chat_id") or body.get("conversation_id") or body.get("id") or str(_now())

        # Prefer runtime-provided `user_message` if present (newer Pipelines versions),
        # otherwise fall back to the last user message (handles string or multimodal list).
        messages = body.get("messages") or []
        last_user_text = self._last_user_message_text(messages)
        raw_text = ((user_message or last_user_text) or "").strip()
        text = raw_text.lower()
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

        def _sync_stream_from_async(async_gen):
            """
            Pipelines' /chat/completions streamer in some deployments does NOT iterate async generators.
            It expects a *sync* iterator yielding strings.
            We bridge an async generator -> sync generator using a background thread + queue.
            """
            q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=256)

            def _runner():
                async def _consume():
                    try:
                        async for item in async_gen:
                            q.put(item)
                    finally:
                        q.put(None)

                try:
                    asyncio.run(_consume())
                except Exception as e:
                    # Emit an error chunk so the client sees something.
                    try:
                        q.put(_sse_chunk(model_id, f"❌ Pipeline stream error: {e}\n"))
                    except Exception:
                        pass
                    q.put(_sse_done(model_id))
                    q.put(None)

            t = threading.Thread(target=_runner, daemon=True)
            t.start()

            def _iter():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    yield item

            return _iter()

        commands_text = (
            "Commands:\n"
            "- /collection list — show this chat’s known/remembered collections\n"
            "- /library on — save future ingests to your library\n"
            "- /library off — do not save future ingests to your library\n"
            "- /library — show this chat's current save-to-library setting\n"
            "- /ingest [collection] — ingest attachments/KBs and stop (optional target collection)\n"
            "- /query [collection] <question> — query remembered collections, or a specific collection\n"
            "- /delete <collection> <filename> — delete a document from a collection\n"
            "- /commands — show this help\n"
        )

        # For slash-commands, match the response type to the request:
        # - stream=true  -> return SSE generator (so OWUI renders it as streaming output)
        # - stream=false -> return a normal JSON completion
        if text in ("/commands", "/help", "/?"):
            if stream:
                async def cmd_stream():
                    if (os.getenv("PIPE_DEBUG", "").lower() in ("1", "true", "yes")):
                        logger.warning("cmd_stream: start")
                    yield _sse_chunk(model_id, role="assistant")
                    yield _sse_chunk(model_id, commands_text)
                    yield _sse_done(model_id)
                return _sync_stream_from_async(cmd_stream())
            return _json_completion(model_id, commands_text)

        # /collection list: show known collections for this chat/user
        if text in ("/collection list", "/collections", "/collections list"):
            async def collections_stream():
                yield _sse_chunk(model_id, role="assistant")
                try:
                    remembered = await self._allowlist_get(user_key, chat_id)
                    remembered = list(dict.fromkeys(remembered))
                    lib_collection = self._library_collection_name(user_key)
                    include_lib = await self._library_include_by_default(user_key)
                    save_to_library = await self._chat_get_save_to_library(user_key, chat_id)
                    chat_collection = self._chat_collection_name(chat_id, user)

                    lines: List[str] = []
                    lines.append("Collections:\n")
                    lines.append(f"- Chat uploads (derived): `{chat_collection}`\n")
                    lines.append(f"- Your library (derived): `{lib_collection}`\n")
                    lines.append(f"- Library included by default: {'ON' if include_lib else 'OFF'}\n")
                    lines.append(f"- Save-to-library for this chat: {'ON' if save_to_library else 'OFF'}\n")

                    if remembered:
                        lines.append("\nRemembered (chat allowlist):\n")
                        for c in remembered:
                            lines.append(f"- `{c}`\n")
                    else:
                        lines.append("\nRemembered (chat allowlist): (none yet)\n")

                    lines.append("\nTips:\n")
                    lines.append("- Use `/ingest <collection>` to ingest into a specific collection.\n")
                    lines.append("- Use `/query <collection> <question>` to query a specific collection.\n")
                    yield _sse_chunk(model_id, "".join(lines))
                except Exception as e:
                    yield _sse_chunk(model_id, f"❌ Collection list failed: {e}\n")
                yield _sse_done(model_id)

            if stream:
                return _sync_stream_from_async(collections_stream())
            return _json_completion(model_id, "Use stream=true for /collection list.")

        # /library helpers (per-chat toggle)
        if text in ("/library", "/library on", "/library off", "/library true", "/library false", "/library enable", "/library disable"):
            async def library_cmd_stream():
                yield _sse_chunk(model_id, role="assistant")
                try:
                    if text == "/library":
                        current = await self._chat_get_save_to_library(user_key, chat_id)
                        msg = (
                            "📚 Library setting for this chat: ON (new ingests will be saved to your library)\n"
                            if current
                            else "📚 Library setting for this chat: OFF (new ingests will NOT be saved to your library)\n"
                        )
                        yield _sse_chunk(model_id, msg)
                        yield _sse_done(model_id)
                        return

                    enable = text in ("/library on", "/library true", "/library enable")
                    await self._chat_set_save_to_library(user_key, chat_id, enable)
                    yield _sse_chunk(
                        model_id,
                        "✅ This chat will save new ingests to your library.\n" if enable else "✅ This chat will NOT save new ingests to your library.\n",
                    )
                except Exception as e:
                    yield _sse_chunk(model_id, f"❌ Library command failed: {e}\n")
                yield _sse_done(model_id)

            if stream:
                return _sync_stream_from_async(library_cmd_stream())

            # Non-stream is not supported reliably here (would require awaiting DB calls).
            # Return a friendly message instead of risking "asyncio.run() cannot be called from a running event loop".
            return _json_completion(model_id, "This command is supported in streaming mode. Please retry with stream=true.")

        # /ingest: ingest current attachments/KBs and stop (no generation)
        if text.startswith("/ingest"):
            raw_after = raw_text[len("/ingest") :].strip()
            # Allow a couple of friendly shorthands
            if raw_after.lower() == "chat":
                target_collection = self._chat_collection_name(chat_id, user)
            elif raw_after.lower() == "library":
                target_collection = self._library_collection_name(user_key)
            else:
                target_collection = self._sanitize_collection_name(raw_after) if raw_after else ""

            async def ingest_only_stream():
                yield _sse_chunk(model_id, role="assistant")

                if not self.valves.OPENWEBUI_API_KEY:
                    yield _sse_chunk(model_id, "❌ OPENWEBUI_API_KEY is not set. Cannot access OWUI files/knowledge.\n")
                    yield _sse_done(model_id)
                    return

                files = body.get("files") or []
                if not files:
                    yield _sse_chunk(model_id, "📎 No attachments/knowledge selected for ingestion.\n")
                    yield _sse_done(model_id)
                    return

                if raw_after and not target_collection:
                    yield _sse_chunk(model_id, "❌ Invalid collection name. Allowed: letters, digits, '-', '_', '.', ':'\n")
                    yield _sse_done(model_id)
                    return

                status_q: asyncio.Queue[str] = asyncio.Queue()

                async def emit(msg: str):
                    await status_q.put(_sse_chunk(model_id, msg))

                async def do_ingest():
                    async with httpx.AsyncClient() as ow_client, httpx.AsyncClient() as worker_client:
                        adhoc_file_ids, kb_ids = self._split_refs(files)
                        newly_used: List[str] = []

                        # KB files
                        for kb_id in kb_ids:
                            kb_meta = await self._ow_get_json(ow_client, f"/api/v1/knowledge/{kb_id}", user_token=user_token)
                            kb_collection = target_collection or self._kb_collection_name(kb_id, kb_meta)
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
                                ow_client, worker_client, entries, kb_collection, emit, model_id, user_token=user_token
                            )
                            newly_used.append(kb_collection)

                        # Ad-hoc files
                        if adhoc_file_ids:
                            chat_collection = target_collection or self._chat_collection_name(chat_id, user)
                            await emit(f"📥 Chat uploads → `{chat_collection}`\n")
                            entries2: List[Tuple[str, str]] = []
                            for fid in adhoc_file_ids:
                                meta = await self._ow_get_json(ow_client, f"/api/v1/files/{fid}", user_token=user_token)
                                fname = meta.get("filename") or meta.get("name") or fid
                                entries2.append((fid, fname))
                            await emit(f"📥 Uploads: {len(entries2)}\n")
                            await self._ingest_entries_into_collection(
                                ow_client, worker_client, entries2, chat_collection, emit, model_id, user_token=user_token
                            )
                            newly_used.append(chat_collection)

                        # Persist chat remember-set for what we just ingested (helps /query defaults)
                        if newly_used:
                            await self._allowlist_add(user_key, chat_id, list(dict.fromkeys(newly_used)))
                        # Clear acknowledgement so the user sees where docs were ingested
                        if newly_used:
                            await emit(f"✅ Ingested into collection(s): {', '.join(f'`{c}`' for c in newly_used)}. You can use /query or ask a question now.\n")
                        else:
                            await emit("✅ No new files to ingest (no attachments or KB selected).\n")

                task = asyncio.create_task(do_ingest())
                try:
                    while True:
                        try:
                            item = await asyncio.wait_for(status_q.get(), timeout=0.2)
                            yield item
                        except asyncio.TimeoutError:
                            if task.done():
                                break

                    # Drain remaining
                    while not status_q.empty():
                        yield await status_q.get()

                    # Propagate errors if any
                    exc = task.exception()
                    if exc:
                        yield _sse_chunk(model_id, f"❌ Ingestion failed: {exc}\n")
                    else:
                        yield _sse_chunk(model_id, "✅ Ingestion complete.\n")
                finally:
                    yield _sse_done(model_id)

            if stream:
                return _sync_stream_from_async(ingest_only_stream())
            return _json_completion(model_id, "Use stream=true for /ingest (streaming status).")

        # /query: run a question against specific collections without ingesting
        if text.startswith("/query"):
            # preserve original casing/content for the query
            raw_after = raw_text[len("/query") :].strip()
            if not raw_after:
                return _json_completion(model_id, "Usage: /query <question>")

            # Optional explicit collection: /query <collection> <question>
            explicit_collection: str = ""
            query = raw_after
            parts = raw_after.split(None, 1)
            if len(parts) == 2:
                token = parts[0].strip()
                rest = parts[1].strip()
                if rest and (token.lower() in ("chat", "library") or self._looks_like_collection_token(token)):
                    if token.lower() == "chat":
                        explicit_collection = self._chat_collection_name(chat_id, user)
                        query = rest
                    elif token.lower() == "library":
                        explicit_collection = self._library_collection_name(user_key)
                        query = rest
                    else:
                        sanitized = self._sanitize_collection_name(token)
                        if not sanitized:
                            return _json_completion(
                                model_id,
                                "Invalid collection name. Allowed: letters, digits, '-', '_', '.', ':'.",
                            )
                        explicit_collection = sanitized
                        query = rest

            async def query_stream():
                yield _sse_chunk(model_id, role="assistant")
                if not self.valves.OPENWEBUI_API_KEY:
                    # Not strictly needed for querying, but keep consistent error surface
                    yield _sse_chunk(model_id, "❌ OPENWEBUI_API_KEY is not set.\n")
                    yield _sse_done(model_id)
                    return

                # Determine collections:
                # - If explicit collection provided: only query that collection.
                # - Else: use remembered allowlist + (optional) library include by default.
                if explicit_collection:
                    collection_names: List[str] = [explicit_collection]
                else:
                    remembered = await self._allowlist_get(user_key, chat_id)
                    collection_names = list(dict.fromkeys(remembered))
                    lib_collection = self._library_collection_name(user_key)
                    if await self._library_include_by_default(user_key):
                        collection_names.append(lib_collection)
                        collection_names = list(dict.fromkeys(collection_names))

                # Build messages: replace last user message content with the query
                q_messages = list(messages)
                if q_messages and q_messages[-1].get("role") == "user":
                    q_messages[-1] = {"role": "user", "content": query}
                else:
                    q_messages.append({"role": "user", "content": query})

                async with httpx.AsyncClient() as ow_client:
                    collection_names = await self._filter_collections_by_owui_access(
                        ow_client, collection_names, user_key, user_token
                    )
                async with httpx.AsyncClient() as worker_client:
                    collection_names = await self._filter_to_existing_collections(worker_client, collection_names)
                    async for line in self._stream_worker_generate(worker_client, q_messages, collection_names):
                        parsed = _parse_worker_sse_line(model_id, line)
                        if parsed is not None:
                            yield parsed
                yield _sse_done(model_id)

            if stream:
                return _sync_stream_from_async(query_stream())
            return _json_completion(model_id, "Use stream=true for /query.")

        # /delete: delete a document from a collection (worker-backed)
        if text.startswith("/delete"):
            raw_after = raw_text[len("/delete") :].strip()
            parts = raw_after.split(None, 1) if raw_after else []
            if len(parts) < 2:
                msg = "Usage: /delete <collection> <filename>\nExample: /delete owui-u-me-library embedded_table.pdf\n"
                if stream:
                    async def del_usage_stream():
                        yield _sse_chunk(model_id, role="assistant")
                        yield _sse_chunk(model_id, msg)
                        yield _sse_done(model_id)
                    return _sync_stream_from_async(del_usage_stream())
                return _json_completion(model_id, msg)

            collection_token = parts[0].strip()
            filename = parts[1].strip()
            # strip simple surrounding quotes
            if (filename.startswith('"') and filename.endswith('"')) or (filename.startswith("'") and filename.endswith("'")):
                filename = filename[1:-1].strip()

            if collection_token.lower() == "chat":
                collection_name = self._chat_collection_name(chat_id, user)
            elif collection_token.lower() == "library":
                collection_name = self._library_collection_name(user_key)
            else:
                collection_name = self._sanitize_collection_name(collection_token)

            if not collection_name:
                msg = "❌ Invalid collection name. Allowed: letters, digits, '-', '_', '.', ':'\n"
                if stream:
                    async def del_bad_coll_stream():
                        yield _sse_chunk(model_id, role="assistant")
                        yield _sse_chunk(model_id, msg)
                        yield _sse_done(model_id)
                    return _sync_stream_from_async(del_bad_coll_stream())
                return _json_completion(model_id, msg)

            if not filename:
                msg = "❌ Missing filename.\nUsage: /delete <collection> <filename>\n"
                if stream:
                    async def del_bad_file_stream():
                        yield _sse_chunk(model_id, role="assistant")
                        yield _sse_chunk(model_id, msg)
                        yield _sse_done(model_id)
                    return _sync_stream_from_async(del_bad_file_stream())
                return _json_completion(model_id, msg)

            async def delete_stream():
                yield _sse_chunk(model_id, role="assistant")
                try:
                    async with httpx.AsyncClient() as worker_client:
                        resp = await self._call_worker_delete_documents(worker_client, collection_name, [filename])
                    yield _sse_chunk(model_id, f"🗑️ Delete request: `{filename}` from `{collection_name}`\n")
                    # Print a compact success message if available
                    if isinstance(resp, dict):
                        deleted = resp.get("documents") or []
                        if deleted:
                            yield _sse_chunk(model_id, f"✅ Deleted: {len(deleted)} document(s)\n")
                        else:
                            yield _sse_chunk(model_id, "✅ Delete request completed.\n")
                    else:
                        yield _sse_chunk(model_id, "✅ Delete request completed.\n")
                except Exception as e:
                    yield _sse_chunk(model_id, f"❌ Delete failed: {e}\n")
                yield _sse_done(model_id)

            if stream:
                return _sync_stream_from_async(delete_stream())
            return _json_completion(model_id, "Use stream=true for /delete.")

        if text.startswith("/"):
            msg = f"Unknown command: {text}. Try /commands."
            if stream:
                async def unknown_stream():
                    yield _sse_chunk(model_id, role="assistant")
                    yield _sse_chunk(model_id, msg + "\n")
                    yield _sse_done(model_id)
                return _sync_stream_from_async(unknown_stream())
            return _json_completion(model_id, msg)

        async def runner():
            if (os.getenv("PIPE_DEBUG", "").lower() in ("1", "true", "yes")):
                logger.warning("runner: start")
            yield _sse_chunk(model_id, role="assistant")

            # Command handling (these won't show as UI autocomplete; we respond explicitly)
            # (slash commands handled above via non-stream JSON return)

            if not self.valves.OPENWEBUI_API_KEY:
                yield _sse_chunk(model_id, "❌ OPENWEBUI_API_KEY is not set. Cannot access OWUI files/knowledge.\n")
                yield _sse_done(model_id)
                return

            status_q: asyncio.Queue[str] = asyncio.Queue()

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
                        kb_meta = await self._ow_get_json(ow_client, f"/api/v1/knowledge/{kb_id}", user_token=user_token)
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
                            ow_client, worker_client, entries, kb_collection, emit, model_id, user_token=user_token
                        )
                        newly_used.append(kb_collection)

                        if save_to_library:
                            await emit(f"📚 Saving KB docs to your library → `{lib_collection}`\n")
                            await self._ingest_entries_into_collection(
                                ow_client, worker_client, entries, lib_collection, emit, model_id, user_token=user_token
                            )

                    # Ad-hoc uploads
                    if adhoc_file_ids:
                        chat_collection = self._chat_collection_name(chat_id, user)
                        await emit(f"📥 Chat uploads → `{chat_collection}`\n")

                        entries: List[Tuple[str, str]] = []
                        for fid in adhoc_file_ids:
                            try:
                                meta = await self._ow_get_json(ow_client, f"/api/v1/files/{fid}", user_token=user_token)
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
                            ow_client, worker_client, entries, chat_collection, emit, model_id, user_token=user_token
                        )
                        newly_used.append(chat_collection)

                        if save_to_library:
                            await emit(f"📚 Saving uploads to your library → `{lib_collection}`\n")
                            await self._ingest_entries_into_collection(
                                ow_client, worker_client, entries, lib_collection, emit, model_id, user_token=user_token
                            )

                    await emit("✅ Ingestion complete.\n")
                    if newly_used:
                        await emit(f"📂 Ingested into: {', '.join(f'`{c}`' for c in newly_used)}\n")

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
                final_collections = await self._filter_collections_by_owui_access(
                    ow_client, final_collections, user_key, user_token
                )
                final_collections = await self._filter_to_existing_collections(worker_client, final_collections)
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
                    "- /collection list — show this chat’s known/remembered collections\n"
                    "- /library on — save future ingests to your library\n"
                    "- /library off — do not save future ingests to your library\n"
                    "- /library — show this chat's current save-to-library setting\n"
                    "- /ingest [collection] — ingest attachments/KBs and stop (optional target collection)\n"
                    "- /query [collection] <question> — query remembered collections, or a specific collection\n"
                    "- /delete <collection> <filename> — delete a document from a collection\n"
                    "- /commands — show this help\n",
                )
            if text.startswith("/"):
                return _json_completion(model_id, f"Unknown command: {text}. Try /commands.")
            # Non-stream request (e.g. follow-up suggestions or client sent stream=false):
            # run the same flow and collect all streamed content so the UI gets a real completion.
            content_parts: List[str] = []
            try:
                async for chunk in runner():
                    part = _extract_content_from_sse_chunk(chunk)
                    if part:
                        content_parts.append(part)
            except Exception as e:
                logger.exception("Non-stream runner collection failed")
                content_parts.append(f"❌ Error: {e}\n")
            return _json_completion(
                model_id,
                "".join(content_parts).strip() or "No response generated. Try again with streaming enabled, or check worker logs.",
            )

        # Stream path: return a *sync* generator (some Pipelines deployments don't iterate async generators).
        return _sync_stream_from_async(runner())
