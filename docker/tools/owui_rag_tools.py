import os
from typing import List, Optional

import asyncpg
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="OWUI RAG Tools",
    description="OpenAPI tool server for Open WebUI to control OWUI RAG pipeline state (library/allowlist).",
    version="1.0.0",
)


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://owui:owui@owui-postgres:5432/owui_bridge")
_pool: Optional[asyncpg.Pool] = None


async def _db() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        # Create tables (same schema as pipeline)
        async with _pool.acquire() as conn:
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
    return _pool


def _now() -> int:
    import time

    return int(time.time())


class LibrarySetRequest(BaseModel):
    user_key: str = Field(..., description="User identifier (e.g. user id or email)")
    chat_id: str = Field(..., description="Chat/conversation id")
    enabled: bool = Field(..., description="Whether to save new ingests to the user's library for this chat")


class LibraryStatusResponse(BaseModel):
    user_key: str
    chat_id: str
    enabled: bool


class AllowlistAddRequest(BaseModel):
    user_key: str = Field(..., description="User identifier (e.g. user id or email)")
    chat_id: str = Field(..., description="Chat/conversation id")
    collection_names: List[str] = Field(..., description="Collection names to allow for this chat")


class AllowlistResponse(BaseModel):
    user_key: str
    chat_id: str
    collection_names: List[str]


@app.get("/health")
async def health():
    pool = await _db()
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1;")
    return {"status": "ok"}


@app.post("/library/set", response_model=LibraryStatusResponse)
async def set_library(req: LibrarySetRequest):
    pool = await _db()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chat_settings (user_key, chat_id, save_to_library, updated_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_key, chat_id)
            DO UPDATE SET save_to_library=EXCLUDED.save_to_library, updated_at=EXCLUDED.updated_at
            """,
            req.user_key,
            req.chat_id,
            bool(req.enabled),
            _now(),
        )
        row = await conn.fetchrow(
            "SELECT save_to_library FROM chat_settings WHERE user_key=$1 AND chat_id=$2",
            req.user_key,
            req.chat_id,
        )
    return LibraryStatusResponse(user_key=req.user_key, chat_id=req.chat_id, enabled=bool(row["save_to_library"]) if row else False)


@app.get("/library/status", response_model=LibraryStatusResponse)
async def library_status(user_key: str, chat_id: str):
    pool = await _db()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT save_to_library FROM chat_settings WHERE user_key=$1 AND chat_id=$2",
            user_key,
            chat_id,
        )
    return LibraryStatusResponse(user_key=user_key, chat_id=chat_id, enabled=bool(row["save_to_library"]) if row else False)


@app.post("/allowlist/add", response_model=AllowlistResponse)
async def allowlist_add(req: AllowlistAddRequest):
    pool = await _db()
    now = _now()
    values = [(req.user_key, req.chat_id, c, now) for c in req.collection_names]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO chat_allowlist (user_key, chat_id, collection_name, added_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
            """,
            values,
        )
        rows = await conn.fetch(
            "SELECT collection_name FROM chat_allowlist WHERE user_key=$1 AND chat_id=$2",
            req.user_key,
            req.chat_id,
        )
    return AllowlistResponse(user_key=req.user_key, chat_id=req.chat_id, collection_names=[r["collection_name"] for r in rows])


@app.get("/allowlist/list", response_model=AllowlistResponse)
async def allowlist_list(user_key: str, chat_id: str):
    pool = await _db()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT collection_name FROM chat_allowlist WHERE user_key=$1 AND chat_id=$2",
            user_key,
            chat_id,
        )
    return AllowlistResponse(user_key=user_key, chat_id=chat_id, collection_names=[r["collection_name"] for r in rows])

