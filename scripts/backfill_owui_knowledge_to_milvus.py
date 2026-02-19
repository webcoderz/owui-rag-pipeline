#!/usr/bin/env python3
"""
Backfill existing Open WebUI knowledge bases into Milvus.

Use this to ingest documents from OWUI knowledge bases that were created
before you integrated this pipeline, so RAG queries can search that content.

- By default skips documents already in the collection (safe to run multiple times).
- Use --confirm when not using --dry-run so a mistake does not ingest by accident.
- Validates OWUI and worker before making any changes.

Examples:
  # Dry run (no ingest, no --confirm needed)
  python scripts/backfill_owui_knowledge_to_milvus.py --kb-ids=abc123 --dry-run

  # Real run (requires --confirm)
  python scripts/backfill_owui_knowledge_to_milvus.py --kb-ids=abc123 --confirm

  # Re-upload even if document already in collection (not recommended)
  python scripts/backfill_owui_knowledge_to_milvus.py --kb-ids=abc123 --confirm --no-skip-existing
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def to_milvus_safe(name: str) -> str:
    """Milvus allows only letters, digits, underscore; must start with letter or underscore."""
    if not (name or "").strip():
        return ""
    s = (name or "").strip()
    out = []
    for ch in s:
        if ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_":
            out.append(ch)
        elif ch in "-.:":
            out.append("_")
    result = "".join(out).strip("_") or ""
    if result and result[0].isdigit():
        result = "_" + result
    return result[:255] if result else ""


def kb_collection_name(kb_id: str, kb_meta: Dict[str, Any], prefix: str = "owui") -> str:
    """Same naming as pipeline: owui-kb-public-{id} or owui-kb-{id}."""
    is_public = kb_meta.get("is_public")
    visibility = kb_meta.get("visibility")
    publicish = str(is_public).lower() in ("true", "1") or str(visibility).lower() == "public"
    suffix = "kb-public" if publicish else "kb"
    return f"{prefix}-{suffix}-{kb_id}"


def owui_get(base_url: str, path: str, token: str, timeout: int = 60) -> Any:
    url = (base_url.rstrip("/") + path).replace("//", "/")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def worker_get(
    worker_url: str,
    path: str,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Any:
    """GET worker URL; returns parsed JSON or raises."""
    url = worker_url.rstrip("/") + path
    if params:
        from urllib.parse import urlencode
        url = url + "?" + urlencode(params)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def worker_list_documents(
    worker_url: str,
    collection_name: str,
    vdb_endpoint: str,
    timeout: int = 30,
) -> Set[str]:
    """Return set of document names already in the collection (for skip-existing). Empty on error."""
    try:
        out = worker_get(
            worker_url,
            "/v1/documents",
            params={"collection_name": collection_name, "vdb_endpoint": vdb_endpoint},
            timeout=timeout,
        )
    except urllib.error.HTTPError as e:
        if e.code == 501:
            return set()
        raise
    except Exception:
        return set()
    names: Set[str] = set()
    if isinstance(out, list):
        for x in out:
            if isinstance(x, str):
                names.add(x)
            elif isinstance(x, dict) and ("name" in x or "filename" in x):
                names.add(str(x.get("name") or x.get("filename") or ""))
    elif isinstance(out, dict):
        for key in ("documents", "result", "files", "names"):
            val = out.get(key)
            if isinstance(val, list):
                for x in val:
                    if isinstance(x, str):
                        names.add(x)
                break
    return {n for n in names if n}


def owui_get_file_content(base_url: str, file_id: str, token: str, timeout: int = 120) -> bytes:
    url = base_url.rstrip("/") + f"/api/v1/files/{file_id}/content"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def worker_ingest(
    worker_url: str,
    collection_name: str,
    vdb_endpoint: str,
    filename: str,
    file_bytes: bytes,
    chunk_size: int = 512,
    chunk_overlap: int = 150,
    timeout: int = 300,
) -> Dict[str, Any]:
    """POST file to worker /ingest (multipart). Collection name must be Milvus-safe."""
    import mimetypes
    boundary = "----BackfillBoundary"
    ct = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # Safe filename for header (no newlines)
    safe_name = filename.replace("\r", "").replace("\n", "").strip() or "file"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="collection_name"\r\n\r\n'
        f"{collection_name}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="vdb_endpoint"\r\n\r\n'
        f"{vdb_endpoint}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="blocking"\r\n\r\n'
        "true\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chunk_size"\r\n\r\n'
        f"{chunk_size}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chunk_overlap"\r\n\r\n'
        f"{chunk_overlap}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="generate_summary"\r\n\r\n'
        "false\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
        f"Content-Type: {ct}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + file_bytes + footer
    req = urllib.request.Request(
        worker_url.rstrip("/") + "/ingest",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def list_knowledge_bases(base_url: str, token: str) -> List[str]:
    """Try GET /api/v1/knowledge to list KB IDs; return empty if not supported."""
    try:
        out = owui_get(base_url, "/api/v1/knowledge", token)
        if isinstance(out, list):
            return [str(item.get("id") or item) for item in out if item]
        if isinstance(out, dict):
            data = out.get("data") or out.get("knowledges") or out.get("items") or []
            return [str(item.get("id") or item) for item in data if item]
        return []
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    except Exception:
        return []


def backfill_kb(
    kb_id: str,
    base_url: str,
    token: str,
    worker_url: str,
    vdb_endpoint: str,
    prefix: str,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    limit_files: Optional[int],
    skip_existing: bool,
) -> Tuple[int, int, int]:
    """Backfill one knowledge base. Returns (files_ok, files_skip, files_fail)."""
    try:
        kb_meta = owui_get(base_url, f"/api/v1/knowledge/{kb_id}", token)
    except urllib.error.HTTPError as e:
        print(f"  [skip] KB {kb_id}: HTTP {e.code}", file=sys.stderr)
        return 0, 0, 0
    except Exception as e:
        print(f"  [error] KB {kb_id}: {e}", file=sys.stderr)
        return 0, 0, 1
    collection_name = kb_collection_name(kb_id, kb_meta, prefix)
    milvus_name = to_milvus_safe(collection_name) or collection_name
    if not milvus_name:
        print(f"  [error] KB {kb_id}: invalid collection name derived", file=sys.stderr)
        return 0, 0, 1
    files = kb_meta.get("files") or []
    if limit_files is not None:
        files = files[: limit_files]
    if not files:
        print(f"  [ok] KB {kb_id} -> {milvus_name} (no files)", file=sys.stderr)
        return 0, 0, 0
    existing_docs: Set[str] = set()
    if not dry_run and skip_existing:
        existing_docs = worker_list_documents(worker_url, milvus_name, vdb_endpoint)
        if existing_docs:
            print(f"  (skipping {len(existing_docs)} already in collection)", file=sys.stderr)
    print(f"  KB {kb_id} -> {milvus_name} ({len(files)} file(s))", file=sys.stderr)
    ok, skip, fail = 0, 0, 0
    for f in files:
        fid = f.get("id") or f.get("file_id")
        if not fid:
            continue
        fname = f.get("filename") or f.get("name") or fid
        if not fname or not isinstance(fname, str):
            fname = str(fid)
        if dry_run:
            print(f"    [dry-run] would ingest: {fname}", file=sys.stderr)
            ok += 1
            continue
        if skip_existing and fname in existing_docs:
            print(f"    [skip] {fname} (already in collection)", file=sys.stderr)
            skip += 1
            continue
        try:
            content = owui_get_file_content(base_url, fid, token)
        except urllib.error.HTTPError as e:
            print(f"    [fail] {fname}: HTTP {e.code}", file=sys.stderr)
            fail += 1
            continue
        except Exception as e:
            print(f"    [fail] {fname}: fetch {e}", file=sys.stderr)
            fail += 1
            continue
        if not content:
            print(f"    [skip] {fname} (empty)", file=sys.stderr)
            skip += 1
            continue
        try:
            worker_ingest(worker_url, milvus_name, vdb_endpoint, fname, content, chunk_size, chunk_overlap)
            print(f"    [ok] {fname}", file=sys.stderr)
            ok += 1
        except urllib.error.HTTPError as e:
            print(f"    [fail] {fname}: worker HTTP {e.code} {e.reason}", file=sys.stderr)
            fail += 1
            continue
        except Exception as e:
            print(f"    [fail] {fname}: ingest {e}", file=sys.stderr)
            fail += 1
            continue
    return ok, skip, fail


def preflight_owui(base_url: str, token: str) -> None:
    """Validate OWUI reachable and token works. Raises SystemExit on failure."""
    try:
        owui_get(base_url, "/api/v1/knowledge", token)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise SystemExit("OPENWEBUI_API_KEY invalid or expired (OWUI returned 401).") from e
        if e.code == 404:
            pass
        else:
            raise SystemExit(f"Open WebUI returned HTTP {e.code} at {base_url}.") from e
    except OSError as e:
        raise SystemExit(f"Cannot reach Open WebUI at {base_url}: {e}.") from e


def preflight_worker(worker_url: str) -> None:
    """Validate worker reachable. Raises on failure."""
    try:
        worker_get(worker_url, "/health", timeout=10)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Worker returned HTTP {e.code} at {worker_url}.") from e
    except OSError as e:
        raise SystemExit(f"Cannot reach worker at {worker_url}: {e}.") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill OWUI knowledge bases into Milvus. Safe to run multiple times (skips existing docs by default)."
    )
    parser.add_argument("--kb-ids", type=str, default="", help="Comma-separated knowledge base IDs (required if listing not supported)")
    parser.add_argument("--dry-run", action="store_true", help="Do not call worker or upload; only validate and list what would be done")
    parser.add_argument("--confirm", action="store_true", help="Required to perform real ingest when not using --dry-run (avoids accidental runs)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Upload even if document already in collection (may duplicate chunks)")
    parser.add_argument("--limit-files", type=int, default=None, help="Max files per KB (for testing)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size for ingest")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for ingest")
    parser.add_argument("--no-validate", action="store_true", help="Skip preflight checks (not recommended)")
    args = parser.parse_args()

    base_url = env("OPENWEBUI_BASE_URL")
    token = env("OPENWEBUI_API_KEY")
    worker_url = env("NVIDIA_WORKER_URL")
    vdb_endpoint = env("VDB_ENDPOINT", "http://milvus:19530")
    prefix = env("COLLECTION_PREFIX", "owui")

    if not base_url or not token:
        print("Error: Set OPENWEBUI_BASE_URL and OPENWEBUI_API_KEY.", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm:
        print("Error: Use --confirm to perform real ingest, or --dry-run to only list what would be done.", file=sys.stderr)
        return 1
    if not worker_url and not args.dry_run:
        print("Error: Set NVIDIA_WORKER_URL for real ingest (or use --dry-run).", file=sys.stderr)
        return 1
    if not worker_url:
        worker_url = "http://localhost:8123"

    skip_existing = not args.no_skip_existing

    if not args.no_validate:
        print("Validating Open WebUI...", file=sys.stderr)
        try:
            preflight_owui(base_url, token)
        except SystemExit:
            raise
        print("  OK", file=sys.stderr)
        if not args.dry_run:
            print("Validating worker...", file=sys.stderr)
            try:
                preflight_worker(worker_url)
            except SystemExit:
                raise
            print("  OK", file=sys.stderr)

    kb_ids: List[str] = [x.strip() for x in args.kb_ids.split(",") if x.strip()]
    if not kb_ids:
        kb_ids = list_knowledge_bases(base_url, token)
    if not kb_ids:
        print("Error: No KB IDs. Use --kb-ids=id1,id2 or ensure GET /api/v1/knowledge returns a list.", file=sys.stderr)
        return 1

    print(
        f"Backfilling {len(kb_ids)} knowledge base(s) | dry_run={args.dry_run} | skip_existing={skip_existing}",
        file=sys.stderr,
    )
    total_ok, total_skip, total_fail = 0, 0, 0
    for kb_id in kb_ids:
        ok, skip, fail = backfill_kb(
            kb_id, base_url, token, worker_url, vdb_endpoint, prefix,
            args.chunk_size, args.chunk_overlap, args.dry_run, args.limit_files,
            skip_existing=skip_existing,
        )
        total_ok += ok
        total_skip += skip
        total_fail += fail
    print(f"Done: {total_ok} ingested, {total_skip} skipped (existing/empty), {total_fail} failed", file=sys.stderr)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
