# Pipelines overlay: main.py that passes request to the pipe

This folder contains a **patched** `main.py` from [open-webui/pipelines](https://github.com/open-webui/pipelines). The only change is that the chat completions endpoint passes the incoming request headers into `pipe(..., __request__=req)` so pipes can read `Authorization` and other headers (e.g. for Bearer token forwarding from Open WebUI).

## Mount as a volume

Mount this file over the Pipelines container’s `main.py`. The image `ghcr.io/open-webui/pipelines:main` typically has the app at `/app` and the entrypoint runs `main.py` from there.

**Example (docker run):**
```bash
-v $(pwd)/pipelines_overlay/main.py:/app/main.py:ro
```

**In this repo’s docker-compose:** Uncomment or add the volume under the `pipelines` service (see below). Then:

```bash
docker compose up -d pipelines
```

If the container fails to start (e.g. `ModuleNotFoundError` or wrong path), inspect the image to find where `main.py` lives (e.g. `docker run --rm ghcr.io/open-webui/pipelines:main find / -name main.py 2>/dev/null`) and adjust the mount target path.

## What was changed

- `generate_openai_chat_completion(request: Request, form_data: ...)` — added `request: Request`.
- Before `run_in_threadpool(job)`, capture `request.headers` into a small object and pass it into `job(req)`.
- Both `pipe(...)` call sites (streaming and non-streaming) now pass `__request__=req`.

See `docs/RESEARCH_OWUI_USER_HEADERS.md` §11 and `docs/pipelines_patch_pass_request.md` for details.

## Regenerating main.py

To refresh from upstream and re-apply the patch (e.g. after an upstream change), run from this directory:

```bash
python3 fetch_and_patch_main.py
```

The script fetches `https://raw.githubusercontent.com/open-webui/pipelines/main/main.py` and applies the request-pass-through patch. Optionally verify syntax with `python3 -c "import ast; ast.parse(open('main.py').read())"`.
