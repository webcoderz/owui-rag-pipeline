#!/usr/bin/env python3
"""Fetch open-webui/pipelines main.py and apply request-pass-through patch. Run in WSL."""
import urllib.request
import sys

URL = "https://raw.githubusercontent.com/open-webui/pipelines/main/main.py"
OUT = "main.py"

def main():
    try:
        with urllib.request.urlopen(URL, timeout=15) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        print(f"Fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 1) Add request: Request to chat completions endpoint
    old_sig = "async def generate_openai_chat_completion(form_data: OpenAIChatCompletionForm):"
    new_sig = "async def generate_openai_chat_completion(request: Request, form_data: OpenAIChatCompletionForm):"
    if old_sig not in raw:
        print("Signature not found (maybe already patched?)", file=sys.stderr)
        sys.exit(1)
    raw = raw.replace(old_sig, new_sig, 1)

    # 2) Before "def job():", add headers capture and request_like
    old_job_intro = """    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {form_data.model} not found",
        )

    def job():"""
    new_job_intro = """    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {form_data.model} not found",
        )

    # Capture headers in async context (Request must not be used in another thread)
    headers_dict = dict(request.headers) if request else {}
    request_like = type("RequestLike", (), {"headers": headers_dict})()

    def job(req):"""
    if old_job_intro not in raw:
        print("job() intro block not found", file=sys.stderr)
        sys.exit(1)
    raw = raw.replace(old_job_intro, new_job_intro, 1)

    # 3) In streaming pipe() call, add __request__=req (16 spaces before res, 20 before args)
    old_stream_pipe = """                res = pipe(
                    user_message=user_message,
                    model_id=pipeline_id,
                    messages=messages,
                    body=form_data.model_dump(),
                )"""
    new_stream_pipe = """                res = pipe(
                    user_message=user_message,
                    model_id=pipeline_id,
                    messages=messages,
                    body=form_data.model_dump(),
                    __request__=req,
                )"""
    if old_stream_pipe not in raw:
        print("Streaming pipe() block not found", file=sys.stderr)
        sys.exit(1)
    raw = raw.replace(old_stream_pipe, new_stream_pipe, 1)

    # 4) In non-streaming pipe() call, add __request__=req (12 spaces before res, 16 before args)
    old_nostream_pipe = """            res = pipe(
                user_message=user_message,
                model_id=pipeline_id,
                messages=messages,
                body=form_data.model_dump(),
            )
            logging.info(f"stream:false:{res}")"""
    new_nostream_pipe = """            res = pipe(
                user_message=user_message,
                model_id=pipeline_id,
                messages=messages,
                body=form_data.model_dump(),
                __request__=req,
            )
            logging.info(f"stream:false:{res}")"""
    if old_nostream_pipe not in raw:
        print("Non-streaming pipe() block not found", file=sys.stderr)
        sys.exit(1)
    raw = raw.replace(old_nostream_pipe, new_nostream_pipe, 1)

    # 5) run_in_threadpool(job) -> run_in_threadpool(job, request_like)
    raw = raw.replace("return await run_in_threadpool(job)", "return await run_in_threadpool(job, request_like)", 1)

    with open(OUT, "w", newline="\n", encoding="utf-8") as f:
        f.write(raw)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
