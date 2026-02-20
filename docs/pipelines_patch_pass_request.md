# Patch: Pass request headers from Pipelines main.py into the pipe

The upstream [open-webui/pipelines](https://github.com/open-webui/pipelines) `main.py` does not pass the incoming HTTP request (or its headers) to the pipe. So pipes never see `Authorization` or other headers. This document gives a minimal patch you can apply to your Pipelines deployment so that the pipe receives a request-like object with headers (e.g. for Bearer token and user forwarding).

**Where to apply:** In your Pipelines server codebase, edit `main.py` (or the file that defines `generate_openai_chat_completion`).

---

## 1. Add `Request` to the chat completions endpoint

**Find:**
```python
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def generate_openai_chat_completion(form_data: OpenAIChatCompletionForm):
```

**Replace with:**
```python
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def generate_openai_chat_completion(request: Request, form_data: OpenAIChatCompletionForm):
```

(Ensure `Request` is imported from `fastapi` — it already is in the stock main.py.)

---

## 2. Capture headers and pass a request-like object into `job`

The `job()` function is run in a thread via `run_in_threadpool(job)`. Do not pass the Starlette `Request` object into another thread; capture the headers in the async context and pass a small object that exposes `.headers`.

**Find the block that defines `job()` and calls `run_in_threadpool(job)`.** It looks like:

```python
def job():
    print(form_data.model)
    pipeline = app.state.PIPELINES[form_data.model]
    ...
    if form_data.stream:
        def stream_content():
            res = pipe(
                user_message=user_message,
                model_id=pipeline_id,
                messages=messages,
                body=form_data.model_dump(),
            )
            ...
        return StreamingResponse(stream_content(), ...)
    else:
        res = pipe(
            user_message=user_message,
            model_id=pipeline_id,
            messages=messages,
            body=form_data.model_dump(),
        )
        ...

return await run_in_threadpool(job)
```

**Do the following:**

1. **Before** `def job():`, in the async function, add:
   ```python
   # Capture headers in async context (Request must not be used in another thread)
   headers_dict = dict(request.headers) if request else {}
   request_like = type("RequestLike", (), {"headers": headers_dict})()
   ```

2. **Change** `def job():` to `def job(req):` (or e.g. `def job(__request__=None):` and pass `request_like` as that argument).

3. **In both** `pipe(...)` call sites (inside `stream_content()` and in the `else` branch), add `__request__=req` (or `__request__=__request__` if you used that name):
   ```python
   res = pipe(
       user_message=user_message,
       model_id=pipeline_id,
       messages=messages,
       body=form_data.model_dump(),
       __request__=req,
   )
   ```

4. **Change** the final line from:
   ```python
   return await run_in_threadpool(job)
   ```
   to:
   ```python
   return await run_in_threadpool(job, request_like)
   ```

---

## 3. Streaming branch: pass `req` into `stream_content`

The streaming path has an inner function `stream_content()` that calls `pipe()`. The inner function must receive `req` (the request-like object) so it can pass it to `pipe()`. So:

- Either close over `request_like` in the outer scope (e.g. name it `req` and use `req` inside `stream_content()` when calling `pipe(..., __request__=req)`),  
- Or pass it as an argument, e.g. `def stream_content(req):` and when calling `pipe(..., __request__=req)`, and have `job(req)` call `StreamingResponse(stream_content(req), ...)`.

**Example (closure approach):** In `job(req)`, set a variable that `stream_content` can use:

```python
def job(req):
    print(form_data.model)
    pipeline = app.state.PIPELINES[form_data.model]
    ...
    if form_data.stream:
        def stream_content():
            res = pipe(
                user_message=user_message,
                model_id=pipeline_id,
                messages=messages,
                body=form_data.model_dump(),
                __request__=req,
            )
            ...
        return StreamingResponse(stream_content(), ...)
    else:
        res = pipe(
            user_message=user_message,
            model_id=pipeline_id,
            messages=messages,
            body=form_data.model_dump(),
            __request__=req,
        )
        ...
```

and keep `return await run_in_threadpool(job, request_like)`.

---

## 4. Verify

After applying the patch and restarting the Pipelines server:

- Send a request from Open WebUI (with Session or a Bearer token) to the pipeline.
- With `PIPE_DEBUG=true` in the pipelines container, you should see `request=True` and `token_source=request_headers` when the pipe receives the request-like object and an `Authorization` header.

---

## 5. Optional: pass `__user__` from body

If the Pipelines framework parses `form_data` from the body and Open WebUI sends a `user` field, you can also pass it into the pipe as `__user__` so the pipe does not rely only on headers. For example, if `form_data.model_dump()` contains a `"user"` key, you could pass `__user__=form_data.model_dump().get("user")` in the `pipe(...)` call. Our pipeline already prefers `body["user"]` when present; this would be redundant if the body is already passed as `body=form_data.model_dump()`.
