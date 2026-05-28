# DISCARD_LEFT — Skip inference when caller already disconnected

## What it does

When `DISCARD_LEFT=1`, the server checks whether the HTTP client has already dropped the
connection before starting inference. If so, it returns 204 immediately instead of running
the model, saving CPU/GPU for requests that still have someone waiting.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DISCARD_LEFT` | `0` | Set to `1` to enable |

## Implementation

`idle.caller_left(request)` uses `psutil.net_connections(kind="tcp")` to check whether the
client's TCP connection is still in `ESTABLISHED` state. It is called at the top of every
predict handler, inside `async with idle.request_tracker():`, before invoking the predictor.

`request.is_disconnected()` (Starlette) was evaluated as an alternative but was found to
always return `False` — see test results below.

## Verification

Tested with `google-bert/bert-base-uncased` / `fill-mask`, using a `/test-disconnect` probe
endpoint that logs both detection methods before and after a 4-second sleep (giving curl time
to disconnect mid-flight).

Four scenarios were tested: direct (graceful FIN via `curl --max-time 1`), direct (abrupt RST
via raw socket with `SO_LINGER`), proxied via socat + FIN, proxied via socat + RST.
Tests 2–5 ran on plain uvicorn (no gunicorn, no `UNLOAD_IDLE`).

```
# FIN, gunicorn (UNLOAD_IDLE=1)
pre-sleep   psutil=False  is_disconnected=False
post-sleep  psutil=True   is_disconnected=False  ✓

# FIN, uvicorn
pre-sleep   psutil=False  is_disconnected=False
post-sleep  psutil=True   is_disconnected=False  ✓

# RST, uvicorn
pre-sleep   psutil=False  is_disconnected=False
post-sleep  psutil=True   is_disconnected=False  ✓

# socat + FIN, uvicorn
pre-sleep   psutil=False  is_disconnected=False
post-sleep  psutil=True   is_disconnected=False  ✓

# socat + RST, uvicorn
pre-sleep   psutil=False  is_disconnected=False
post-sleep  psutil=True   is_disconnected=False  ✓
```

## Conclusions

- `request.is_disconnected()` always returns `False` in this stack, both under gunicorn and
  plain uvicorn. It cannot be used as a replacement.
- `psutil` correctly detects disconnection in all cases, including through socat.
- The socat result shows that transparent proxies (nginx ingress, traefik, envoy) close the
  backend connection when the client disconnects, so psutil works correctly in a typical k8s
  setup. The only failure case would be a proxy with connection pooling/keepalives toward the
  backend.
