# Zero-footprint idle unload (`UNLOAD_IDLE`)

## What it does

When `UNLOAD_IDLE=1`, workers skip eager model loading at startup. The model
is loaded on the first incoming request. A background loop sends `SIGTERM` to
the worker after `IDLE_TIMEOUT` seconds (default: 15s) of inactivity. The
gunicorn master (always active) immediately forks a fresh cold worker to
replace it.

The result: idle workers cost almost nothing, and memory is freed without
relying on an external supervisor.

## Env vars

| Variable        | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `UNLOAD_IDLE`   | `0`     | Set to `1` to enable                            |
| `IDLE_TIMEOUT`  | `15`    | Seconds of inactivity before the worker exits   |
| `WORKERS`       | `1`     | Number of gunicorn workers (auto-used with idle) |

## Memory observations

Measured with `google-bert/bert-base-uncased` (fill-mask, CPU, ~440 MB on disk)
using `smaps_rollup` PSS (proportional set size, which correctly accounts for
pages shared between master and worker via fork COW).

### Before first request (cold worker)

```
Pss:               94 MB
Pss_Anon:          89 MB   # Python runtime + shared heap (COW with master)
Pss_File:           5 MB   # almost nothing loaded privately
Private_Clean:    252 kB   # negligible
Private_Dirty:     10 MB
```

The worker's true private footprint beyond the shared baseline is ~17 MB.

### After first request (model loaded)

```
Pss:              791 MB
Pss_Anon:         291 MB   # Python heap, tensor copies, runtime allocations
Pss_File:         499 MB   # model weights (memory-mapped from HF cache)
Private_Clean:    470 MB   # mmap'd weights — file-backed, reclaimable by kernel
Private_Dirty:    244 MB   # heap allocations — NOT reclaimable without swap
```

### Memory freed on SIGTERM (~708 MB)

| Component       | Size    | Reclaimable by kernel? |
|-----------------|---------|------------------------|
| Private_Clean   | ~470 MB | Yes (file-backed mmap) |
| Private_Dirty   | ~244 MB | No                     |

The kernel would eventually reclaim the `Private_Clean` pages under memory
pressure anyway (they are backed by the on-disk model files). The hard,
unconditional saving is the **~244 MB of `Private_Dirty`** that can only be
freed by terminating the process.

### Scale

`bert-base-uncased` is a small model (~110M parameters, 440 MB on disk).
The `Private_Dirty` component scales roughly with model size — a 7B model
typically yields several GB of non-reclaimable anonymous memory per worker.
With multiple replicas serving different models, idle unload can reclaim
tens of GB across a node.
