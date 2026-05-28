import asyncio
import os
import signal
import time
from contextlib import asynccontextmanager

from anyio import Semaphore

from hf_inference_sdk.logging import logger

UNLOAD_IDLE = os.getenv("UNLOAD_IDLE", "").lower() in ("1", "true")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "15"))

_MAX_CONCURRENT = 1000
_in_flight = Semaphore(_MAX_CONCURRENT)
_last_start: float | None = None
_last_end: float | None = None


async def live_check_loop() -> None:
    global _last_start, _last_end
    pid = os.getpid()
    sleep_time = max(IDLE_TIMEOUT // 5, 1)
    logger.info("Idle checker started (pid=%d, timeout=%ds)", pid, IDLE_TIMEOUT)

    while True:
        await asyncio.sleep(sleep_time)

        if _last_start is None:
            continue

        if _in_flight.value < _MAX_CONCURRENT:
            logger.debug("Idle checker: %d request(s) in flight, not idle", _MAX_CONCURRENT - _in_flight.value)
            continue

        if _last_end is None or _last_start >= _last_end:
            continue

        age = time.monotonic() - _last_end
        if age >= IDLE_TIMEOUT:
            logger.info("Idle checker: no request for %.1fs, sending SIGTERM to pid=%d", age, pid)
            os.kill(pid, signal.SIGTERM)
            return


@asynccontextmanager
async def request_tracker():
    global _last_start, _last_end
    async with _in_flight:
        _last_start = time.monotonic()
        try:
            yield
        finally:
            _last_end = time.monotonic()
