import asyncio
import ipaddress
import os
import signal
import time
from contextlib import asynccontextmanager

import psutil
from anyio import Semaphore
from starlette.requests import Request

from hf_inference_sdk.logging import logger

UNLOAD_IDLE = os.getenv("UNLOAD_IDLE", "").lower() in ("1", "true")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "15"))
DISCARD_LEFT = os.getenv("DISCARD_LEFT", "").lower() in ("1", "true")

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


def caller_left(request: Request) -> bool:
    """
    Check whether the caller has already closed the connection before inference starts.
    Uses psutil.net_connections() to inspect the TCP state of the client connection —
    request.is_disconnected() is broken in Starlette (consumes the payload and returns wrong status).
    Fails safe: returns False (keep going) on any error or ambiguity.
    """
    if not DISCARD_LEFT:
        return False
    try:
        client = request.client
        if not client or not client.host:
            return False
        host = ipaddress.ip_address(client.host)
        port = int(client.port)
        if port <= 0 or port > 65535:
            return False
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status != "ESTABLISHED":
                continue
            if not conn.raddr:
                continue
            if int(conn.raddr.port) != port:
                continue
            if not conn.raddr.ip or ipaddress.ip_address(conn.raddr.ip) != host:
                continue
            logger.debug("Caller connection still ESTABLISHED, keeping request")
            return False
    except Exception:
        logger.warning("caller_left: unexpected error checking TCP state, assuming caller is still there")
        return False
    logger.info("No ESTABLISHED connection found for caller, discarding request")
    return True


@asynccontextmanager
async def request_tracker():
    global _last_start, _last_end
    async with _in_flight:
        _last_start = time.monotonic()
        try:
            yield
        finally:
            _last_end = time.monotonic()
