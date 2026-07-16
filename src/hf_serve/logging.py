import logging
import os
import sys

_log_level_name = os.getenv("HF_SERVE_LOGGING_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logging.getLogger("uvicorn").handlers.clear()
logging.getLogger("uvicorn.access").handlers.clear()
logging.getLogger("uvicorn.error").handlers.clear()

logger = logging.getLogger("hf-serve")
logger.setLevel(_log_level)
