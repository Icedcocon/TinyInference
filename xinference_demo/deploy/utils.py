import os
import time
import logging
from typing import Optional

import xoscar as xo

logger = logging.getLogger(__name__)

XINFERENCE_HOME = "/TRTDir/Custom/xinference_demo"
XINFERENCE_LOG_DIR = os.path.join(XINFERENCE_HOME, "logs")
XINFERENCE_DEFAULT_LOG_FILE_NAME = "xinference.log"

async def create_worker_actor_pool(
    address: str, logging_conf: Optional[dict] = None
) -> "MainActorPoolType":
    subprocess_start_method = "forkserver" if os.name != "nt" else "spawn"

    return await xo.create_actor_pool(
        address=address,
        n_process=0,
        auto_recover="process",
        subprocess_start_method=subprocess_start_method,
        logging_conf={"dict": logging_conf},
    )

def get_log_file(sub_dir: str):
    """
    sub_dir should contain a timestamp.
    """
    log_dir = os.path.join(XINFERENCE_LOG_DIR, sub_dir)
    # Here should be creating a new directory each time, so `exist_ok=False`
    os.makedirs(log_dir, exist_ok=False)
    return os.path.join(log_dir, XINFERENCE_DEFAULT_LOG_FILE_NAME)

class LoggerNameFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("xinference") or (
            record.name.startswith("uvicorn.error")
            and record.getMessage().startswith("Uvicorn running on")
        )

def get_config_dict(
    log_level: str, log_file_path: str, log_backup_count: int, log_max_bytes: int
) -> dict:
    # for windows, the path should be a raw string.
    log_file_path = (
        log_file_path.encode("unicode-escape").decode()
        if os.name == "nt"
        else log_file_path
    )
    log_level = log_level.upper()
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "formatter": {
                "format": (
                    "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
                )
            },
        },
        "filters": {
            "logger_name_filter": {
                "()": __name__ + ".LoggerNameFilter",
            },
        },
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
                "stream": "ext://sys.stderr",
                "filters": ["logger_name_filter"],
            },
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "formatter",
                "level": log_level,
                "filename": log_file_path,
                "mode": "a",
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "xinference": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            }
        },
        "root": {
            # "level": "WARN",
            "level": log_level,
            "handlers": ["stream_handler", "file_handler"],
        },
    }
    return config_dict


def get_timestamp_ms():
    t = time.time()
    return int(round(t * 1000))
