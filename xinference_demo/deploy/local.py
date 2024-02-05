import asyncio          # _start_local_cluster
import logging          # _start_local_cluster
import sys              # run
import signal           # run
import multiprocessing  # run_in_subprocess
import time
from typing import Dict, Optional

import xoscar as xo
from xoscar.utils import get_next_port

from ..core.supervisor import SupervisorActor
from .worker import start_worker_components


async def _start_local_cluster(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
):
    from .utils import create_worker_actor_pool

    logging.config.dictConfig(logging_conf)  # type: ignore

    pool = None
    try:
        pool = await create_worker_actor_pool(
            address=address, logging_conf=logging_conf
        )
        await xo.create_actor(
            SupervisorActor, address=address, uid=SupervisorActor.uid()
        )
        await start_worker_components(
            address=address,
            supervisor_address=address,
            main_pool=pool,
            metrics_exporter_host=metrics_exporter_host,
            metrics_exporter_port=metrics_exporter_port,
        )
        await pool.join()
    except asyncio.CancelledError:
        if pool is not None:
            await pool.stop()

def run(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
):
    def sigterm_handler(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    loop = asyncio.get_event_loop()
    task = loop.create_task(
        _start_local_cluster(
            address=address,
            metrics_exporter_host=metrics_exporter_host,
            metrics_exporter_port=metrics_exporter_port,
            logging_conf=logging_conf,
        )
    )
    loop.run_until_complete(task)

def run_in_subprocess(
    address: str,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
) -> multiprocessing.Process:
    p = multiprocessing.Process(
        target=run,
        args=(address, metrics_exporter_host, metrics_exporter_port, logging_conf),
    )
    p.start()
    return p

def main(
    host: str,
    port: int,
    metrics_exporter_host: Optional[str] = None,
    metrics_exporter_port: Optional[int] = None,
    logging_conf: Optional[Dict] = None,
    auth_config_file: Optional[str] = None,
):
    '''
        开启 Worker 进程并启动 FastAPI Server
    '''
    supervisor_address = f"{host}:{get_next_port()}"
    print(f"=== supervisor_address: {supervisor_address}")
    local_cluster = run_in_subprocess(
        supervisor_address, metrics_exporter_host, metrics_exporter_port, logging_conf
    )

    try:
        from ..api import restful_api
        
        print(f"=== host:port: {host}:{port}")
        restful_api.run(
            supervisor_address=supervisor_address,
            host=host,
            port=port,
            logging_conf=logging_conf,
            auth_config_file=auth_config_file,
        )
    finally:
        local_cluster.terminate()