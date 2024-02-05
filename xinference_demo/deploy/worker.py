import asyncio
import logging
import os
from typing import Any, Optional

import xoscar as xo
from xoscar import MainActorPoolType

from ..core.worker import WorkerActor
from ..utils import cuda_count

logger = logging.getLogger(__name__)


async def start_worker_components(
    address: str,
    supervisor_address: str,
    main_pool: MainActorPoolType,
    metrics_exporter_host: Optional[str],
    metrics_exporter_port: Optional[int],
):
    cuda_device_indices = []
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        cuda_device_indices.extend([int(i) for i in cuda_visible_devices.split(",")])
    else:
        cuda_device_indices = list(range(cuda_count()))

    await xo.create_actor(
        WorkerActor,
        address=address,
        uid=WorkerActor.uid(),
        supervisor_address=supervisor_address,
        main_pool=main_pool,
        cuda_devices=cuda_device_indices,
        metrics_exporter_host=metrics_exporter_host,
        metrics_exporter_port=metrics_exporter_port,
    )