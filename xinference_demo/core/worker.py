import asyncio
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Optional, Dict, Set

import xoscar as xo
from xoscar import MainActorPoolType

from ..constants import XINFERENCE_CACHE_DIR
from .resource import gather_node_info
from .utils import purge_dir

logger = getLogger(__name__)

DEFAULT_NODE_HEARTBEAT_INTERVAL = 5 # 每 5 秒向 Supervisor 汇报一次状态

class WorkerActor(xo.StatelessActor):
    '''
    每个节点一个 WorkerActor 实例，负责处理 ModelActor 集合
    WorkerActor 实例中组合 MainActorPool , 由 MainActorPool 实际处理 SubPool(ModelActor) 的业务
    '''
    
    def __init__(
        self,
        supervisor_address: str,
        main_pool: MainActorPoolType,
        cuda_devices: List[int],
        metrics_exporter_host: Optional[str] = None,
        metrics_exporter_port: Optional[int] = None,
    ):
        super().__init__()
        # static attrs.
        self._total_cuda_devices = cuda_devices
        self._supervisor_address = supervisor_address
        self._supervisor_ref = None
        self._main_pool = main_pool
        # self._main_pool.recover_sub_pool = self.recover_sub_pool

        # internal states.
        self._model_uid_to_model: Dict[str, xo.ActorRefType["ModelActor"]] = {}
        # self._model_uid_to_model_spec: Dict[str, ModelDescription] = {}
        self._gpu_to_model_uid: Dict[int, str] = {}
        self._gpu_to_embedding_model_uids: Dict[int, Set[str]] = defaultdict(set)
        self._model_uid_to_addr: Dict[str, str] = {}
        self._model_uid_to_recover_count: Dict[str, int] = {}
        self._model_uid_to_launch_args: Dict[str, Dict] = {}

        # # metrics export server.
        # if metrics_exporter_host is not None or metrics_exporter_port is not None:
        #     logger.info(
        #         f"Starting metrics export server at {metrics_exporter_host}:{metrics_exporter_port}"
        #     )
        #     q: queue.Queue = queue.Queue()
        #     self._metrics_thread = threading.Thread(
        #         name="Metrics Export Server",
        #         target=launch_metrics_export_server,
        #         args=(q, metrics_exporter_host, metrics_exporter_port),
        #         daemon=True,
        #     )
        #     self._metrics_thread.start()
        #     logger.info("Checking metrics export server...")
        #     while self._metrics_thread.is_alive():
        #         try:
        #             host, port = q.get(block=False)[:2]
        #             logger.info(f"Metrics server is started at: http://{host}:{port}")
        #             break
        #         except queue.Empty:
        #             pass
        #     else:
        #         raise Exception("Metrics server thread exit.")

        self._lock = asyncio.Lock()

        logger.debug("Worker running")

    async def __post_create__(self):
        '''
        xo.StatelessActor 定义的后处理函数，在 WorkerActor 创建后调用
        绑定 SupervisorActor

        '''
        from .supervisor import SupervisorActor

        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = await xo.actor_ref(
            address=self._supervisor_address, uid=SupervisorActor.uid()
        )
        await self._supervisor_ref.add_worker(self.address)
        # xo.StatelessActor._upload_task 会被自动执行吗？
        self._upload_task = asyncio.create_task(self._periodical_report_status())
        logger.info(f"Xinference worker {self.address} started")
        logger.info("Purge cache directory: %s", XINFERENCE_CACHE_DIR)
        purge_dir(XINFERENCE_CACHE_DIR)


    async def __pre_destroy__(self):
        self._upload_task.cancel()

    @classmethod
    def uid(cls) -> str:
        return "worker"
    
    @staticmethod
    def get_devices_count():
        from ..utils import cuda_count

        return cuda_count()
    
    async def report_status(self):
        '''
        向 SupervisorAcotr 汇报节点 CPU 和内存的状态信息
        '''
        status = await asyncio.to_thread(gather_node_info)
        await self._supervisor_ref.report_worker_status(self.address, status)

    async def _periodical_report_status(self):
        '''
        周期性调用 report_status 向 SupervisorAcotr 汇报
        '''
        while True:
            try:
                await self.report_status()
            except asyncio.CancelledError:  # pragma: no cover
                break
            except RuntimeError as ex:  # pragma: no cover
                if "cannot schedule new futures" not in str(ex):
                    # when atexit is triggered, the default pool might be shutdown
                    # and to_thread will fail
                    break
            except (
                Exception
            ) as ex:  # pragma: no cover  # noqa: E722  # nosec  # pylint: disable=bare-except
                logger.error(f"Failed to upload node info: {ex}")
            try:
                await asyncio.sleep(DEFAULT_NODE_HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:  # pragma: no cover
                break