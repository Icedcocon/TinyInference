import asyncio
import time
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Any, List

import xoscar as xo

from .resource import ResourceStatus
from .utils import (
    log_async,
    log_sync
)

# 当开启类型检查时，导入以下模块
if TYPE_CHECKING:
    # from ..model.embedding import EmbeddingModelSpec
    # from ..model.image import ImageModelFamilyV1
    # from ..model.llm import LLMFamilyV1
    # from ..model.multimodal import LVLMFamilyV1
    # from ..model.rerank import RerankModelSpec
    from .worker import WorkerActor

logger = getLogger(__name__)

@dataclass
class WorkerStatus:
    '''
        Worker 状态信息：
            Worker 启动时间
            Worker 所在节点资源信息
    '''
    update_time: float
    status: Dict[str, ResourceStatus]

class SupervisorActor(xo.StatelessActor):
    '''
    一个集群只有一个 SupervisorActor 实例, 用于管理集群中各个节点的 WorkerActor
    SupervisorActor 聚合 WorkerActor, 维护 WorkerActor 的状态信息
    '''

    def __init__(self):
        super().__init__()
        self._worker_address_to_worker: Dict[str, xo.ActorRefType["WorkerActor"]] = {}
        self._worker_status: Dict[str, WorkerStatus] = {}
        self._replica_model_uid_to_worker: Dict[
            str, xo.ActorRefType["WorkerActor"]
        ] = {}
        # self._model_uid_to_replica_info: Dict[str, ReplicaInfo] = {}
        self._uptime = None
        self._lock = asyncio.Lock()

    @classmethod
    def uid(cls) -> str:
        return "supervisor"
    
    async def __post_create__(self):
        self._uptime = time.time()

    @staticmethod
    async def get_builtin_prompts() -> Dict[str, Any]:
        from ..model.llm.llm_family import BUILTIN_LLM_PROMPT_STYLE

        data = {}
        for k, v in BUILTIN_LLM_PROMPT_STYLE.items():
            # data[k] = v.dict()
            data[k] = v.model_dump()
        return data

    @staticmethod
    async def get_builtin_families() -> Dict[str, List[str]]:
        from ..model.llm.llm_family import (
            BUILTIN_LLM_MODEL_CHAT_FAMILIES,
            BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
            BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES,
        )

        return {
            "chat": list(BUILTIN_LLM_MODEL_CHAT_FAMILIES),
            "generate": list(BUILTIN_LLM_MODEL_GENERATE_FAMILIES),
            "tool_call": list(BUILTIN_LLM_MODEL_TOOL_CALL_FAMILIES),
            "hello": list(["hello","world"])
        }
    
    async def get_devices_count(self) -> int:
        from ..utils import cuda_count

        if self.is_local_deployment():
            return cuda_count()
        # distributed deployment, choose a worker and return its cuda_count.
        # Assume that each worker has the same count of cards.
        worker_ref = await self._choose_worker()
        return await worker_ref.get_devices_count()

    async def _choose_worker(self) -> xo.ActorRefType["WorkerActor"]:
        '''
            被 self.get_devices_count 调用

        '''
        # TODO: better allocation strategy.
        min_running_model_count = None
        target_worker = None

        workers = list(self._worker_address_to_worker.values())
        for worker in workers:
            running_model_count = await worker.get_model_count()
            if (
                min_running_model_count is None
                or running_model_count < min_running_model_count
            ):
                min_running_model_count = running_model_count
                target_worker = worker

        if target_worker:
            return target_worker

        raise RuntimeError("No available worker found")

    @log_sync(logger=logger)
    def get_status(self) -> Dict:
        '''
            被 restful_api 调用
            返回 Worker 状态
        '''
        return {
            "uptime": int(time.time() - self._uptime),
            "workers": self._worker_status,
        }
    
    def is_local_deployment(self) -> bool:
        # TODO: temporary.
        return (
            len(self._worker_address_to_worker) == 1
            and list(self._worker_address_to_worker)[0] == self.address
        )
    
    @log_async(logger=logger)
    async def add_worker(self, worker_address: str):
        '''
        WorkerActor 调用所属 Supervisor.add_worker 将自身注册
        '''
        from .worker import WorkerActor

        assert (
            worker_address not in self._worker_address_to_worker
        ), f"Worker {worker_address} exists"

        # 通过 xo.actor_ref 函数和 Worker传入的 worker_address 获取 WorkerActor 实例引用
        worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.uid())
        self._worker_address_to_worker[worker_address] = worker_ref
        logger.debug("Worker %s has been added successfully", worker_address)

    @log_async(logger=logger)
    async def remove_worker(self, worker_address: str):
        '''
        WorkerActor 调用所属 Supervisor.remove_worker 将自身注销
        '''
        if worker_address in self._worker_address_to_worker:
            del self._worker_address_to_worker[worker_address]
            logger.debug("Worker %s has been removed successfully", worker_address)
        else:
            logger.warning(
                f"Worker {worker_address} cannot be removed since it is not registered to supervisor."
            )

    async def report_worker_status(
        self, worker_address: str, status: Dict[str, ResourceStatus]
    ):
        if worker_address not in self._worker_status:
            logger.debug("Worker %s resources: %s", worker_address, status)
        self._worker_status[worker_address] = WorkerStatus(
            update_time=time.time(), status=status
        )