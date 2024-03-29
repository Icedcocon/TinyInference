from dataclasses import dataclass
from typing import Dict

import psutil

@dataclass
class ResourceStatus:
    '''
        Worker 所在节点资源信息
    '''
    available: float
    total: float
    memory_available: float
    memory_total: float

def gather_node_info() -> Dict[str, ResourceStatus]:
    '''
    计算节点 CPU 和内存资源信息并返回
    '''
    node_resource = dict()
    mem_info = psutil.virtual_memory()
    node_resource["cpu"] = ResourceStatus(
        available=psutil.cpu_percent() / 100.0,
        total=psutil.cpu_count(),
        memory_available=mem_info.available,
        memory_total=mem_info.total,
    )
    # TODO: record GPU stats
    # for idx, gpu_card_stat in enumerate(resource.cuda_card_stats()):
    #     node_resource[f"gpu-{idx}"] = ResourceStatus(
    #         available=gpu_card_stat.gpu_usage / 100.0,
    #         total=1,
    #         memory_available=gpu_card_stat.fb_mem_info.available,
    #         memory_total=gpu_card_stat.fb_mem_info.total,
    #     )

    return node_resource