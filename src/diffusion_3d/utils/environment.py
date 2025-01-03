import os
import socket

from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.plugins.environments import LightningEnvironment


def set_multi_node_environment(nodes, port=10051):
    # nodes should be of the form [(node_name, (node_IP, num_devices)), ...]

    # Set some global variables
    MASTER_PORT = port
    MASTER_ADDR = nodes[0][1][0]
    WORLD_SIZE = sum([node_info[1] for _, node_info in nodes])

    # Set config num_nodes
    num_nodes = len(nodes)

    # Set config devices, NODE_RANK, and global rank starting point
    NODE_RANK = ...
    GLOBAL_RANK_OFFSET = 0
    devices = ...
    for i, (node, node_info) in enumerate(nodes):
        if node == socket.gethostname():
            devices = node_info[1]
            NODE_RANK = i
            break
        GLOBAL_RANK_OFFSET += node_info[1]

    # Set environment variables
    os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["NODE_RANK"] = str(NODE_RANK)

    class MyClusterEnvironment(LightningEnvironment):
        def set_world_size(self, size: int):
            # Here, size = num_nodes * len(devices)  which does not work for heterogenous clusters
            self._world_size = WORLD_SIZE

        def set_global_rank(self, rank: int):
            # Here, global_rank = node_rank * len(devices) + local_rank  which does not work for heterogenous clusters
            global_rank = GLOBAL_RANK_OFFSET + self.local_rank()
            self._global_rank = global_rank
            rank_zero_only.rank = global_rank

    return {
        "num_nodes": num_nodes,
        "devices": devices,
        "cluster_environment": MyClusterEnvironment(),
    }
