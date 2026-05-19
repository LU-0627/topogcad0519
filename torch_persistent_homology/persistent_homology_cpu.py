import torch

# 导入我们将要编译出的 C++ 扩展模块
import torch_persistent_homology_cpu


def compute_persistent_homology(edge_index, edge_weight, num_nodes):
    """
    Python wrapper for the C++ persistent homology computation.
    """
    # 确保输入是 CPU 上的连续张量，这是 C++ 扩展的基本要求
    edge_index = edge_index.cpu().contiguous()
    edge_weight = edge_weight.cpu().contiguous()

    # 调用底层 C++ 函数
    return torch_persistent_homology_cpu.compute_persistent_homology(
        edge_index, edge_weight, num_nodes
    )
