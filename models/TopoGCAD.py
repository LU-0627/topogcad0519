import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from models.graph_layer import GraphLayer
from models.topo_pooling import TopologyLayer


class MSConv_Fallback(nn.Module):
    """
    备用的时序特征提取模块：
    将 (B, N, T) 的原始时序数据映射为 (B, N, F) 的节点特征。
    """
    def __init__(self, input_dim, feature_dim):
        super(MSConv_Fallback, self).__init__()
        # 假设每个传感器是一个单变量序列 (input_dim=1)
        self.conv = nn.Conv1d(input_dim, feature_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, N, T)
        B, N, T = x.shape
        # Conv1d 期望的输入是 (Batch, Channel, Length)
        # 我们把 B 和 N 压平，当作独立的样本处理
        x_flat = x.view(B * N, 1, T)

        # 卷积提取特征: (B*N, 1, T) -> (B*N, F, T)
        out = self.conv(x_flat)

        # 在时间维度上做全局平均池化，消除 T 维度: (B*N, F, T) -> (B*N, F)
        out = out.mean(dim=-1)

        # 恢复 (B, N, F) 维度
        return out.view(B, N, -1)

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """生成批次化的 PyG edge_index 格式"""
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num
    return batch_edge_index.long()

class GNNLayer_Topo(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, use_topo=True):
        super(GNNLayer_Topo, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.use_topo = use_topo
        
        if self.use_topo:
            self.topoPooling = TopologyLayer(out_channel, out_channel, num_filtrations=8, 
                                            num_coord_funs={"Triangle_transform": 3,
                                                            "Gaussian_transform": 3,
                                                            "Line_transform": 3,
                                                            "RationalHat_transform": 3}, 
                                            filtration_hidden=32)

    def forward(self, x, edge_index, embedding=None, node_num=0):
        if embedding is None:
            embedding = x.new_zeros((x.size(0), self.gnn.inter_dim))

        # 1. 空间 GAT 消息传递
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        # 2. 拓扑信息提取 (TDA)
        if self.use_topo:
            data = Data(x=out, edge_index=new_edge_index)
            batch_data = Batch.from_data_list([data])  
            try:
                topoOut, _, _ = self.topoPooling(batch_data.x, batch_data)
                out = out + topoOut 
            except Exception as e:
                # C++ 尚未编译时的安全回退
                print(f"[Warning] 拓扑层计算跳过 (可能由于 C++ 环境未编译): {e}")

        out = self.bn(out)
        return self.relu(out)


class TopoGCAD(nn.Module):
    def __init__(self, node_num, seq_len, input_dim=1, feature_dim=64, topk=20, use_topo=True):
        super(TopoGCAD, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.topk = topk
        self.use_topo = use_topo
        
        # 1. 时序卷积模块 (MSConv)
        # 将原始数据 (B, N, T) 映射到特征空间 (B, N, F)
        self.msconv = MSConv_Fallback(input_dim, feature_dim)
        
        # 2. GCAD 纯因果图提取器 (基于 Jacobian)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 预测未来 1 步
        )
        
        # 3. 带有拓扑的图神经网络层
        # 接收 GCAD 动态提取的图，进行消息传递
        self.gnn_layer = GNNLayer_Topo(feature_dim, feature_dim, inter_dim=128, heads=1, use_topo=use_topo)
        
        # 4. Dropout 与融合输出层
        self.dp = nn.Dropout(0.2)
        self.out_layer = nn.Linear(feature_dim, 1)

    def extract_causal_graph(self, x_temp):
        """
        核心：不使用 Loss 反向传播，而是基于预测值求纯 Jacobian 偏导获得因果图
        """
        B, N, F = x_temp.shape
        assert N == self.node_num, f"节点数维度错误: 预期 {self.node_num}, 实际 {N}"

        # 断开计算图以防 OOM，开启 requires_grad 用于求导
        x_eval = x_temp.clone().detach().requires_grad_(True)
        preds = self.predictor(x_eval) # (B, N, 1)
        
        adj_matrix = torch.zeros((N, N), device=x_temp.device)
        
        for target_node in range(N):
            pred_sum = preds[:, target_node, :].sum()
            grad = torch.autograd.grad(pred_sum, x_eval, retain_graph=True)[0] 
            
            # (B, N, F) -> 聚合为 (N,) 表示各节点对 target_node 的因果强度
            causal_strength = grad.abs().mean(dim=(0, 2)) 
            adj_matrix[target_node] = causal_strength
            
        adj_matrix.fill_diagonal_(0)
        
        # 稠密邻接矩阵 -> 稀疏 edge_index
        topk_indices = torch.topk(adj_matrix, self.topk, dim=1)[1] # (N, topk)
        row = torch.arange(N, device=x_temp.device).unsqueeze(1).repeat(1, self.topk).flatten()
        col = topk_indices.flatten()
        edge_index = torch.stack([col, row], dim=0) # (2, E)
        
        return edge_index, adj_matrix

    def forward(self, x):
        # x shape: (B, N, T)
        B, N, T = x.shape
        assert T == self.seq_len, f"序列长度维度错误: 预期 {self.seq_len}, 实际 {T}"
        
        # 1. 提取时序特征 (B, N, T) -> (B, N, F)
        # MSConv 期待的输入可能是 (B, N, T) 视具体实现而定，请通过注释和 assert 保持关注
        x_temp = self.msconv(x) 
        
        # 这里进行防御性维度校验，防止 MSConv 后 T 维度没被消除
        if len(x_temp.shape) == 4 or x_temp.shape[2] != self.feature_dim:
            x_temp = x_temp.view(B, N, -1) # 如果是 (B, N, F, T) 等情况强制摊平
            
        # 2. 动态提取纯 Jacobian 因果图
        edge_index, causal_adj = self.extract_causal_graph(x_temp)
        
        # 3. 维度适配，准备输入 PyG
        batch_edge_index = get_batch_edge_index(edge_index, B, N).to(x.device)
        
        # GNN 层要求合并 Batch 和 Node 维度：(B, N, F) -> (B*N, F)
        x_flat = x_temp.view(B * N, -1)
        
        # 4. 图与拓扑消息传递
        x_spatial = self.gnn_layer(x_flat, batch_edge_index, node_num=B*N)
        
        # 恢复 (B, N, F)
        x_spatial = x_spatial.view(B, N, -1)
        
        # 5. 残差连接与最终预测
        out_features = x_temp + x_spatial
        out_features = self.dp(out_features)
        
        preds = self.out_layer(out_features).squeeze(-1) # (B, N)
        
        return preds, causal_adj
