import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from models.graph_layer import GraphLayer
from models.mstcn import TCN1d
from models.topo_pooling import TopologyLayer


class MSConv_Real(nn.Module):
    """Temporal feature extractor backed by the real MSTCN TCN1d module."""

    def __init__(self, input_dim, feature_dim):
        super(MSConv_Real, self).__init__()
        self.proj = nn.Conv1d(input_dim, feature_dim, kernel_size=1)
        self.tcn = TCN1d(feature_num=feature_dim)

    def forward(self, x):
        # x: (B, N, T)
        B, N, T = x.shape
        x_flat = x.view(B * N, 1, T)

        # (B*N, 1, T) -> (B*N, F, T)
        out = self.proj(x_flat)
        out = self.tcn(out)

        # (B*N, F, T) -> (B*N, F) -> (B, N, F)
        out = out.mean(dim=-1)
        return out.view(B, N, -1)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """Generate batched PyG edge_index."""
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * node_num
    return batch_edge_index.long()


class GNNLayer_Topo(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, use_topo=True):
        super(GNNLayer_Topo, self).__init__()
        self.gnn = GraphLayer(
            in_channel,
            out_channel,
            inter_dim=inter_dim,
            heads=heads,
            concat=False,
        )
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.use_topo = use_topo

        if self.use_topo:
            self.topoPooling = TopologyLayer(
                out_channel,
                out_channel,
                num_filtrations=8,
                num_coord_funs={
                    "Triangle_transform": 3,
                    "Gaussian_transform": 3,
                    "Line_transform": 3,
                    "RationalHat_transform": 3,
                },
                filtration_hidden=32,
            )

    def forward(self, x, edge_index, embedding=None, node_num=0):
        if embedding is None:
            embedding = x.new_zeros((x.size(0), self.gnn.inter_dim))

        out, (new_edge_index, att_weight) = self.gnn(
            x,
            edge_index,
            embedding,
            return_attention_weights=True,
        )
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        if self.use_topo:
            data = Data(x=out, edge_index=new_edge_index)
            batch_data = Batch.from_data_list([data])
            try:
                topo_out, _, _ = self.topoPooling(batch_data.x, batch_data)
                out = out + topo_out
            except Exception as e:
                print(f"[Warning] Topology layer skipped: {e}")

        out = self.bn(out)
        return self.relu(out)


class TopoGCAD(nn.Module):
    def __init__(
        self,
        node_num,
        seq_len,
        input_dim=1,
        feature_dim=64,
        topk=20,
        use_topo=True,
    ):
        super(TopoGCAD, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.topk = topk
        self.use_topo = use_topo

        self.msconv = MSConv_Real(input_dim, feature_dim)

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gnn_layer = GNNLayer_Topo(
            feature_dim,
            feature_dim,
            inter_dim=128,
            heads=1,
            use_topo=use_topo,
        )
        self.dp = nn.Dropout(0.2)
        self.out_layer = nn.Linear(feature_dim, 1)

    def extract_causal_graph(self, x_temp):
        B, N, F = x_temp.shape
        assert N == self.node_num, f"节点数维度错误: 预期 {self.node_num}, 实际 {N}"
        adj_matrix = torch.zeros((N, N), device=x_temp.device)

        with torch.enable_grad():
            x_eval = x_temp.clone().detach().requires_grad_(True)
            preds = self.predictor(x_eval)

            for target_node in range(N):
                pred_sum = preds[:, target_node, :].sum()
                retain = target_node < N - 1
                grad = torch.autograd.grad(pred_sum, x_eval, retain_graph=retain)[0]

                causal_strength = grad.abs().mean(dim=(0, 2))
                adj_matrix[target_node] = causal_strength

        adj_matrix.fill_diagonal_(0)

        topk_indices = torch.topk(adj_matrix, self.topk, dim=1)[1]
        row = torch.arange(N, device=x_temp.device).unsqueeze(1).repeat(1, self.topk).flatten()
        col = topk_indices.flatten()
        edge_index = torch.stack([col, row], dim=0)

        return edge_index, adj_matrix

    def forward(self, x):
        B, N, T = x.shape
        assert T == self.seq_len, f"序列长度维度错误: 预期 {self.seq_len}, 实际 {T}"

        x_temp = self.msconv(x)

        edge_index, causal_adj = self.extract_causal_graph(x_temp)

        batch_edge_index = get_batch_edge_index(edge_index, B, N).to(x.device)

        x_flat = x_temp.view(B * N, -1)
        x_spatial = self.gnn_layer(x_flat, batch_edge_index, node_num=B * N)
        x_spatial = x_spatial.view(B, N, -1)

        out_features = x_temp + x_spatial
        out_features = self.dp(out_features)

        preds = self.out_layer(out_features).squeeze(-1)

        return preds, causal_adj
