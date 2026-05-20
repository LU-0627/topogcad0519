import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from models.TopoGCAD import TopoGCAD
from datasets.TimeDataset import TimeDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Topo-GCAD for Multivariate Time Series Anomaly Detection')
    parser.add_argument('--data_path', type=str, default='./data/swat', help='数据路径')
    parser.add_argument('--node_num', type=int, default=51, help='传感器/节点数量')
    parser.add_argument('--seq_len', type=int, default=5, help='滑动窗口长度')
    parser.add_argument('--feature_dim', type=int, default=64, help='隐层特征维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--topk', type=int, default=10, help='因果图稀疏化 Top-K')
    parser.add_argument('--use_topo', action='store_true', help='是否启用 TDA 拓扑层')
    return parser.parse_args()


def train_model(model, train_loader, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model.train()

    print("🚀 开始训练 Topo-GCAD...")
    baseline_causal_graph = torch.zeros((args.node_num, args.node_num)).to(device)

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device) # x: (B, N, T), y: (B, N)

            optimizer.zero_grad()
            preds, causal_adj = model(x)

            # 主任务：预测未来的节点状态
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 收集正常状态下的平均因果图作为基线 (只在最后一个 Epoch 收集即可，为了简单这里使用滑动平均)
            if epoch == args.epochs - 1:
                baseline_causal_graph += causal_adj.detach().mean(dim=0) if len(causal_adj.shape)==3 else causal_adj.detach()

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    baseline_causal_graph = baseline_causal_graph / len(train_loader)
    return baseline_causal_graph


def test_model(model, test_loader, baseline_causal_graph, device):
    model.eval()
    all_scores = []
    all_labels = []

    # 提取正常状态的节点出度 (因果影响发出力)
    baseline_out_degree = baseline_causal_graph.sum(dim=1)

    print("🔍 开始异常检测评估 (Rank-based 双信号融合)...")
    for x, y, labels in test_loader:
        x, y = x.to(device), y.to(device)

        # 这里的模型带有 requires_grad=True 的图提取，必须在 enable_grad 下运行
        with torch.enable_grad():
            preds, causal_adj = model(x)

        # 1. GCAD 优化报告核心：SWaT 数据集的预测误差取反（攻击导致系统更平稳）
        # 预测误差越小，反而越异常
        pred_error = ((preds - y) ** 2).mean(dim=1) # (B,)
        negated_pred_error = -1.0 * pred_error

        # 2. 因果图拓扑偏离度 (Causal Score)
        # 当前 batch 提取的 Jacobian 矩阵与 baseline 的差异
        if len(causal_adj.shape) == 3: # 如果模型返回了 batched adjacency
            current_out_degree = causal_adj.sum(dim=2) # (B, N)
        else:
            current_out_degree = causal_adj.sum(dim=1).unsqueeze(0).repeat(x.shape[0], 1)

        causal_deviation = torch.abs(current_out_degree - baseline_out_degree).mean(dim=1) # (B,)

        # 3. 双信号融合 (简单相加或标准化后相加)
        # 实际使用中可以加 Alpha / Beta 权重
        anomaly_score = negated_pred_error + causal_deviation

        all_scores.append(anomaly_score.detach().cpu().numpy())
        all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # 计算 ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_scores)
    print(f"🎯 测试完成! 评估指标 ROC-AUC: {roc_auc:.4f}")
    return roc_auc


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📦 运行设备: {device}")

    # --- 这里替换为你的真实数据加载逻辑 ---
    # 临时生成假数据以保证脚本能跑通
    print(f"📂 正在加载真实数据集: {args.data_path}")
    # 假设你的数据放在 data/swat/train.csv 和 test.csv
    # 且前 node_num 列是传感器特征，最后一列是 Label (0正常, 1异常)
    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    
    # 提取特征和标签
    train_data_raw = train_df.iloc[:, :args.node_num].values
    test_data_raw = test_df.iloc[:, :args.node_num].values
    test_labels = test_df.iloc[:, -1].values
    
    # ⚠️ 极其重要：传感器数据必须做 Min-Max 归一化
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)

    train_dataset = TimeDataset(train_data, labels=None, seq_len=args.seq_len)
    test_dataset = TimeDataset(test_data, labels=test_labels, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TopoGCAD(
        node_num=args.node_num,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        topk=args.topk,
        use_topo=args.use_topo
    ).to(device)

    # 1. 训练模型并获取基线因果图
    baseline_graph = train_model(model, train_loader, device, args)

    # 2. 测试集评估
    test_model(model, test_loader, baseline_graph, device)
