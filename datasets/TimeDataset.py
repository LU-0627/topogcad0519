import torch
from torch.utils.data import Dataset
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, data, labels, seq_len=15, pred_len=1):
        """
        data: (Samples, Nodes) 的 numpy 数组或 tensor
        labels: (Samples,) 的异常标签 (0 为正常，1 为异常)
        seq_len: 滑动窗口长度 T
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # 截取滑动窗口: (T, N)
        x = self.data[index : index + self.seq_len]
        # 目标预测值: (N,)
        y = self.data[index + self.seq_len + self.pred_len - 1]

        # 转换为 TopoGCAD 需要的维度: (N, T)
        x = x.transpose(0, 1)

        if self.labels is not None:
            # 只要预测窗口内有异常，就标记为异常
            label = self.labels[index + self.seq_len : index + self.seq_len + self.pred_len].max()
            return x, y, label
        return x, y
