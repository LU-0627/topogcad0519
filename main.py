import torch
from models.TopoGCAD import TopoGCAD


def test_forward_pass():
    print("🚀 开始测试 Topo-GCAD 前向传播数据流...")

    # 1. 设定模拟参数
    BATCH_SIZE = 16
    NODE_NUM = 51   # 假设是 SWaT 数据集的 51 个传感器
    SEQ_LEN = 15    # 滑动窗口长度
    FEATURE_DIM = 64

    print(f"📌 设定维度 -> Batch: {BATCH_SIZE}, Nodes: {NODE_NUM}, Time Window: {SEQ_LEN}")

    # 2. 实例化模型 (use_topo=True, 本地测试 C++ 报错会安全回退)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TopoGCAD(
        node_num=NODE_NUM,
        seq_len=SEQ_LEN,
        input_dim=1,
        feature_dim=FEATURE_DIM,
        topk=10,
        use_topo=True
    ).to(device)

    model.eval() # 测试模式

    # 3. 构造模拟的多元时序输入数据 (B, N, T)
    dummy_input = torch.randn(BATCH_SIZE, NODE_NUM, SEQ_LEN).to(device)
    print(f"✅ 生成模拟输入张量，Shape: {dummy_input.shape}")

    # 4. 执行前向传播
    try:
        preds, causal_adj = model(dummy_input)

        print("-" * 40)
        print("🎯 前向传播成功完成！")
        print(f"📊 预测输出 Shape: {preds.shape} (预期为 {BATCH_SIZE}, {NODE_NUM})")
        print(f"🕸️ 提取的因果图 Shape: {causal_adj.shape} (预期为 {NODE_NUM}, {NODE_NUM})")

        assert preds.shape == (BATCH_SIZE, NODE_NUM), "预测值维度不匹配！"
        assert causal_adj.shape == (NODE_NUM, NODE_NUM), "因果图维度不匹配！"
        print("✨ 所有维度断言校验通过！底层 B-N-T 数据流转逻辑严丝合缝！")

    except Exception as e:
        print(f"❌ 前向传播失败，报错信息：\n{e}")


if __name__ == '__main__':
    test_forward_pass()
