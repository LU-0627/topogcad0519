from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TopoGCAD tensor flow.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--node-num", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--use-topo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from models.TopoGCAD import TopoGCAD
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise SystemExit(
            f"Missing dependency: {missing}. Install project requirements before "
            "running the TopoGCAD flow test."
        ) from exc

    torch.manual_seed(42)

    batch_size = args.batch_size
    node_num = args.node_num
    seq_len = args.seq_len
    feature_dim = args.feature_dim
    topk = args.topk

    assert topk <= node_num, "topk must be <= node_num for this flow test."

    model = TopoGCAD(
        node_num=node_num,
        seq_len=seq_len,
        input_dim=1,
        feature_dim=feature_dim,
        topk=topk,
        use_topo=args.use_topo,
    )
    model.eval()

    shape_log: dict[str, tuple[int, ...]] = {}

    def capture_msconv(_module, _inputs, output):
        shape_log["msconv_out"] = tuple(output.shape)

    def capture_gnn(_module, inputs, output):
        shape_log["gnn_in_x"] = tuple(inputs[0].shape)
        shape_log["gnn_in_edge_index"] = tuple(inputs[1].shape)
        shape_log["gnn_out"] = tuple(output.shape)

    handles = [
        model.msconv.register_forward_hook(capture_msconv),
        model.gnn_layer.register_forward_hook(capture_gnn),
    ]

    x = torch.randn(batch_size, node_num, seq_len)
    preds, causal_adj = model(x)

    for handle in handles:
        handle.remove()

    expected_edges = batch_size * node_num * topk

    assert shape_log["msconv_out"] == (batch_size, node_num, feature_dim)
    assert shape_log["gnn_in_x"] == (batch_size * node_num, feature_dim)
    assert shape_log["gnn_in_edge_index"] == (2, expected_edges)
    assert shape_log["gnn_out"] == (batch_size * node_num, feature_dim)
    assert tuple(preds.shape) == (batch_size, node_num)
    assert tuple(causal_adj.shape) == (node_num, node_num)
    assert torch.isfinite(preds).all()
    assert torch.isfinite(causal_adj).all()

    print("TopoGCAD flow test passed.")
    print(f"input:              {tuple(x.shape)}")
    print(f"msconv_out:         {shape_log['msconv_out']}")
    print(f"gnn_in_x:           {shape_log['gnn_in_x']}")
    print(f"gnn_in_edge_index:  {shape_log['gnn_in_edge_index']}")
    print(f"gnn_out:            {shape_log['gnn_out']}")
    print(f"preds:              {tuple(preds.shape)}")
    print(f"causal_adj:         {tuple(causal_adj.shape)}")


if __name__ == "__main__":
    main()
