# SPDX-License-Identifier: Apache-2.0
"""Fixed-input router intervention checks on an actual captured CUDA graph."""

import pytest
import torch
from vllm.model_hooks.steering.algorithms.moe_router import MoERouterAlgorithm
from vllm.model_hooks.steering.controllers import RouterLogitsController
from vllm.model_hooks.steering.graph.kernels import apply_gate_intervention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph kernel test"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_router_graph_replay_reads_modes_experts_strength_and_topk(dtype):
    controller = RouterLogitsController()
    controller.hook_target = torch.nn.Linear(4, 13, bias=False)
    rows = torch.tensor([1, 2, 3, 4, 0, 1, 4], device="cuda")
    mask = torch.tensor([1, 1, 1, 1, 1, 0, 1], device="cuda", dtype=dtype)
    controller.init_graph_buffers(
        {"moe_gate"}, {"gate": mask}, capacity=4, hidden_size=4, max_rank=1,
        dtype=dtype, device=torch.device("cuda"), token_rows=rows,
    )
    payloads = {
        1: {"mode": "activate", "expert_ids": [1, 3], "deactivate_ids": [3]},
        2: {"mode": "deactivate", "expert_ids": [5], "activate_ids": [1]},
        3: {"mode": "soft", "expert_ids": [1, 11], "lambda": -0.75},
        4: {"mode": "soft_topk", "expert_ids": [1, 11], "lambda": 0.75, "topk": 3},
    }
    for row, payload in payloads.items():
        controller.set_graph_row(row, "moe_router", payload, 1.0)
    # Unique, exactly representable expert values avoid unspecified eager
    # torch.topk boundary tie ordering. E=13 also exercises padded lanes.
    logits = torch.arange(-6, 7, device="cuda", dtype=dtype).expand(7, -1).clone()
    logits += torch.arange(7, device="cuda", dtype=dtype)[:, None] * 0.125

    def forward():
        output = logits.clone()
        apply_gate_intervention(controller.graph_tables, mask, rows, output)
        return output

    compiled = torch.compile(forward, fullgraph=True)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled()

    def assert_current_payloads():
        graph.replay()
        expected = logits.cpu()
        algo = MoERouterAlgorithm()
        for token, (row, selected) in enumerate(
            zip(rows.cpu().tolist(), mask.cpu().tolist())
        ):
            if selected and row in payloads:
                expected[token] = algo._transform(
                    expected[token : token + 1], payloads[row]
                )[0]
        tolerance = max(1e-5, 2 * torch.finfo(dtype).eps)
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=tolerance, atol=tolerance
        )
        inactive = (mask == 0) | (rows == 0)
        assert torch.equal(actual[inactive], logits[inactive])
        assert not torch.equal(actual, logits)

    assert_current_payloads()
    payloads[1] = {
        "mode": "soft_topk",
        "expert_ids": [1, 11],
        "lambda": -0.5,
        "topk": 1,
    }
    payloads[4] = {"mode": "soft", "expert_ids": [2, 7], "lambda": 1.25}
    for row in (1, 4):
        controller.set_graph_row(row, "moe_router", payloads[row], 1.0)
    rows.copy_(torch.tensor([4, 1, 0, 2, 3, 1, 4], device="cuda"))
    mask.copy_(torch.tensor([1, 1, 1, 0, 1, 1, 0], device="cuda", dtype=dtype))
    assert_current_payloads()
    for row in payloads:
        controller.clear_graph_row(row)
    graph.replay()
    assert torch.equal(actual, logits)


def test_router_graph_topk_boundary_ties_use_expert_id_order():
    """A variable-k graph uses a defined tie rule without changing eager topk."""
    controller = RouterLogitsController()
    controller.hook_target = torch.nn.Linear(4, 5, bias=False)
    rows = torch.ones(1, device="cuda", dtype=torch.long)
    mask = torch.ones(1, device="cuda")
    controller.init_graph_buffers(
        {"moe_gate"}, {"gate": mask}, capacity=1, hidden_size=4, max_rank=1,
        dtype=torch.float32, device=torch.device("cuda"), token_rows=rows,
    )
    controller.set_graph_row(
        1,
        "moe_router",
        {
            "mode": "soft_topk",
            "expert_ids": [0, 1, 2, 3, 4],
            "lambda": 1.0,
            "topk": 2,
        },
        1.0,
    )
    logits = torch.tensor([[3.0, 2.0, 2.0, 2.0, 0.0]], device="cuda")
    graph = torch.cuda.CUDAGraph()
    apply_gate_intervention(controller.graph_tables, mask, rows, logits.clone())
    with torch.cuda.graph(graph):
        actual = logits.clone()
        apply_gate_intervention(controller.graph_tables, mask, rows, actual)
    graph.replay()
    expected = logits.clone()
    expected[:, 2:] += logits.std(dim=-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
