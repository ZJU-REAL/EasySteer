# SPDX-License-Identifier: Apache-2.0
"""Tensor-level graph replay checks, independent of model generation."""

import pytest
import torch
from vllm.model_hooks.steering.graph.kernels import apply_decoder_families

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph kernel test"
)


def _reference(vectors, mask, normalize, rows, hidden, residual):
    x = hidden if residual is None else hidden + residual
    delta = mask[:, None] * vectors[rows]
    y = x + delta
    norm_x = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)
    norm_y = torch.linalg.vector_norm(y.float(), dim=-1, keepdim=True)
    renormed = (y.float() * norm_x / (norm_y + 1e-8)).to(y.dtype)
    flag = normalize[rows, None] * mask[:, None]
    return hidden + delta + flag * (renormed - y)


def _check_replay(dtype, has_residual, compiled=False):
    generator = torch.Generator(device="cuda").manual_seed(17)
    hidden_size = 1536 if compiled else 67
    hidden = torch.randn(
        7, hidden_size * 2, device="cuda", dtype=dtype, generator=generator
    )[:, ::2]
    residual = (
        torch.randn(hidden_size, 7, device="cuda", dtype=dtype, generator=generator).T
        if has_residual
        else None
    )
    vectors = torch.randn(
        4, hidden_size * 2, device="cuda", dtype=dtype, generator=generator
    )[:, ::2]
    vectors[0].zero_()
    mask = torch.tensor([1, 0, 1, 1, 1, 1, 1], device="cuda", dtype=dtype)
    rows = torch.tensor([1, 2, 0, 2, 3, 1, 3], device="cuda")
    normalize = torch.tensor([0, 0, 1, 1], device="cuda", dtype=dtype)

    def forward():
        return apply_decoder_families(
            {"additive": {"V": vectors}},
            mask,
            None,
            normalize,
            rows,
            hidden,
            residual,
        )

    run = torch.compile(forward, fullgraph=True) if compiled else forward
    run()  # Compile/JIT before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()

    def assert_matches():
        graph.replay()
        expected = _reference(vectors, mask, normalize, rows, hidden, residual)
        tolerance = max(1e-5, 2 * torch.finfo(dtype).eps)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        inactive = (mask == 0) | (rows == 0)
        assert torch.equal(actual[inactive], hidden[inactive])
        unnormalized = ~inactive & (normalize[rows] == 0)
        assert torch.equal(
            actual[unnormalized],
            (hidden + mask[:, None] * vectors[rows])[unnormalized],
        )

    assert_matches()
    # The same captured graph must read new routing, masks and normalize flags.
    rows.copy_(torch.tensor([3, 1, 2, 0, 2, 3, 1], device="cuda"))
    mask.copy_(torch.tensor([0, 1, 1, 1, 1, 0, 1], device="cuda", dtype=dtype))
    normalize.copy_(torch.tensor([0, 1, 0, 1], device="cuda", dtype=dtype))
    assert_matches()
    # Zero payloads are exact no-ops when normalization is disabled.
    vectors.zero_()
    normalize.zero_()
    graph.replay()
    assert torch.equal(actual, hidden)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_residual", [False, True])
def test_additive_graph_replay_uses_current_buffers(dtype, has_residual):
    _check_replay(dtype, has_residual)


def test_compiled_additive_graph_replay_uses_current_buffers():
    _check_replay(torch.bfloat16, True, compiled=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multifamily_graph_normalize_tracks_replayed_flags(dtype):
    """Normalization changes fixed hidden states even if generated tokens agree."""
    from vllm.model_hooks.steering.graph.kernels import GRAPH_FAMILIES

    hidden_size, rank, table_rows = 64, 4, 5
    half = hidden_size // 2
    dimensions = {"h": hidden_size, "r": rank}
    tables = {
        family: {
            name: torch.zeros(
                table_rows, *(dimensions[d] for d in shape), device="cuda", dtype=dtype
            )
            for name, shape in schema.items()
        }
        for family, schema in GRAPH_FAMILIES.items()
    }
    # Each table row uses one family, as with a single-vector request. The
    # declared family set keeps this on the general Torch graph path.
    tables["additive"]["V"][1, :half] = 1.0
    tables["projection"]["B"][2] = 1.0 / hidden_size
    tables["projection"]["C"][2, :half] = 1.0
    tables["lowrank"]["A"][3, :, 0] = 1.0 / hidden_size
    tables["lowrank"]["Rout"][3, :half, 0] = 1.0
    tables["lowrank"]["b"][3, 0] = 0.25
    tables["replace"]["V"][4, :half] = 3.0
    tables["replace"]["V"][4, half:] = 2.0

    hidden = torch.ones(6, hidden_size, device="cuda", dtype=dtype)
    residual = torch.full_like(hidden, 0.5)
    rows = torch.tensor([1, 2, 3, 4, 0, 1], device="cuda")
    mask = torch.tensor([1, 1, 1, 0, 0, 0], device="cuda", dtype=dtype)
    replace_mask = torch.tensor([0, 0, 0, 1, 0, 0], device="cuda", dtype=dtype)
    normalize = torch.zeros(table_rows, device="cuda", dtype=dtype)
    active = (mask + replace_mask) > 0

    # Independently calculated deltas for x=1.5: additive=1, projection=1.5,
    # lowrank=1.5+0.25, replacement=(3, 2)-1.5. The last two tokens are idle.
    delta = torch.zeros_like(hidden)
    delta[0, :half] = 1.0
    delta[1, :half] = 1.5
    delta[2, :half] = 1.75
    delta[3, :half] = 1.5
    delta[3, half:] = 0.5
    original_state = hidden + residual
    original_norm = original_state.float().norm(dim=-1)
    tolerance = max(2e-5, 2 * torch.finfo(dtype).eps)

    def forward():
        return apply_decoder_families(
            tables, mask, replace_mask, normalize, rows, hidden, residual
        )

    run = torch.compile(forward, fullgraph=True)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()

    def assert_matches_formula():
        graph.replay()
        transformed = original_state + delta
        renormalized = (
            transformed.float()
            * original_norm[:, None]
            / (transformed.float().norm(dim=-1, keepdim=True) + 1e-8)
        ).to(dtype)
        flags = normalize[rows, None] * active[:, None]
        expected = hidden + delta + flags * (renormalized - transformed)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        assert torch.equal(actual[~active], hidden[~active])

    assert_matches_formula()
    baseline = actual.clone()
    assert torch.equal(baseline, hidden + delta)

    normalize[1:] = 1
    assert_matches_formula()
    change = (actual.float() - baseline.float()).abs().amax(dim=-1)
    assert (change[active] > 0.1).all(), "normalization had no tensor-level effect"
    torch.testing.assert_close(
        (actual + residual).float().norm(dim=-1)[active],
        original_norm[active],
        atol=tolerance,
        rtol=tolerance,
    )

    normalize.zero_()
    normalize[2] = 1
    assert_matches_formula()
    unchanged = rows != 2
    assert torch.equal(actual[unchanged], baseline[unchanged])
    normalize.zero_()
    assert_matches_formula()
    assert torch.equal(actual, baseline)
