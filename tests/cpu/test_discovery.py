# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for shared module identity and decoder output adaptation.

Tiny model trees cover global layer numbering, incomplete hybrid stacks, and
capture/steering agreement without allocating weights or starting an engine.
"""

from abc import ABCMeta
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.model_hooks.capture.session import CaptureSession
from vllm.model_hooks.capture.store import StreamConfig, StreamStore
from vllm.model_hooks.components import outputs
from vllm.model_hooks.components.discovery import (
    ModelDiscovery,
    resolve_moe_gate,
)
from vllm.model_hooks.components.registry import (
    COMPONENTS,
    HIDDEN_STATES,
    ROUTER_LOGITS,
    discover_components,
    get_component,
)
from vllm.model_hooks.steering.capabilities import (
    ALGORITHM_CAPABILITIES,
    algorithm_target,
)
from vllm.model_hooks.steering.controllers.manager import ControllerManager


class TinyAttention(nn.Module, AttentionLayerBase):
    def get_attn_backend(self):
        return None

    def get_kv_cache_spec(self, vllm_config):
        return None


class TinyMamba(nn.Module, MambaBase):
    def get_state_shape(self):
        return ()

    def get_state_dtype(self):
        return ()

    @property
    def mamba_type(self):
        return None


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()

    def forward(self, x):
        return x + 1


class Stack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            if not isinstance(layer, PPMissingLayer):
                x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def forward(self, hidden_states, residual=None):
        return hidden_states, residual


class TinyRunner(nn.Module):
    def __init__(self, gate=None, fused=False):
        super().__init__()
        self.gate = gate
        self._fse_fuse_gate = fused


# PluggableLayer.register is a plugin decorator, not ABC virtual registration.
ABCMeta.register(MoERunnerInterface, TinyRunner)


def test_nested_numeric_prefix_uses_decoder_stack_index():
    model = nn.Sequential(Stack([Decoder(), Decoder()]))
    layers = ModelDiscovery(model).decoder_layers
    assert [(layer.name, layer.layer_id) for layer in layers] == [
        ("0.layers.0", 0),
        ("0.layers.1", 1),
    ]
    assert layers[1].module is model[0].layers[1]
    assert issubclass(MambaBase, AttentionLayerBase)


def test_mamba_uses_the_shared_attention_interface_without_class_names():
    layer = nn.Module()
    layer.mixer = TinyMamba()
    found = ModelDiscovery(Stack([layer])).decoder_layers
    assert len(found) == 1
    assert found[0].module is layer and found[0].layer_id == 0


@pytest.mark.parametrize("compact", [False, True])
def test_pipeline_parallel_indices_remain_global(compact):
    local_layers = [Decoder(), Decoder()]
    model = Stack(local_layers if compact else [PPMissingLayer()] * 4 + local_layers)
    model.start_layer, model.end_layer = 4, 6
    assert [layer.layer_id for layer in ModelDiscovery(model).decoder_layers] == [4, 5]


def test_hybrid_stack_discovers_residual_blocks_without_model_names():
    misleading = type("NemotronHMLPDecoderLayer", (nn.Identity,), {})
    model = Stack([Decoder(), ResidualBlock(), misleading()])
    assert [layer.layer_id for layer in ModelDiscovery(model).decoder_layers] == [0, 1]


def test_bounded_stack_discovers_pure_mlp_layers_without_attention():
    model = Stack([ResidualBlock(), ResidualBlock()])
    assert ModelDiscovery(model).decoder_layers == []
    model.start_layer, model.end_layer = 4, 6
    assert [layer.layer_id for layer in ModelDiscovery(model).decoder_layers] == [4, 5]


def test_explicit_layer_index_is_shared_without_numeric_name():
    model = nn.Module()
    model.block = ResidualBlock()
    model.block.layer_idx = 8
    assert ModelDiscovery(model).decoder_layers[0].layer_id == 8
    del model.block.layer_idx
    assert ModelDiscovery(model).decoder_layers == []


def test_nested_stack_does_not_hook_its_enclosing_residual_block():
    outer = ResidualBlock()
    outer.inner = Stack([Decoder(), Decoder()])
    model = Stack([outer])
    model.start_layer, model.end_layer = 0, 1
    assert [layer.name for layer in ModelDiscovery(model).decoder_layers] == [
        "layers.0.inner.layers.0", "layers.0.inner.layers.1",
    ]


def test_independent_stacks_with_same_indices_are_ambiguous():
    model = nn.ModuleDict({"first": Stack([Decoder()]), "second": Stack([Decoder()])})
    with pytest.raises(ValueError, match="Ambiguous decoder layer index"):
        _ = ModelDiscovery(model).decoder_layers


def test_reused_module_cannot_be_assigned_distinct_hook_indices():
    block = Decoder()
    with pytest.raises(ValueError, match="Decoder module is shared"):
        _ = ModelDiscovery(Stack([block, block])).decoder_layers


def test_moe_uses_owning_decoder_index_and_runner_gate():
    gate = nn.Linear(4, 3)
    block = Decoder()
    block.moe = nn.Module()
    block.moe.experts = TinyRunner(gate)
    # A formerly listed expert-weight container must not shadow its MoE owner.
    block.moe.experts.routed_experts = type("DbrxExperts", (nn.Module,), {})()
    assert isinstance(block.moe.experts, MoERunnerInterface)
    model = nn.Sequential(Stack([Decoder(), block]))
    layers = ModelDiscovery(model).moe_blocks
    assert [(layer.name, layer.layer_id) for layer in layers] == [("0.layers.1.moe", 1)]
    assert resolve_moe_gate(layers[0].name, layers[0].module) is gate
    block.moe.experts._fse_fuse_gate = True
    assert resolve_moe_gate(layers[0].name, layers[0].module) is None


@pytest.mark.parametrize(
    "count_attr, gate_attr",
    [("num_experts", "gate"), ("num_total_experts", "gate"),
     ("num_total_experts", "router")],
)
def test_moe_with_direct_expert_kernels_needs_no_class_name(count_attr, gate_attr):
    model = Stack([Decoder()])
    block = nn.Module()
    gate = nn.Linear(4, 3)
    setattr(block, count_attr, 3)
    setattr(block, gate_attr, gate)
    block.top_k = 2
    model.layers[0].mlp = block
    discovered = ModelDiscovery(model).moe_blocks
    assert [(layer.name, layer.layer_id) for layer in discovered] == [
        ("layers.0.mlp", 0),
    ]
    assert resolve_moe_gate(discovered[0].name, block) is gate
    del block.top_k
    assert ModelDiscovery(model).moe_blocks == []


def test_capture_and_steering_share_discovered_indices(monkeypatch):
    from vllm.model_hooks.components import discovery

    model = nn.Sequential(Stack([Decoder(), Decoder()]))
    find_decoders = Mock(wraps=discovery._find_decoder_layers)
    monkeypatch.setattr(discovery, "_find_decoder_layers", find_decoders)
    components = discover_components(model)
    manager = ControllerManager(components)
    expected = [controller.layer_id for controller in manager.controllers.values()]
    assert len(manager.controllers_for_layer(1, HIDDEN_STATES)) == 1
    manager.remove_hooks()  # Capture runs on CPU without dispatching steering ops.
    session = CaptureSession()
    session.attach(model, components)
    assert find_decoders.call_count == 1
    store = StreamStore(StreamConfig(budget_rows=20))
    append = Mock(wraps=store.append)
    monkeypatch.setattr(store, "append", append)
    session._streams[HIDDEN_STATES] = store
    request_index = store.req_index("sample")
    labels = torch.tensor(
        [[request_index, 0, 10], [request_index, 1, 11]], dtype=torch.int32
    )
    monkeypatch.setattr(
        "vllm.model_hooks.capture.session.prepare_rows", lambda *args: (args[0], labels)
    )
    try:
        model(torch.zeros(2, 4))
        assert [call.args[0] for call in append.call_args_list] == expected == [0, 1]
    finally:
        session.detach()


def test_supported_decoder_output_preserves_residual():
    hidden, residual = torch.ones(2, 4), torch.full((2, 4), 2.0)
    split = outputs.split_decoder_output((hidden, residual))
    assert split[0] is hidden and split[1] is residual
    rebuilt = outputs.reconstruct_decoder_output(
        hidden + 1, *split[1:], (hidden, residual)
    )
    assert rebuilt[1] is residual
    assert torch.equal(rebuilt[0], hidden + 1)


@pytest.mark.parametrize("output", [(), {"hidden_states": torch.zeros(2, 4)}])
def test_unknown_decoder_output_fails_explicitly(output):
    with pytest.raises(TypeError):
        outputs.split_decoder_output(output)


def test_long_tuple_preserves_auxiliary_outputs_without_guessing_residual():
    hidden, auxiliary = torch.zeros(2, 4), torch.ones(2, 4)
    output = (hidden, auxiliary, None)
    split = outputs.split_decoder_output(output)
    assert split[1] is None
    rebuilt = outputs.reconstruct_decoder_output(hidden + 1, *split[1:], output)
    assert rebuilt[1] is auxiliary and rebuilt[2] is None


def test_capture_selection_refuses_missing_runner_geometry(monkeypatch):
    from vllm.model_hooks.capture.selection import prepare_rows

    ctx = SimpleNamespace(
        batch_geometry=None,
        attn_metadata=SimpleNamespace(query_start_loc=torch.tensor([0, 1])),
    )
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: ctx)
    store = SimpleNamespace(config=SimpleNamespace(selects_rows=True, reduce="all"))
    with pytest.raises(RuntimeError, match="BatchGeometry"):
        prepare_rows(torch.zeros(1, 4), store, 0, HIDDEN_STATES, {})


def test_component_adapters_preserve_residual_and_gate_bias_semantics():
    hidden = torch.ones(2, 4)
    extra = torch.full((2, 4), 2.0)
    original = (hidden, extra)
    decoder = COMPONENTS[HIDDEN_STATES].adapter
    gate = COMPONENTS[ROUTER_LOGITS].adapter
    captured, owned = decoder.capture_rows(original)
    assert owned and torch.equal(captured, hidden + extra)
    captured, owned = gate.capture_rows(original)
    assert captured is hidden and not owned
    assert decoder.read_output(original)[1] is extra
    assert gate.read_output(original)[1] is None
    for adapter in (decoder, gate):
        parts = adapter.read_output(original)
        rebuilt = adapter.write_output(hidden + 3, *parts[1:], original)
        assert rebuilt[1] is extra
        assert torch.equal(rebuilt[0], hidden + 3)
    parts = gate.read_output(original)
    assert gate.write_output(*parts, original) is original


@pytest.mark.parametrize("component_id", [HIDDEN_STATES, ROUTER_LOGITS])
def test_component_output_adapter_remains_fullgraph_traceable(component_id):
    def transform(values, extra):
        output = (values, extra)
        adapter = COMPONENTS[component_id].adapter
        parts = adapter.read_output(output)
        return adapter.write_output(parts[0] + 3, *parts[1:], output)

    values, extra = torch.ones(2, 4), torch.full((2, 4), 2.0)
    compiled = torch.compile(transform, backend="eager", fullgraph=True)
    actual = compiled(values, extra)
    assert torch.equal(actual[0], values + 3)
    assert actual[1] is extra


def test_component_availability_excludes_bypassed_fused_gate():
    model = Stack([Decoder()])
    model.layers[0].moe = nn.Module()
    model.layers[0].moe.experts = TinyRunner(nn.Linear(4, 3), fused=True)
    component = COMPONENTS[ROUTER_LOGITS]
    matches = component.discover(ModelDiscovery(model))
    assert len(matches) == 1
    assert component.resolve_target(matches[0].name, matches[0].module) is None
    components = discover_components(model)
    manager = ControllerManager(components)
    try:
        assert manager.controllers_for_layer(0, component.id) == []
        assert len(manager.controllers_for_layer(0, HIDDEN_STATES)) == 1
        with pytest.raises(ValueError, match="Unknown steering/capture component"):
            manager.controllers_for_layer(0, "unregistered")
    finally:
        manager.remove_hooks()


def test_worker_target_selection_uses_algorithm_capability(monkeypatch):
    from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

    manager = WorkerSteeringState.__new__(WorkerSteeringState)
    decoder, gate = Mock(), Mock()
    targets = {HIDDEN_STATES: decoder, ROUTER_LOGITS: gate}
    manager._controller_manager = SimpleNamespace(
        controllers_for_layer=lambda layer, kind: [targets[kind]]
    )
    specs = [({"algorithm": "direct"}, {0: torch.ones(4)})]
    manager._configure_layer_slots(2, specs, "priority")
    decoder.configure_slot.assert_called_once()
    gate.configure_slot.assert_not_called()
    monkeypatch.setitem(
        ALGORITHM_CAPABILITIES, "direct",
        replace(ALGORITHM_CAPABILITIES["direct"], target_component=ROUTER_LOGITS),
    )
    manager._configure_layer_slots(3, specs, "priority")
    gate.configure_slot.assert_called_once()
    assert gate.configure_slot.call_args.args[0] == 3
    assert decoder.configure_slot.call_count == 1


@pytest.mark.parametrize(
    "algorithms, expected",
    [(["direct"], {HIDDEN_STATES}), (["moe_router"], {ROUTER_LOGITS}),
     ("all", {HIDDEN_STATES, ROUTER_LOGITS})],
)
def test_worker_hooks_only_declared_components_without_changing_capture(
    algorithms, expected,
):
    from vllm.config import SteerVectorConfig
    from vllm.model_hooks.steering.worker_manager import WorkerSteeringState

    model = Stack([Decoder()])
    model.layers[0].moe = nn.Module()
    model.layers[0].moe.experts = TinyRunner(nn.Linear(4, 3))
    components = discover_components(model)
    worker = WorkerSteeringState(
        torch.device("cpu"),
        SteerVectorConfig(algorithms=algorithms, graph_mode="split",
                          steer_vector_dtype="float32", max_steer_vectors=2),
        hidden_size=4,
    )
    worker.attach_steering_hooks(components)
    manager = worker._controller_manager
    try:
        assert {c.component_id for c in manager.controllers.values()} == expected
        for kind, targets in components.items():
            assert len(targets) == 1
            assert len(targets[0].module._forward_hooks) == int(kind in expected)
        session = CaptureSession()
        session.attach(model, components)
        try:
            assert all(session._hooked_layers[kind] == {0} for kind in components)
        finally:
            session.detach()
    finally:
        manager.remove_hooks()


def test_component_and_algorithm_lookup_reject_unknown_targets():
    assert algorithm_target("moe_router") == ROUTER_LOGITS
    with pytest.raises(ValueError, match="Unknown steering algorithm"):
        algorithm_target("unregistered")
    with pytest.raises(ValueError, match="Unknown steering/capture component"):
        get_component("unregistered")


def test_router_op_applies_intervention_only_to_selected_token_rows(monkeypatch):
    """The eager custom-op body must use the shared controller routing path."""
    from vllm.model_hooks.steering import ops
    from vllm.model_hooks.steering.controllers import RouterLogitsController

    controller = RouterLogitsController()
    controller.configure_slot(0, [{
        "algorithm": "moe_router",
        "payload": {"mode": "deactivate", "expert_ids": [0]},
        "apply_spec": {"prompt": "all"},
    }])
    group = controller.slot_position_groups[0]
    context = SimpleNamespace(
        steer_active_slots=[0],
        steer_slot_positions={(0, group, 0): torch.tensor([0])},
    )
    monkeypatch.setattr(
        "vllm.model_hooks.steering.controllers.base.get_forward_context",
        lambda: context,
    )
    original = torch.tensor([[3., 2., 1.], [4., 3., 2.]])
    logits = original.clone()
    expected = original.clone()
    expected[0] = torch.log_softmax(original[0], dim=-1)
    expected[0, 0] = expected[0].min() - 0.01
    key = "test-router-op::gate"
    ops.register_controller(key, controller)
    try:
        # Call the registered implementation on CPU: GPU dispatch and the
        # full-graph kernel are covered by integration/kernel tests separately.
        ops.steer_moe_gate(logits, key)
        torch.testing.assert_close(logits, expected)
        assert torch.equal(logits[1], original[1])
    finally:
        ops.unregister_controller(key, controller)
