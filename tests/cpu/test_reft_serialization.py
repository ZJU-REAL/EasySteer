"""Real CPU save/load contracts for intervention metadata and source buffers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from easysteer.reft.pyreft.core import IntervenableConfig, IntervenableModel
from easysteer.reft.pyreft.core.modeling.basic_utils import get_type_from_string
from easysteer.reft.pyreft.core.modeling.interventions import AdditionIntervention


class ProjectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_attention_heads=2)
        # The explicit component path has no profile dimension. Its output
        # width deliberately differs from hidden_size to expose guessed widths.
        self.projection = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(32).reshape(8, 4) / 32)

    @property
    def device(self):
        return self.projection.weight.device

    def forward(self, input_ids):
        return self.projection(input_ids)


@pytest.mark.parametrize("dimension", [None, 3])
@pytest.mark.parametrize("source_in_config", [False, True])
def test_constant_source_save_load_forward(tmp_path, dimension, source_in_config):
    model = ProjectionModel()
    source = torch.arange(1, 9, dtype=torch.float32)
    representation = {"component": "projection.output"}
    if source_in_config:
        representation["source_representation"] = source
    else:
        representation["intervention"] = AdditionIntervention(
            source_representation=source
        )
    wrapper = IntervenableModel(
        IntervenableConfig(
            representations=representation,
            intervention_types=AdditionIntervention,
        ),
        model,
    )
    intervention = next(iter(wrapper.interventions.values()))
    intervention.set_interchange_dim(dimension)
    inputs = {"input_ids": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)}
    locations = {"sources->base": (None, [[[1]]])}
    expected = model(**inputs).detach().clone()
    expected[:, 1, :dimension] += source[:dimension]
    torch.testing.assert_close(wrapper(inputs, unit_locations=locations)[1], expected)

    directory = tmp_path / "checkpoint"
    wrapper.save(str(directory))
    metadata = json.loads((directory / "config.json").read_text())
    assert metadata["intervention_dimensions"] == [dimension]
    restored = IntervenableModel.load(str(directory), model)
    loaded = next(iter(restored.interventions.values()))
    if dimension is None:
        assert loaded.interchange_dim is None
    else:
        assert loaded.interchange_dim.item() == dimension
    assert "source_representation" in dict(loaded.named_buffers())
    torch.testing.assert_close(loaded.source_representation, source)
    torch.testing.assert_close(restored(inputs, unit_locations=locations)[1], expected)

    # A restored source must move with the module and survive another save.
    loaded.set_source_representation(source.clone())
    restored.to(dtype=torch.float64)
    assert loaded.source_representation.dtype == torch.float64
    torch.testing.assert_close(
        restored({"input_ids": inputs["input_ids"].double()}, unit_locations=locations)[
            1
        ],
        expected.double(),
    )
    restored.save(str(tmp_path / "resaved"))
    state = torch.load(next((tmp_path / "resaved").glob("*.bin")))
    torch.testing.assert_close(state["source_representation"], source.double())


def test_interchange_dimension_can_clear_and_restore_registered_buffer():
    intervention = AdditionIntervention(embed_dim=8)
    intervention.set_interchange_dim(None)
    assert intervention.interchange_dim is None
    intervention.set_interchange_dim(3)
    assert intervention.interchange_dim.item() == 3
    assert "interchange_dim" in dict(intervention.named_buffers())


@pytest.mark.parametrize(
    "relative_path",
    [
        "tests/fixtures/reft/loreft/config.json",
        "tests/fixtures/reft/ssv/config.json",
        "hf-space/results/emoji_loreft/config.json",
        "replications/loreft/weight/config.json",
    ],
)
def test_published_intervention_types_are_importable(relative_path):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / relative_path).read_text())
    for type_name in config["intervention_types"]:
        intervention_type = get_type_from_string(type_name)
        assert issubclass(intervention_type, torch.nn.Module)
        assert intervention_type.__module__.startswith("easysteer.reft.pyreft.")
