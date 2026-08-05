# SPDX-License-Identifier: Apache-2.0
"""SteerVectorConfig workload-declaration validation.

The declaration (steer_algorithms) is the user-facing steering
contract: names are validated against the registry, "all" is exclusive,
and the retired graph-mode names point at their replacements. The
resolution ladder itself (auto -> in_graph/split) runs in VllmConfig
finalization and is covered end to end by the e2e engine modules.
"""

import pytest

from vllm.config.steer_vector import SteerVectorConfig


class TestAlgorithmsDeclaration:
    def test_list_is_normalized_sorted_deduped(self):
        cfg = SteerVectorConfig(algorithms=["lm_steer", "direct", "direct"])
        assert cfg.algorithms == ["direct", "lm_steer"]

    def test_comma_string_form(self):
        cfg = SteerVectorConfig(algorithms="lm_steer, direct")
        assert cfg.algorithms == ["direct", "lm_steer"]

    def test_all_wildcard(self):
        assert SteerVectorConfig(algorithms="all").algorithms == "all"

    def test_all_cannot_mix_with_names(self):
        with pytest.raises(Exception, match="cannot be combined"):
            SteerVectorConfig(algorithms=["all", "direct"])

    def test_unknown_name_rejected_with_available_list(self):
        with pytest.raises(Exception, match="available"):
            SteerVectorConfig(algorithms=["direct", "does_not_exist"])

    def test_empty_declaration_rejected(self):
        with pytest.raises(Exception, match="must not be empty"):
            SteerVectorConfig(algorithms=[])

    def test_none_passes_validator(self):
        """None defers to VllmConfig finalization, which derives the
        declaration from steering_config or raises the declare-your-
        workload error."""
        assert SteerVectorConfig().algorithms is None


class TestGraphModeValues:
    def test_retired_names_point_at_replacements(self):
        with pytest.raises(Exception, match="in_graph"):
            SteerVectorConfig(graph_mode="full")
        with pytest.raises(Exception, match="split"):
            SteerVectorConfig(graph_mode="piecewise")

    def test_unknown_value_rejected(self):
        with pytest.raises(Exception, match="steer_graph_mode"):
            SteerVectorConfig(graph_mode="eager")

    def test_valid_values_accepted(self):
        for mode in ("auto", "in_graph", "split"):
            assert SteerVectorConfig(graph_mode=mode).graph_mode == mode
