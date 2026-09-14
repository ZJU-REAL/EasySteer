# SPDX-License-Identifier: Apache-2.0
"""Named worker RPCs for observing real TP router interventions in tests."""


class RouterProbeWorkerExtension:
    def router_test_observe(self):
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        manager = self.model_runner.steer_vector_manager._controller_manager
        controller, = manager.controllers_for_layer(
            layer_id=0, component_id="router_logits"
        )
        original = controller.apply_steering
        records = []

        def record(logits, residual=None):
            before = logits.clone()
            result = original(logits, residual)
            with gpu_sync_allowed():
                records.append((result != before).cpu().tolist())
            return result

        controller.apply_steering = record
        self._router_tp_probe = (controller, records)
        return get_tensor_model_parallel_rank()

    def router_test_collect(self):
        state = getattr(self, "_router_tp_probe", None)
        if state is None:
            return []
        controller, records = state
        del controller.apply_steering
        del self._router_tp_probe
        return records
