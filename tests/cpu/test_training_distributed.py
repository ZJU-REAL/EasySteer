# SPDX-License-Identifier: Apache-2.0
"""Distributed training contracts with real gloo processes and unequal lengths."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from easysteer.training.api import _response_loss, _training_arguments

WORKER = Path(__file__).resolve().parents[1] / "training" / "ddp_check.py"


@pytest.mark.parametrize("algorithm", ["direct", "loreft"])
def test_two_ranks_match_global_batch_and_only_main_rank_exports(tmp_path, algorithm):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    }
    env.update(
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false"
    )
    arguments = ["--device", "cpu", "--algorithm", algorithm, "--output", str(tmp_path)]
    for launcher, mode in (
        ([sys.executable], "reference"),
        (
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
            ],
            "distributed",
        ),
    ):
        result = subprocess.run(
            [*launcher, str(WORKER), "--mode", mode, *arguments],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    reports = [
        json.loads((tmp_path / "distributed" / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    assert [report["save_calls"] for report in reports] == [3, 0]
    assert all(report["world_size"] == 2 for report in reports)


def test_response_loss_weights_tokens_instead_of_unequal_microbatch_means():
    logits = torch.tensor(
        [[[1.0, -1.0], [2.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [2.0, 0.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, -100], [-100, 1, 0]])
    expected = -torch.stack(
        [
            logits[0, 0].log_softmax(-1)[0],
            logits[1, 0].log_softmax(-1)[1],
            logits[1, 1].log_softmax(-1)[0],
        ]
    ).mean()
    accumulated = sum(
        _response_loss(
            SimpleNamespace(logits=logits[index : index + 1]),
            labels[index : index + 1],
            num_items_in_batch=3,
        )
        for index in range(2)
    )
    torch.testing.assert_close(accumulated, expected)
    actual_gradient = torch.autograd.grad(accumulated, logits, retain_graph=True)[0]
    torch.testing.assert_close(
        actual_gradient, torch.autograd.grad(expected, logits)[0]
    )


@pytest.mark.parametrize(
    "option", ["fsdp", "deepspeed", "parallelism_config", "tensor_parallel_size"]
)
def test_sharded_training_rejected_before_model_loading(option):
    with pytest.raises(ValueError, match="DDP|tensor parallelism"):
        _training_arguments("cpu", {option: "unsupported"})


def test_conflicting_device_policy_rejected():
    with pytest.raises(ValueError, match="different device types"):
        _training_arguments("cpu", {"use_cpu": False})
