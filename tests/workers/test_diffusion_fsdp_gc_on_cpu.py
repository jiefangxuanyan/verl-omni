# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verl_omni.workers.config.diffusion.actor import DiffusionFSDPEngineConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine, PPODiffusersFSDPEngine


def make_engine(*, mode, train_gc=True, eval_gc=True, diagnostics=False):
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.engine_config = DiffusionFSDPEngineConfig(
        forward_only=False,
        gc_on_train_device_load=train_gc,
        gc_on_eval_device_load=eval_gc,
    )
    engine.gc_diagnostics = diagnostics
    engine.optimizer = None
    engine.mode = mode
    return engine


@pytest.mark.parametrize(
    "mode,train_gc,eval_gc",
    [("train", False, True), ("eval", True, False)],
)
def test_gc_can_be_disabled_independently(monkeypatch, mode, train_gc, eval_gc):
    collect = Mock()
    engine = make_engine(mode=mode, train_gc=train_gc, eval_gc=eval_gc)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl.utils.memory_utils.gc.collect", collect)

    DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)

    collect.assert_not_called()


@pytest.mark.parametrize(
    "mode,train_gc,eval_gc,expected_generation",
    [("train", 0, 1, 0), ("eval", 0, 1, 1)],
)
def test_mode_specific_gc_generation_is_forwarded(monkeypatch, mode, train_gc, eval_gc, expected_generation):
    collect = Mock()
    engine = make_engine(mode=mode, train_gc=train_gc, eval_gc=eval_gc)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl.utils.memory_utils.gc.collect", collect)

    DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)

    collect.assert_called_once_with(expected_generation)


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_device_load_forwards_diagnostics_point(monkeypatch, mode, diagnostics):
    collect = Mock()
    engine = make_engine(mode=mode, diagnostics=diagnostics)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.collect_garbage", collect)

    DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)

    expected_point = f"{mode}_device_load" if diagnostics else None
    collect.assert_called_once_with(True, diagnostics_point=expected_point)


@pytest.mark.parametrize(
    ("diagnostics", "expected_point"),
    [(False, None), (True, "actor_offload")],
)
def test_actor_offload_forwards_diagnostics_point(monkeypatch, diagnostics, expected_point):
    from verl_omni.workers import engine_workers

    cleanup = Mock()
    worker = SimpleNamespace(
        actor=SimpleNamespace(engine=SimpleNamespace(is_param_offload_enabled=False)),
        config=SimpleNamespace(rollout=SimpleNamespace(gc_on_actor_offload=1)),
        gc_diagnostics=diagnostics,
    )
    monkeypatch.setattr(engine_workers, "aggressive_empty_cache", cleanup)

    engine_workers.ActorRolloutRefWorker._offload_actor_and_empty_cache(worker)

    cleanup.assert_called_once_with(
        force_sync=True,
        gc_setting=1,
        gc_diagnostics_point=expected_point,
    )


def test_accelerator_load_requires_engine_context(monkeypatch):
    engine = make_engine(mode=None)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")

    with pytest.raises(RuntimeError, match="requires a train or eval context"):
        DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)


def test_cpu_offload_remains_available_without_engine_context(monkeypatch):
    engine = make_engine(mode=None)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")

    DiffusersFSDPEngine.to(engine, device="cpu", model=False, optimizer=False, grad=False)
