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
"""CPU tests for the intentional FA3 varlen metadata graph break."""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models import attention_dispatch

from verl_omni.workers.engine.fsdp.diffusers_impl import _keep_varlen_attention_metadata_eager

_HELPER_NAMES = (
    "_prepare_for_flash_attn_or_sage_varlen_with_mask",
    "_prepare_for_flash_attn_or_sage_varlen_without_mask",
)
_ORIGINAL_HELPERS = {helper_name: getattr(attention_dispatch, helper_name) for helper_name in _HELPER_NAMES}


@pytest.mark.parametrize("with_mask", [False, True])
def test_fa3_varlen_preparation_runs_outside_compiled_region(monkeypatch, with_mask):
    helper_name = (
        "_prepare_for_flash_attn_or_sage_varlen_with_mask"
        if with_mask
        else "_prepare_for_flash_attn_or_sage_varlen_without_mask"
    )
    for name, helper in _ORIGINAL_HELPERS.items():
        monkeypatch.setattr(attention_dispatch, name, helper)
    original_prepare = _ORIGINAL_HELPERS[helper_name]
    preparation_compile_states = []

    def observed_prepare(*args, **kwargs):
        preparation_compile_states.append(torch.compiler.is_compiling())
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(attention_dispatch, helper_name, observed_prepare)
    backend = attention_dispatch.AttentionBackendName._FLASH_3_VARLEN_HUB
    monkeypatch.setitem(
        attention_dispatch._HUB_KERNELS_REGISTRY,
        backend,
        SimpleNamespace(kernel_fn=lambda **kwargs: kwargs["q"]),
    )
    _keep_varlen_attention_metadata_eager()

    compiled_backend = torch.compile(
        attention_dispatch._flash_attention_3_varlen_hub,
        backend="inductor",
        fullgraph=False,
        dynamic=True,
    )
    query = torch.ones((1, 4, 1, 8))
    mask = torch.tensor([[True, False, True, False]]) if with_mask else None

    output = compiled_backend(query, query, query, attn_mask=mask)

    assert preparation_compile_states == [False]
    torch.testing.assert_close(output, query)


def test_varlen_preparation_wrappers_are_installed_once(monkeypatch):
    for helper_name, helper in _ORIGINAL_HELPERS.items():
        monkeypatch.setattr(attention_dispatch, helper_name, helper)

    _keep_varlen_attention_metadata_eager()
    wrapped_helpers = tuple(getattr(attention_dispatch, helper_name) for helper_name in _HELPER_NAMES)
    _keep_varlen_attention_metadata_eager()

    assert tuple(getattr(attention_dispatch, helper_name) for helper_name in _HELPER_NAMES) == wrapped_helpers
