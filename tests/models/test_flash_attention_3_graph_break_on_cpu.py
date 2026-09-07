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
"""CPU tests for the intentional FA3 varlen preparation graph break."""

from types import SimpleNamespace
from unittest.mock import patch

import diffusers
import torch

from verl_omni.models.diffusers import flash_attention_3


def test_varlen_preparation_runs_outside_compiled_region(monkeypatch):
    backend = "fake_flash_3_varlen_hub"
    preparation_compile_states = []

    class FakeAttentionBackendName:
        _FLASH_3_VARLEN_HUB = backend

    class FakeAttentionBackendRegistry:
        _backends = {backend: lambda: None}

    def fake_prepare(batch_size, seq_len_q, seq_len_kv, *, attn_mask, device):
        preparation_compile_states.append(torch.compiler.is_compiling())
        seqlens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int32, device=device)
        seqlens_k = attn_mask.sum(dim=1, dtype=torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(seqlens_q.cumsum(dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(seqlens_k.cumsum(dim=0), (1, 0))
        return (
            (seqlens_q, seqlens_k),
            (cu_seqlens_q, cu_seqlens_k),
            (seq_len_q, int(seqlens_k.max())),
        )

    def fake_kernel(**kwargs):
        metadata = kwargs["k"].sum() + kwargs["cu_seqlens_q"].sum() + kwargs["cu_seqlens_k"].sum()
        return kwargs["q"] + metadata * 0

    fake_attention_dispatch = SimpleNamespace(
        _AttentionBackendRegistry=FakeAttentionBackendRegistry,
        AttentionBackendName=FakeAttentionBackendName,
        _HUB_KERNELS_REGISTRY={backend: SimpleNamespace(kernel_fn=fake_kernel)},
        _normalize_attn_mask=lambda attn_mask, _batch_size, _seq_len_kv: attn_mask,
        _prepare_for_flash_attn_or_sage_varlen=fake_prepare,
    )
    monkeypatch.setattr(diffusers, "__version__", "0.38.0")
    monkeypatch.setattr(diffusers.models, "attention_dispatch", fake_attention_dispatch)

    with patch.object(flash_attention_3.logger, "warning") as warning:
        flash_attention_3.apply_flash_attention_3_varlen_hub_fix()
        warning.assert_called_once()

        compiled_backend = torch.compile(
            FakeAttentionBackendRegistry._backends[backend],
            backend="inductor",
            fullgraph=False,
            dynamic=True,
        )
        query = torch.ones((1, 2, 1, 1))
        output = compiled_backend(query, query, query, attn_mask=torch.tensor([[True, False]]))

    assert preparation_compile_states == [False]
    torch.testing.assert_close(output, query)
