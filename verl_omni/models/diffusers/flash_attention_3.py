# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import logging

import diffusers
import torch
from packaging import version

logger = logging.getLogger(__name__)


# TODO (mike): drop this once it is fixed in upstream diffusers.
def apply_flash_attention_3_varlen_hub_fix() -> None:
    """Patch ``_flash_attention_3_varlen_hub`` to support non-contiguous attention masks.

    diffusers==0.38.0 packs the valid key/value tokens with ``key[b, :valid_len]``, which
    only selects the correct tokens when the valid positions form a contiguous prefix.
    Qwen-Image joint text/image masks are *not* contiguous, so the wrong key/value tokens
    get gathered. We instead select the actual valid positions via
    ``attn_mask.flatten().nonzero()`` (matching the upstream diffusers ``fa3`` branch fix).

    Remove this patch once the fix is upstreamed to diffusers.
    """
    if version.parse(diffusers.__version__) < version.parse("0.38.0"):
        return

    from diffusers.models import attention_dispatch as _ad

    registry = _ad._AttentionBackendRegistry
    backend = _ad.AttentionBackendName._FLASH_3_VARLEN_HUB

    current = registry._backends.get(backend)
    if current is None or getattr(current, "_verl_omni_fa3_varlen_patched", False):
        return

    # Keep varlen metadata preparation outside compiled regions. This single
    # intentional graph boundary avoids two independent full-graph failures:
    #
    # 1. For dense queries, Diffusers builds cumulative lengths as
    #    full((batch_size,), seq_len_q).cumsum(). With dynamic shapes, PyTorch
    #    2.13's pointless-cumsum Inductor pass mistakes the symbolic fill value
    #    for a regular scalar while rewriting that expression.
    # 2. For masked keys, Diffusers computes max_seqlen_k from Tensor data with
    #    seqlens_k.max().item(). Inside a graph this is an unbacked SymInt, but
    #    FA3's fake backward uses it in Python control flow to choose window,
    #    kernel, and workspace behavior, causing GuardOnDataDependentSymNode.
    #
    # Eager preparation bypasses the faulty cumsum rewrite and materializes
    # max_seqlen_q/k as concrete Python integers before FA3 is traced. Dynamo
    # resumes after this call, so packing, the FA3 custom op, and the rest of
    # the transformer block remain eligible for regional compilation. Once
    # upstream fixes both the symbolic cumsum lowering and FA3's handling or
    # contract for data-dependent max_seqlen values, remove this disable
    # boundary and revalidate the block with fullgraph=True.
    prepare_varlen = torch.compiler.disable(_ad._prepare_for_flash_attn_or_sage_varlen)

    def _patched_flash_attention_3_varlen_hub(
        query,
        key,
        value,
        attn_mask=None,
        scale=None,
        is_causal=False,
        return_lse=False,
        _parallel_config=None,
    ):
        batch_size, seq_len_q, _, _ = query.shape
        _, seq_len_kv, _, _ = key.shape

        if attn_mask is not None:
            attn_mask = _ad._normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

        (_, _seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = prepare_varlen(
            batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
        )

        query_packed = query.flatten(0, 1)
        if attn_mask is not None:
            # Gather valid key/value tokens by their actual (possibly non-contiguous)
            # positions rather than assuming a contiguous prefix `key[b, :valid_len]`.
            indices_k = attn_mask.flatten().nonzero(as_tuple=False).flatten()
            key_packed = key.reshape(-1, *key.shape[2:])[indices_k]
            value_packed = value.reshape(-1, *value.shape[2:])[indices_k]
        else:
            key_packed = key.flatten(0, 1)
            value_packed = value.flatten(0, 1)

        func = _ad._HUB_KERNELS_REGISTRY[backend].kernel_fn
        out = func(
            q=query_packed,
            k=key_packed,
            v=value_packed,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=return_lse,
        )
        lse = None
        if return_lse:
            out, lse, *_ = out
        out = out.unflatten(0, (batch_size, -1))

        return (out, lse) if return_lse else out

    _patched_flash_attention_3_varlen_hub._verl_omni_fa3_varlen_patched = True

    registry._backends[backend] = _patched_flash_attention_3_varlen_hub
    _ad._flash_attention_3_varlen_hub = _patched_flash_attention_3_varlen_hub
    logger.warning(
        "verl_omni patch applied: diffusers `_flash_attention_3_varlen_hub` has been "
        "monkey-patched to gather key/value tokens by non-contiguous mask indices "
        "instead of assuming a contiguous prefix (diffusers==0.38). "
        "Remove this patch once the fix is upstreamed to diffusers."
    )
