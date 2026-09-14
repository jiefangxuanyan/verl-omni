#!/usr/bin/env bash
# One-GPU, one-step Qwen-Image smoke test for GC diagnostics.
set -euo pipefail

kernel_release=$(uname -r)
if [[ ${kernel_release,,} == *microsoft-standard-wsl2* ]]; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    export VERL_FORCE_SHM_WEIGHT_TRANSFER=1
fi

compile_disabled_log=$(mktemp /tmp/verl-omni-gc-diagnostics-compile-disabled-XXXXXX.log)
compile_enabled_log=$(mktemp /tmp/verl-omni-gc-diagnostics-compile-enabled-XXXXXX.log)
extra_args=("$@")

run_case() {
    local use_regional_compile=$1
    local log_file=$2

    if ! NUM_GPUS=1 \
        bash tests/special_e2e/run_flowgrpo_qwen_image.sh \
            "${extra_args[@]}" \
            actor_rollout_ref.actor.strategy=fsdp2 \
            actor_rollout_ref.gc_diagnostics=True \
            actor_rollout_ref.model.use_regional_compile="${use_regional_compile}" \
            reward.reward_model.enable=False \
            reward.custom_reward_function.path=null \
            reward.custom_reward_function.name=null \
            reward.reward_manager.name=VisualRewardManager \
            trainer.total_training_steps=1 \
            2>&1 | tee "${log_file}"; then
        echo "GC diagnostics smoke failed; log preserved at ${log_file}" >&2
        exit 1
    fi
}

run_case False "${compile_disabled_log}"

for point in train_device_load eval_device_load actor_offload weight_transfer_cleanup; do
    if ! grep -q "\[gc_diagnostics\] point=${point} .* generation=full " "${compile_disabled_log}"; then
        echo "Missing GC diagnostics point=${point} generation=full; log preserved at ${compile_disabled_log}" >&2
        exit 1
    fi
done

run_case True "${compile_enabled_log}"

for point in train_device_load eval_device_load actor_offload; do
    if grep -q "\[gc_diagnostics\] point=${point} " "${compile_enabled_log}"; then
        echo "Unexpected GC diagnostics point=${point}; log preserved at ${compile_enabled_log}" >&2
        exit 1
    fi
done

if ! grep -q '\[gc_diagnostics\] point=weight_transfer_cleanup .* generation=1 ' "${compile_enabled_log}"; then
    echo "Missing GC diagnostics point=weight_transfer_cleanup generation=1; log preserved at ${compile_enabled_log}" >&2
    exit 1
fi

rm -f "${compile_disabled_log}" "${compile_enabled_log}"
echo "GC diagnostics smoke passed."
