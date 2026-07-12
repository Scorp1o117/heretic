#!/bin/bash
# Run Heretic ARA LoRA for Ornith-1.0-35B on ROCm + bnb 4-bit
set -e
cd "$(dirname "$0")"

export QWEN35_MOE_UNPACK_EXPERTS=1
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1

source /home/s117/heretic-env/bin/activate
exec heretic "$@"
