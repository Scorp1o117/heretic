# Agents-A1 ROCm/bitsandbytes 4-bit notes

This document records the extra setup needed to run Agents-A1 / Qwen3.5-MoE
with Heretic, ROCm/HIP, and bitsandbytes 4-bit quantization.

## Problem

Setting Heretic to `quantization = "bnb_4bit"` is not enough for Agents-A1.
Qwen3.5-MoE stores routed expert weights as packed 3D parameters by default.
bitsandbytes replaces normal `nn.Linear` modules, so those packed expert
parameters can remain in FP16/BF16 and exhaust VRAM.

The fix is to make the routed experts load as ordinary expert MLP modules before
Transformers applies the bnb quantizer.

## Heretic settings

The tested configuration uses LoRA-based ARA and saves LoRA adapters instead of
merging into the base model:

```toml
dtypes = ["float16"]
quantization = "bnb_4bit"
device_map = "auto"
max_memory = {"0" = "92GB", "cpu" = "24GB"}
trust_remote_code = true

use_ara = true
use_ara_lora = true
ara_lora_rank = 128
target_components = ["attn.o_proj", "mlp.down_proj"]
row_normalization = "full"

batch_size = 256
max_batch_size = 256
skip_common_response_prefix = true
kl_divergence_target = 0.02
```

Tune `max_memory` and `batch_size` for the local GPU. On Windows UMA systems,
Task Manager may report both dedicated and shared GPU memory; avoid assuming that
shared memory behaves like fast VRAM.

## Required environment variables

Set these before launching Heretic:

```bash
export QWEN35_MOE_UNPACK_EXPERTS=1
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
```

For debugging bnb quantization, also set:

```bash
export BNB_QUANT_TRACE=1
```

## Required Transformers patch

Patch the installed Transformers package or carry an equivalent downstream patch.

In `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`:

1. Import `os`.
2. In `Qwen3_5MoeExperts.__init__`, if `QWEN35_MOE_UNPACK_EXPERTS` is set:
   - set `config._experts_implementation = "eager"`
   - create `self.experts = nn.ModuleList([...Qwen3_5MoeMLP...])`
   - do not create packed `gate_up_proj` / `down_proj` parameters.
3. In `Qwen3_5MoeExperts.forward`, route active tokens through
   `self.experts[int(expert_idx.item())](current_state)` when unpacked.

In `transformers/conversion_mapping.py`, before quantizer conversion updates,
remove the default expert-packing `WeightConverter`s when
`QWEN35_MOE_UNPACK_EXPERTS` is set, then rename checkpoint keys:

```python
if os.environ.get("QWEN35_MOE_UNPACK_EXPERTS"):
    weight_conversions = [
        conversion
        for conversion in weight_conversions
        if not (
            isinstance(conversion, WeightConverter)
            and any("mlp.experts.*." in pattern for pattern in conversion.source_patterns)
        )
    ]
    weight_conversions.append(
        WeightRenaming(
            source_patterns=r"mlp\.experts\.(\d+)\.",
            target_patterns=r"mlp.experts.experts.\1.",
        )
    )
```

If allocator warmup OOMs before quantization, patch
`transformers/modeling_utils.py` so `caching_allocator_warmup` returns early when
`TRANSFORMERS_SKIP_ALLOCATOR_WARMUP` is set.

## Verification

Before running a long Heretic job, verify that routed experts become bnb
`Linear4bit` modules:

```python
import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

assert os.environ.get("QWEN35_MOE_UNPACK_EXPERTS") == "1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Agents-A1",
    dtype="float16",
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

expert = model.model.layers[1].mlp.experts.experts[0]
print(type(expert.gate_proj), type(expert.up_proj), type(expert.down_proj))
assert "Linear4bit" in type(expert.gate_proj).__name__
assert "Linear4bit" in type(expert.up_proj).__name__
assert "Linear4bit" in type(expert.down_proj).__name__
```

If `mlp.experts.experts[0]` does not exist, the unpack/conversion patch is not
active. If the projections are plain `Linear`, bnb did not quantize the experts.

## Export guidance

Do not merge a 4-bit LoRA result into the full base model on a memory-constrained
machine. Merging requires loading/dequantizing the base model in system RAM. Save
the LoRA adapter only, then merge later on a machine with enough RAM.
