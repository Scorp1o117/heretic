# Classic Ornith-1.0-35B abliteration (vs ARA-LoRA)

## Why this path

Public uncensored Ornith-35B builds (e.g. llmfan46 heretic GGUF, AEON-7 BF16)
report **KL ≈ 0.001–0.002** with **~0–9/100 refusals** using **classic /
MPOA-style direction ablation**, not ARA-LoRA.

Local ARA-LoRA search on the same model hit a Pareto cliff:

| Region | Typical KL | Refusals |
|--------|------------|----------|
| Weak ARA | ≤ 0.05 | 37–60 |
| Best usable ARA (Trial 3) | ~0.27 | 8 |
| Over-ablated ARA | ≥ 6 | 0 (often collapsed) |

This classic run is the controlled experiment: same data, same ROCm + bnb4bit
stack, different algorithm.

## How to run

```bash
cd /home/s117/heretic-ara-lora
./run-ornith-classic.sh
# optional overrides:
./run-ornith-classic.sh --n-trials 80 --n-startup-trials 30
```

Requirements (same as ARA path):

```bash
export QWEN35_MOE_UNPACK_EXPERTS=1
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
```

(The script sets these.)

## Config summary (`config.classic-ornith.toml`)

| Setting | Value | Notes |
|---------|-------|--------|
| `use_ara` | `false` | Classic ortho ablation |
| `use_ara_lora` | `false` | Rank-1/3 LoRA write, not LBFGS |
| `row_normalization` | `full` | MPOA-style magnitude preserve |
| `target_components` | `attn.o_proj`, `mlp.down_proj` | Hybrid maps linear_attn.out_proj → o_proj |
| `kl_divergence_target` | `0.01` | Hunt low-KL solutions |
| `study_checkpoint_dir` | `checkpoints-classic` | Isolated from ARA journal |
| `quantization` | `bnb_4bit` | Fits 8060S UMA; KL is vs 4bit base |

## Search space (code)

`main.py` classic ranges were widened to cover published Ornith-35B wins:

- `direction_index`: 35%–85% of last layer (~13.6–33.2 for 40 layers)
- `max_weight`: 0.5–2.1 (llmfan peak ~1.98)
- `max_weight_position`: mid-to-late stack
- `min_weight_distance`: up to 75% of last layer index

On a **fresh** classic study (empty `checkpoints-classic/`), trial 0 is
**enqueued** with llmfan46-style parameters (`direction_index=20.57`,
o_proj/mlp weight schedules) so the first evaluation is a direct transfer test.


## Success criteria (gentle-knee)

Prefer shipping a trial that is **all** of:

1. Full refusals ≤ 15/100 (≤ 9 is excellent)
2. Optimization KL ≤ 0.05 (≤ 0.01 ideal; public cards ~0.002)
3. Validation KL finite and not >> opt KL
4. Generation health **passed** (repetition / empty — JSON is soft-only)
5. Prefer lighter edits among similar refusals (do **not** pick min-refusal if KL exploded)

## Restore ARA config.toml

`run-ornith-classic.sh` copies the classic TOML over `config.toml` and keeps
`config.toml.ara-backup` if it had to overwrite a real ARA config:

```bash
cp -f config.toml.ara-backup config.toml   # back to ARA-LoRA
# or: cp -f config.classic-ornith.toml config.toml
```

## Related public references

- llmfan46: Heretic 1.2.0 + MPOA variant, KL 0.0019, 9/100 refusals
- AEON-7: abliterix/EGA, KL ~0.0014, expert-granular, gentle-knee selection
- huihui: 9B dense GGUF ablation (architecture differs; do not copy layer counts)
