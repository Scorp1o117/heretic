# Ornith-1.0-35B-Heretic Audit Report

**Model path:** `/run/media/s117/OS/Models/Ornith-1.0-35B-Heretic`
**Original path:** `/run/media/s117/OS/Models/Ornith-1.0-35B`
**Timestamp:** 2026-07-12T14:11:10+0800
**Overall status:** PASS
**Warnings:** 16
**Errors:** 0

## Warnings
- [1_file_shard_integrity] total_size mismatch: index=69321221376, actual=69325606920
- [4_numerical_health] Extreme outlier tensors: 1
- [6_compare_original] Dtype mismatches: [('model.language_model.layers.17.mlp.experts.225.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.18.mlp.experts.124.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.2.mlp.experts.91.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.14.mlp.experts.109.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.1.mlp.experts.61.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.1.mlp.experts.136.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.7.mlp.experts.210.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.0.mlp.experts.73.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.23.mlp.experts.146.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.2.mlp.experts.214.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.23.mlp.experts.40.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.33.mlp.experts.9.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.32.mlp.experts.227.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.7.mlp.experts.9.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.39.mlp.experts.177.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.9.mlp.experts.27.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.2.mlp.experts.30.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.29.mlp.experts.54.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.3.mlp.experts.48.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.1.mlp.experts.242.up_proj.weight', 'torch.bfloat16', 'torch.float16')]
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.3.self_attn.q_proj.weight (l2=1.60845e-06, cos=1.00112)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.19.self_attn.q_proj.weight (l2=1.73662e-06, cos=1.00118)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: lm_head.weight (l2=8.25056e-06, cos=1.21356)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.23.self_attn.q_proj.weight (l2=1.66131e-06, cos=1.0016)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.15.self_attn.q_proj.weight (l2=1.73801e-06, cos=1.00115)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.27.self_attn.q_proj.weight (l2=1.65453e-06, cos=1.00127)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.embed_tokens.weight (l2=1.00538e-05, cos=1.18755)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.7.self_attn.q_proj.weight (l2=1.65218e-06, cos=1.00102)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.11.self_attn.q_proj.weight (l2=1.71613e-06, cos=1.00103)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.39.self_attn.q_proj.weight (l2=1.63869e-06, cos=1.00141)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.31.self_attn.q_proj.weight (l2=1.64579e-06, cos=1.00129)
- [6_compare_original] UNCHANGED pattern tensor CHANGED: model.language_model.layers.35.self_attn.q_proj.weight (l2=1.64971e-06, cos=1.00143)
- [7_minimal_load] Tokenizer vocab_size (248044) != config (248320)

## 1_file_shard_integrity

  - Metadata: {"total_parameters": 34660610688, "total_size": 69321221376}
- **PASS** All 2 referenced shards exist on disk
- **PASS** All shard files are referenced in index
- **PASS** Shard numbering consecutive: 1-2
- **WARN** total_size mismatch: index=69321221376, actual=69325606920
- **PASS** No duplicate tensor keys (31333 unique)
  - Total tensors: 31333
  - Total shards: 2
  - Total size on disk: 69325606920 bytes (64.56 GB)
  - Total parameters (from metadata): 34660610688
  -   model-00001-of-00002.safetensors: 50002609328 bytes, SHA256=b98e94e1861091d9...
  -   model-00002-of-00002.safetensors: 19322997592 bytes, SHA256=d1d592176feee5a8...

## 2_safetensors_readability

- **PASS** All 31333 tensors readable across 2 shards
- **PASS** All dtypes supported: {'torch.float16'}
- **PASS** No zero-dim tensors
  - Dtypes seen: {'torch.float16'}
  - Total tensors read: 31333

## 3_model_structure

  - num_hidden_layers=40
  - hidden_size=2048
  - num_attention_heads=16
  - num_key_value_heads=2
  - num_experts=256
  - moe_intermediate_size=512
  - shared_expert_intermediate_size=512
  - head_dim=256
  - vocab_size=248320
  - tie_word_embeddings=False
- **PASS** layer_types length matches num_hidden_layers (40)
- **PASS** embed_tokens.weight exists
- **PASS** lm_head.weight exists
- **PASS** model.language_model.norm.weight exists
  - tie_word_embeddings=False
  - Full attention layers (10): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
  - Linear attention layers (30): [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38]
- **PASS** All layer structures verified OK (norms, attn, experts, shared_expert, gate)

## 4_numerical_health

  - Category 'embedding': 1 tensors
  - Category 'lm_head': 1 tensors
  - Category 'norms': 101 tensors
  - Category 'o_proj': 10 tensors
  - Category 'routed_experts_down_proj': 10240 tensors
  - Category 'routed_experts_gate_proj': 10240 tensors
  - Category 'routed_experts_up_proj': 10240 tensors
  - Category 'router': 40 tensors
  - Category 'self_attn_other': 30 tensors
  - Category 'shared_expert': 160 tensors
  - Category 'ssm_linear_attn': 270 tensors
- **PASS** No NaN values found
- **PASS** No Inf values found
- **PASS** No all-zero tensors
- **PASS** No near-zero tensors
- **PASS** No abnormal constant tensors
- **WARN** Extreme outlier tensors: 1
  -   Outlier: model.language_model.layers.36.linear_attn.out_proj.weight in model-00002-of-00002.safetensors count=1
  - Non-finite details:
  - Category summary:
  -   embedding: 1 tensors, NaN=0, Inf=0, min=-0.296875, max=0.255859, L2norm=278.664, numel=508559360
  -   lm_head: 1 tensors, NaN=0, Inf=0, min=-0.300781, max=0.263672, L2norm=412.332, numel=508559360
  -   norms: 101 tensors, NaN=0, Inf=0, min=-1.00781, max=2.48438, L2norm=154.552, numel=171008
  -   o_proj: 10 tensors, NaN=0, Inf=0, min=-0.828125, max=0.613281, L2norm=143.389, numel=83886080
  -   routed_experts_down_proj: 10240 tensors, NaN=0, Inf=0, min=-0.652344, max=0.652344, L2norm=1085.9, numel=10737418240
  -   routed_experts_gate_proj: 10240 tensors, NaN=0, Inf=0, min=-0.425781, max=0.402344, L2norm=1126.61, numel=10737418240
  -   routed_experts_up_proj: 10240 tensors, NaN=0, Inf=0, min=-0.359375, max=0.285156, L2norm=1078.76, numel=10737418240
  -   router: 40 tensors, NaN=0, Inf=0, min=-0.441406, max=0.386719, L2norm=79.1712, numel=20971520
  -   self_attn_other: 30 tensors, NaN=0, Inf=0, min=-0.433594, max=0.410156, L2norm=220.712, numel=188743680
  -   shared_expert: 160 tensors, NaN=0, Inf=0, min=-0.525391, max=0.831543, L2norm=117.148, numel=125911040
  -   ssm_linear_attn: 270 tensors, NaN=0, Inf=0, min=-7.71875, max=15.5625, L2norm=568.656, numel=1011553920

## 5_config_metadata

  - architectures: ['Qwen3_5MoeForCausalLM']
- **PASS** Architecture is Qwen3_5MoeForCausalLM
  - model_type: qwen3_5_moe_text
- **PASS** model_type is qwen3_5_moe_text
  - No auto_map (using standard transformers classes)
  - dtype in config: float16
  - Actual dtypes sampled: {'torch.float16'}
- **PASS** Config dtype matches tensor dtype (float16)
- **PASS** No residual quantization_config (not bnb-quantized)
  - rope_parameters: {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}
  - max_position_embeddings: 262144
  - MoE: num_experts=256, num_experts_per_tok=8
  - Hybrid layers: 30 linear_attention, 10 full_attention
  - mamba_ssm_dtype: float32
  - Embedding shape: [248320, 2048], config vocab_size: 248320
- **PASS** Embedding vocab (248320) matches config vocab_size (248320)
  - Special tokens: bos=248044, eos=248044, pad=248044
- **PASS** bos_token_id (248044) in valid range
- **PASS** eos_token_id (248044) in valid range
- **PASS** pad_token_id (248044) in valid range
- **PASS** chat_template.jinja exists (7536 bytes)
- **PASS** chat_template.jinja parses OK
  - Template references enable_thinking parameter
- **PASS** No temp files found
- **PASS** No LoRA adapter residuals found
- **PASS** No optimizer state found

## 6_compare_original

  - Original text-only keys: 31333
  - Heretic keys: 31333
- **PASS** All original text keys present in heretic
- **PASS** No extra keys in heretic beyond original
  - Shared keys: 31333
- **WARN** Dtype mismatches: [('model.language_model.layers.17.mlp.experts.225.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.18.mlp.experts.124.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.2.mlp.experts.91.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.14.mlp.experts.109.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.1.mlp.experts.61.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.1.mlp.experts.136.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.7.mlp.experts.210.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.0.mlp.experts.73.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.23.mlp.experts.146.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.2.mlp.experts.214.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.23.mlp.experts.40.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.33.mlp.experts.9.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.32.mlp.experts.227.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.7.mlp.experts.9.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.39.mlp.experts.177.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.9.mlp.experts.27.gate_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.2.mlp.experts.30.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.29.mlp.experts.54.up_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.3.mlp.experts.48.down_proj.weight', 'torch.bfloat16', 'torch.float16'), ('model.language_model.layers.1.mlp.experts.242.up_proj.weight', 'torch.bfloat16', 'torch.float16')]
- **PASS** All shared keys have matching shapes
  - Comparing 10533 tensors numerically
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.3.self_attn.q_proj.weight (l2=1.60845e-06, cos=1.00112)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.19.self_attn.q_proj.weight (l2=1.73662e-06, cos=1.00118)
- **WARN** UNCHANGED pattern tensor CHANGED: lm_head.weight (l2=8.25056e-06, cos=1.21356)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.23.self_attn.q_proj.weight (l2=1.66131e-06, cos=1.0016)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.15.self_attn.q_proj.weight (l2=1.73801e-06, cos=1.00115)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.27.self_attn.q_proj.weight (l2=1.65453e-06, cos=1.00127)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.embed_tokens.weight (l2=1.00538e-05, cos=1.18755)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.7.self_attn.q_proj.weight (l2=1.65218e-06, cos=1.00102)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.11.self_attn.q_proj.weight (l2=1.71613e-06, cos=1.00103)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.39.self_attn.q_proj.weight (l2=1.63869e-06, cos=1.00141)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.31.self_attn.q_proj.weight (l2=1.64579e-06, cos=1.00129)
- **WARN** UNCHANGED pattern tensor CHANGED: model.language_model.layers.35.self_attn.q_proj.weight (l2=1.64971e-06, cos=1.00143)
  - Compared tensors: changed=54, unchanged=10479
- **PASS** No unexpected changes detected
- **PASS** All 231 checked unchanged tensors are truly unchanged
  - Layer distribution of changes:
  -   Layer 3: 2 tensors changed
  -   Layer 7: 2 tensors changed
  -   Layer 9: 1 tensors changed
  -   Layer 10: 1 tensors changed
  -   Layer 11: 3 tensors changed
  -   Layer 12: 1 tensors changed
  -   Layer 13: 2 tensors changed
  -   Layer 14: 1 tensors changed
  -   Layer 15: 3 tensors changed
  -   Layer 16: 1 tensors changed
  -   Layer 17: 1 tensors changed
  -   Layer 18: 1 tensors changed
  -   Layer 19: 3 tensors changed
  -   Layer 20: 1 tensors changed
  -   Layer 21: 1 tensors changed
  -   Layer 22: 1 tensors changed
  -   Layer 23: 3 tensors changed
  -   Layer 24: 1 tensors changed
  -   Layer 25: 1 tensors changed
  -   Layer 26: 1 tensors changed
  -   Layer 27: 3 tensors changed
  -   Layer 28: 1 tensors changed
  -   Layer 29: 1 tensors changed
  -   Layer 30: 1 tensors changed
  -   Layer 31: 3 tensors changed
  -   Layer 32: 1 tensors changed
  -   Layer 33: 1 tensors changed
  -   Layer 34: 1 tensors changed
  -   Layer 35: 3 tensors changed
  -   Layer 36: 1 tensors changed
  -   Layer 37: 1 tensors changed
  -   Layer 38: 1 tensors changed
  -   Layer 39: 3 tensors changed
  - Module distribution of changes:
  -   down_proj: 32 tensors changed
  -   o_proj: 10 tensors changed
  -   other: 12 tensors changed
  - Top changed tensors by L2 diff:
  -   model.language_model.layers.27.self_attn.o_proj.weight: L2=2.388, rel_L2=0.04913, max_abs=0.04944, cos_sim=0.999303
  -   model.language_model.layers.23.self_attn.o_proj.weight: L2=2.109, rel_L2=0.04563, max_abs=0.05921, cos_sim=0.999412
  -   model.language_model.layers.35.self_attn.o_proj.weight: L2=1.97, rel_L2=0.03719, max_abs=0.04541, cos_sim=0.999771
  -   model.language_model.layers.31.self_attn.o_proj.weight: L2=1.938, rel_L2=0.04018, max_abs=0.03403, cos_sim=0.999666
  -   model.language_model.layers.19.self_attn.o_proj.weight: L2=1.895, rel_L2=0.04589, max_abs=0.03505, cos_sim=0.999442
  -   model.language_model.layers.15.self_attn.o_proj.weight: L2=1.847, rel_L2=0.04441, max_abs=0.04449, cos_sim=0.999544
  -   model.language_model.layers.7.self_attn.o_proj.weight: L2=1.734, rel_L2=0.04123, max_abs=0.05066, cos_sim=0.999584
  -   model.language_model.layers.39.self_attn.o_proj.weight: L2=1.682, rel_L2=0.03443, max_abs=0.02628, cos_sim=0.999905
  -   model.language_model.layers.3.self_attn.o_proj.weight: L2=1.549, rel_L2=0.03742, max_abs=0.03641, cos_sim=0.999735
  -   model.language_model.layers.11.self_attn.o_proj.weight: L2=1.53, rel_L2=0.03802, max_abs=0.03795, cos_sim=0.999778
  -   model.language_model.layers.18.mlp.shared_expert.down_proj.weight: L2=0.5819, rel_L2=0.05356, max_abs=0.07605, cos_sim=0.998539
  -   model.language_model.layers.28.mlp.shared_expert.down_proj.weight: L2=0.5693, rel_L2=0.05198, max_abs=0.05841, cos_sim=0.998622
  -   model.language_model.layers.14.mlp.shared_expert.down_proj.weight: L2=0.5509, rel_L2=0.05139, max_abs=0.04987, cos_sim=0.998655
  -   model.language_model.layers.21.mlp.shared_expert.down_proj.weight: L2=0.5424, rel_L2=0.05156, max_abs=0.0769, cos_sim=0.998645
  -   model.language_model.layers.34.mlp.shared_expert.down_proj.weight: L2=0.538, rel_L2=0.05146, max_abs=0.07858, cos_sim=0.998648
  -   model.language_model.layers.24.mlp.shared_expert.down_proj.weight: L2=0.5357, rel_L2=0.05084, max_abs=0.09204, cos_sim=0.998684
  -   model.language_model.layers.30.mlp.shared_expert.down_proj.weight: L2=0.5262, rel_L2=0.04981, max_abs=0.06306, cos_sim=0.998731
  -   model.language_model.layers.17.mlp.shared_expert.down_proj.weight: L2=0.5215, rel_L2=0.04766, max_abs=0.0636, cos_sim=0.998836
  -   model.language_model.layers.26.mlp.shared_expert.down_proj.weight: L2=0.5144, rel_L2=0.04959, max_abs=0.05634, cos_sim=0.998746
  -   model.language_model.layers.16.mlp.shared_expert.down_proj.weight: L2=0.5139, rel_L2=0.04701, max_abs=0.04202, cos_sim=0.998868

## 7_minimal_load

- **PASS** AutoConfig loaded: Qwen3_5MoeTextConfig
- **PASS** AutoTokenizer loaded, vocab_size=248044
- **PASS** Chat template applied OK (77 chars)
  - Rendered template: <|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
<think>
...
- **WARN** Tokenizer vocab_size (248044) != config (248320)
  - Tokenizer special tokens: eos=248046, bos=None, pad=248044
  - Available RAM: 112.8 GB
- **PASS** Model loaded successfully

## _summary

  - Audit completed in 177.0 seconds
