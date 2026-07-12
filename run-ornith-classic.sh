#!/bin/bash
# Classic Heretic abliteration for Ornith-1.0-35B (not ARA-LoRA).
# Uses config.classic-ornith.toml; checkpoints go to checkpoints-classic/.
set -euo pipefail
cd "$(dirname "$0")"

export QWEN35_MOE_UNPACK_EXPERTS=1
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1

# Optional: surface bnb quant issues
# export BNB_QUANT_TRACE=1

source /home/s117/heretic-env/bin/activate

CONFIG="${HERETIC_CONFIG:-config.classic-ornith.toml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

echo "=== Classic Ornith abliteration ==="
echo "  config: $CONFIG"
echo "  QWEN35_MOE_UNPACK_EXPERTS=$QWEN35_MOE_UNPACK_EXPERTS"
echo "  TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=$TRANSFORMERS_SKIP_ALLOCATOR_WARMUP"
echo "  extra args: $*"
echo

# Heretic 1.4 loads ./config.toml from CWD. Install classic config for this
# process only; restore the previous config.toml on exit.
RESTORED=0
restore_config() {
  if [[ "$RESTORED" -eq 1 ]]; then
    return
  fi
  RESTORED=1
  if [[ -f config.toml.classic-run-backup ]]; then
    mv -f config.toml.classic-run-backup config.toml
    echo "Restored previous config.toml"
  fi
}
trap restore_config EXIT INT TERM

if [[ "$CONFIG" != "config.toml" ]]; then
  if [[ -f config.toml ]]; then
    cp -a config.toml config.toml.classic-run-backup
  fi
  cp -f "$CONFIG" config.toml
  echo "Using $CONFIG as config.toml for this run (will restore on exit)"
fi

# Do not exec: keep trap so config.toml is restored after heretic exits.
heretic "$@"
status=$?
exit "$status"
