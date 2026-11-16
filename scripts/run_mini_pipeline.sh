#!/usr/bin/env bash
# Run manually: bash scripts/run_mini_pipeline.sh /tmp/chen-minimind-mini 1
set -euo pipefail
umask 077
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${1:?Usage: bash scripts/run_mini_pipeline.sh /tmp/your-data-dir GPU_INDEX}
GPU_INDEX=${2:?Specify the physical GPU index after checking availability}
PYTHON=${MINIMIND_PYTHON:-/work2/chen/miniconda3/envs/mimi/bin/python}
case "$DATA_ROOT" in /tmp/*) ;; *) echo 'Data directory must be under /tmp' >&2; exit 1;; esac
if [[ -e "$DATA_ROOT" && ( ! -O "$DATA_ROOT" || -L "$DATA_ROOT" ) ]]; then
    echo 'Refusing a data directory owned by another user or a symbolic link' >&2; exit 1
fi
mkdir -p "$DATA_ROOT" "$PROJECT/logs"
export HF_HOME="$DATA_ROOT/hf"
export HF_DATASETS_CACHE="$DATA_ROOT/hf/datasets"
export XDG_CACHE_HOME="$DATA_ROOT/cache"
export TMPDIR="$DATA_ROOT/tmp"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
# Prevent duplicate runs using the same output prefixes.
exec 9>"$PROJECT/logs/mini_pipeline.lock"
flock -n 9 || { echo 'Mini pipeline already running'; exit 1; }
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: refusing CPU fallback"; print(torch.cuda.get_device_name(0), flush=True)'
BASE=https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master
for name in pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl; do
    if [[ ! -f "$DATA_ROOT/$name" ]]; then
        curl --fail --location --retry 5 --connect-timeout 30 --continue-at - \
            "$BASE/$name" --output "$DATA_ROOT/$name.part"
        mv -- "$DATA_ROOT/$name.part" "$DATA_ROOT/$name"
    fi
done
# Verify JSON and required fields before committing GPU time.
"$PYTHON" - "$DATA_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, key in [('pretrain_t2t_mini.jsonl', 'text'), ('sft_t2t_mini.jsonl', 'conversations')]:
    count = 0
    with (root / name).open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            assert key in row, (name, count, key)
            count += 1
    assert count > 0, f'Empty dataset: {name}'
    print(f'{name}: {count} records', flush=True)
PY
cd "$PROJECT/trainer"
# Keep the official 768-wide, 8-layer model and two epochs per stage.
# Smaller microbatches limit memory; accumulation preserves effective batch sizes.
if [[ ! -f ../logs/mini_pretrain.done ]]; then
    "$PYTHON" -u train_pretrain.py --device cuda:0 --epochs 2 \
        --hidden_size 768 --num_hidden_layers 8 --batch_size 8 --accumulation_steps 32 \
        --num_workers 2 --max_seq_len 340 --log_interval 10 --save_interval 500 \
        --data_path "$DATA_ROOT/pretrain_t2t_mini.jsonl" \
        --save_weight mini_pretrain --from_weight none --from_resume 1 \
        2>&1 | tee -a ../logs/mini_pretrain.log
    touch ../logs/mini_pretrain.done
fi
test -s ../out/mini_pretrain_768.pth
if [[ ! -f ../logs/mini_sft.done ]]; then
    "$PYTHON" -u train_full_sft.py --device cuda:0 --epochs 2 \
        --hidden_size 768 --num_hidden_layers 8 --batch_size 4 --accumulation_steps 4 \
        --num_workers 2 --max_seq_len 768 --log_interval 10 --save_interval 500 \
        --data_path "$DATA_ROOT/sft_t2t_mini.jsonl" \
        --save_weight mini_full_sft --from_weight mini_pretrain --from_resume 1 \
        2>&1 | tee -a ../logs/mini_sft.log
    touch ../logs/mini_sft.done
fi
echo 'Pretrain and SFT completed. Weights: out/mini_pretrain_768.pth and out/mini_full_sft_768.pth'
