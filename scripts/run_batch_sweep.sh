#!/usr/bin/env bash
set -euo pipefail

# Batch-size sweep from 1 to GPU memory limit with a few midpoints.
#
# 目的与思路（数据测试目标）：
# 1) 找到当前模型/上下文长度在本机 GPU 上可容纳的最大 batch size，作为性能/显存上限。
# 2) 在 [1, 最大值] 范围内挑选典型与中间规模（如 64/128）做短跑训练，
#    观察 loss 走势与吞吐（tokens/s），用于选择更稳妥或更高效的 batch size。
# 3) 探测阶段只跑 1 iter（MAX_ITERS_PROBE），避免长时间占用；
#    扫描阶段使用较小的 SWEEP_ITERS 快速对比。
# 4) 每个 batch size 会自动缩放学习率，并可用短跑挑选更优学习率（TUNE_LR）。
#
# Deliverables:
# - 每个 batch size 目录会生成 loss_curve.png（学习曲线）
# - lr_tune 子目录保留快速学习率搜索日志（若启用）
#
# Configurable via env vars:
#   GPU_IDS=0,1,2 SWEEP_ITERS=200 LOG_INTERVAL=10 EVAL_INTERVAL=100 MAX_ITERS_PROBE=1
#   BASE_LR=5e-4 BASE_BS=128 LR_SCALE=linear
#   TUNE_LR=1 TUNE_ITERS=20 TUNE_MULTIPLIERS=0.5,1,2
#
# 说明：
# - 会先在每张 GPU 上探测最大 batch size（逐步翻倍 + 二分），避免 OOM。
# - 取所有 GPU 的最小最大值作为 sweep 上限，保证所有 GPU 都能跑。
# - 每个 batch size 先按 BASE_LR/BASE_BS 做缩放，再用 TUNE_MULTIPLIERS 做短跑挑选更优学习率。

GPU_IDS=${GPU_IDS:-0,1,2}
SWEEP_ITERS=${SWEEP_ITERS:-200}
LOG_INTERVAL=${LOG_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
MAX_ITERS_PROBE=${MAX_ITERS_PROBE:-1}
BASE_LR=${BASE_LR:-5e-4}
BASE_BS=${BASE_BS:-128}
LR_SCALE=${LR_SCALE:-linear}
TUNE_LR=${TUNE_LR:-1}
TUNE_ITERS=${TUNE_ITERS:-20}
TUNE_MULTIPLIERS=${TUNE_MULTIPLIERS:-0.5,1,2}

PROBE_DIR_BASE=checkpoints/batch_probe
SWEEP_DIR=checkpoints/batch_sweep

mkdir -p "$SWEEP_DIR"

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"

compute_lr() {
  local batch_size=$1
  local lr
  case "$LR_SCALE" in
    linear)
      lr=$(awk -v base="$BASE_LR" -v bs="$batch_size" -v ref="$BASE_BS" 'BEGIN {printf "%.8g", base * bs / ref}')
      ;;
    sqrt)
      lr=$(awk -v base="$BASE_LR" -v bs="$batch_size" -v ref="$BASE_BS" 'BEGIN {printf "%.8g", base * sqrt(bs / ref)}')
      ;;
    *)
      lr="$BASE_LR"
      ;;
  esac
  echo "$lr"
}

try_batch() {
  local gpu_id=$1
  local batch_size=$2
  local probe_dir="$PROBE_DIR_BASE/gpu_${gpu_id}"
  local log_file="$probe_dir/loss_log.csv"
  local out_log="$probe_dir/probe_bs_${batch_size}.log"
  mkdir -p "$probe_dir"
  rm -f "$log_file"
  CUDA_VISIBLE_DEVICES=$gpu_id uv run training_together.py \
    --device cuda \
    --checkpoint-dir "$probe_dir" \
    --log-file "$log_file" \
    --batch-size "$batch_size" \
    --grad-accum-steps 1 \
    --max-iters "$MAX_ITERS_PROBE" \
    --log-interval 1 \
    --eval-interval 1000000 \
    --save-interval 1000000 \
    > "$out_log" 2>&1
}

find_max_batch() {
  local gpu_id=$1
  local bs=1
  local last_ok=1
  local upper=1

  while true; do
    if try_batch "$gpu_id" "$bs"; then
      last_ok=$bs
      upper=$((bs * 2))
      bs=$upper
    else
      upper=$bs
      break
    fi
  done

  local low=$last_ok
  local high=$upper
  while (( low + 1 < high )); do
    local mid=$(((low + high) / 2))
    if try_batch "$gpu_id" "$mid"; then
      low=$mid
    else
      high=$mid
    fi
  done

  echo "$low"
}

get_last_train_loss() {
  local csv_path=$1
  uv run python - "$csv_path" <<'PY'
import csv
import sys

path = sys.argv[1]
last = None
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        value = (row.get("train_loss") or "").strip()
        if value:
            try:
                last = float(value)
            except ValueError:
                pass

print("") if last is None else print(last)
PY
}

tune_lr_for_batch() {
  local gpu_id=$1
  local batch_size=$2
  local run_dir=$3
  local base_lr=$4
  if (( TUNE_LR == 0 )); then
    echo "$base_lr"
    return
  fi

  IFS=',' read -r -a multipliers <<< "$TUNE_MULTIPLIERS"
  local best_lr="$base_lr"
  local best_loss=""

  for mult in "${multipliers[@]}"; do
    local lr
    lr=$(awk -v base="$base_lr" -v m="$mult" 'BEGIN {printf "%.8g", base * m}')
    local tune_dir="$run_dir/lr_tune_${lr}"
    mkdir -p "$tune_dir"
    CUDA_VISIBLE_DEVICES=$gpu_id uv run training_together.py \
      --device cuda \
      --checkpoint-dir "$tune_dir" \
      --log-file "$tune_dir/loss_log.csv" \
      --batch-size "$batch_size" \
      --lr "$lr" \
      --grad-accum-steps 1 \
      --max-iters "$TUNE_ITERS" \
      --log-interval 1 \
      --eval-interval 1000000 \
      --save-interval 1000000 \
      > "$tune_dir/train.log" 2>&1
    local loss
    loss=$(get_last_train_loss "$tune_dir/loss_log.csv")
    if [[ -z "$loss" ]]; then
      continue
    fi
    if [[ -z "$best_loss" ]] || awk -v a="$loss" -v b="$best_loss" 'BEGIN {exit !(a < b)}'; then
      best_loss="$loss"
      best_lr="$lr"
    fi
  done

  echo "$best_lr"
}

add_size() {
  local value=$1
  if (( value < 1 )); then
    return
  fi
  sizes+=("$value")
}

max_batches=()
for gpu_id in "${GPU_LIST[@]}"; do
  max_for_gpu=$(find_max_batch "$gpu_id")
  max_batches+=("$max_for_gpu")
  echo "GPU ${gpu_id} max batch size (probe): ${max_for_gpu}"
done
max_batch=$(printf "%s\n" "${max_batches[@]}" | sort -n | head -n1)

sizes=()
add_size 1
add_size 4
add_size 8
add_size 16
add_size 32
add_size 64
add_size 128
add_size $((max_batch / 2))
add_size "$max_batch"

# Deduplicate and sort
readarray -t sizes < <(printf "%s\n" "${sizes[@]}" | awk '!seen[$1]++' | sort -n)

echo "Sweep max batch size (min across GPUs): $max_batch"
echo "Sweep batch sizes: ${sizes[*]}"

declare -a GPU_SIZE_LISTS
for idx in "${!GPU_LIST[@]}"; do
  GPU_SIZE_LISTS[$idx]=""
done

for i in "${!sizes[@]}"; do
  bs="${sizes[$i]}"
  if (( bs > max_batch )); then
    continue
  fi
  target_idx=$((i % ${#GPU_LIST[@]}))
  GPU_SIZE_LISTS[$target_idx]="${GPU_SIZE_LISTS[$target_idx]} $bs"
done

run_sizes_on_gpu() {
  local gpu_id=$1
  shift
  local bs_list=("$@")
  for bs in "${bs_list[@]}"; do
    if [[ -z "$bs" ]]; then
      continue
    fi
    run_dir="$SWEEP_DIR/bs_${bs}"
    mkdir -p "$run_dir"
    base_lr=$(compute_lr "$bs")
    tuned_lr=$(tune_lr_for_batch "$gpu_id" "$bs" "$run_dir" "$base_lr")
    {
      echo "batch_size=$bs"
      echo "base_lr=$base_lr"
      echo "tuned_lr=$tuned_lr"
      echo "gpu_id=$gpu_id"
      echo "lr_scale=$LR_SCALE"
      echo "tune_lr=$TUNE_LR"
      echo "tune_multipliers=$TUNE_MULTIPLIERS"
    } > "$run_dir/run_config.txt"
    echo "[GPU $gpu_id] Running batch size $bs with lr=$tuned_lr"
    CUDA_VISIBLE_DEVICES=$gpu_id uv run training_together.py \
      --device cuda \
      --checkpoint-dir "$run_dir" \
      --log-file "$run_dir/loss_log.csv" \
      --batch-size "$bs" \
      --lr "$tuned_lr" \
      --grad-accum-steps 1 \
      --max-iters "$SWEEP_ITERS" \
      --log-interval "$LOG_INTERVAL" \
      --eval-interval "$EVAL_INTERVAL" \
      --save-interval "$SAVE_INTERVAL" \
      > "$run_dir/train.log" 2>&1
    uv run scripts/plot_loss_curve.py \
      --log-file "$run_dir/loss_log.csv" \
      --out-file "$run_dir/loss_curve.png" \
      --title "Batch size ${bs} (lr=${tuned_lr})" \
      >> "$run_dir/train.log" 2>&1
    echo "[GPU $gpu_id] Finished batch size $bs"
  done
}

pids=()
for idx in "${!GPU_LIST[@]}"; do
  gpu_id="${GPU_LIST[$idx]}"
  read -r -a bs_list <<< "${GPU_SIZE_LISTS[$idx]}"
  run_sizes_on_gpu "$gpu_id" "${bs_list[@]}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

printf "Batch sweep complete. Logs under %s\n" "$SWEEP_DIR"
