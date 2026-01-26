#!/usr/bin/env bash
set -euo pipefail

# Launch three training runs on GPUs 0/1/2 with different optimizer settings.
#
# 目的：并行比较三组不同学习率/优化器超参（betas、weight decay），
# 观察收敛速度、loss 波动与稳定性，帮助挑选更优配置。
#
# 输出说明：
# - 每组输出到独立 checkpoint 目录（tinystories_exp1/2/3）
# - loss 曲线 CSV 写入对应目录的 loss_log.csv
# - 训练日志重定向到 train.log
#
# 如需调整其他超参（如 --batch-size、--context-length），直接在下方命令中添加即可。

mkdir -p checkpoints/tinystories_exp1 checkpoints/tinystories_exp2 checkpoints/tinystories_exp3

# 实验 1：基准学习率 + 默认 betas
CUDA_VISIBLE_DEVICES=0 uv run training_together.py \
  --device cuda \
  --checkpoint-dir checkpoints/tinystories_exp1 \
  --log-file checkpoints/tinystories_exp1/loss_log.csv \
  --lr 5e-4 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight-decay 0.01 \
  --seed 1337 \
  > checkpoints/tinystories_exp1/train.log 2>&1 &

# 实验 2：降低学习率 + 更高 beta2 + 更强 weight decay
CUDA_VISIBLE_DEVICES=1 uv run training_together.py \
  --device cuda \
  --checkpoint-dir checkpoints/tinystories_exp2 \
  --log-file checkpoints/tinystories_exp2/loss_log.csv \
  --lr 3e-4 \
  --beta1 0.9 \
  --beta2 0.98 \
  --weight-decay 0.02 \
  --seed 1337 \
  > checkpoints/tinystories_exp2/train.log 2>&1 &

# 实验 3：提高学习率 + 更低 beta1 + 更小 weight decay
CUDA_VISIBLE_DEVICES=2 uv run training_together.py \
  --device cuda \
  --checkpoint-dir checkpoints/tinystories_exp3 \
  --log-file checkpoints/tinystories_exp3/loss_log.csv \
  --lr 7e-4 \
  --beta1 0.85 \
  --beta2 0.95 \
  --weight-decay 0.005 \
  --seed 1337 \
  > checkpoints/tinystories_exp3/train.log 2>&1 &

wait

echo "All three runs finished."