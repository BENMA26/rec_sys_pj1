#!/bin/bash
# 双塔召回模型训练示例
# 用法: bash scripts/run_train_recall.sh

set -e

DATA_ROOT="/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset"

# ---------- 示例1: Listwise (in-batch + easy neg，默认) ----------
python -m scripts.train_recall \
    --train_path "${DATA_ROOT}/datasetsali_ccp_train.parquet" \
    --val_path   "${DATA_ROOT}/datasetsali_ccp_val.parquet" \
    --test_path  "${DATA_ROOT}/datasetsali_ccp_test.parquet" \
    --embed_dim 16 \
    --hidden_dims 128 32 \
    --dropout 0.2 \
    --temperature 0.05 \
    --learning_rate 1e-3 \
    --training_mode listwise \
    --batch_size 1024 \
    --num_workers 8 \
    --num_easy_neg 8 \
    --use_inbatch_neg true \
    --user_col 101 \
    --item_col 205 \
    --max_epochs 10 \
    --accelerator gpu \
    --devices 4 \
    --val_check_interval 10 \
    --limit_val_batches 100 \
    --gradient_clip_val 1.0 \
    --early_stop_patience 5 \
    --exp_dir experiments/two_tower_listwise

# ---------- 示例2: Listwise (仅 in-batch 负样本) ----------
# python -m scripts.train_recall \
#     --training_mode listwise \
#     --num_easy_neg 0 \
#     --use_inbatch_neg true \
#     --exp_dir experiments/two_tower_listwise_inbatch_only

# ---------- 示例3: Listwise (仅 easy 负样本) ----------
# python -m scripts.train_recall \
#     --training_mode listwise \
#     --num_easy_neg 16 \
#     --use_inbatch_neg false \
#     --exp_dir experiments/two_tower_listwise_easy_only

# ---------- 示例4: Pointwise ----------
# python -m scripts.train_recall \
#     --training_mode pointwise \
#     --num_easy_neg 8 \
#     --exp_dir experiments/two_tower_pointwise

# ---------- 示例5: Pairwise (BPR-like) ----------
# python -m scripts.train_recall \
#     --training_mode pairwise \
#     --num_easy_neg 8 \
#     --margin 0.5 \
#     --exp_dir experiments/two_tower_pairwise
