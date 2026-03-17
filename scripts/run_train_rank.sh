#!/bin/bash
# 排序模型训练示例
# 用法: bash scripts/run_train_rank.sh

set -e

DATA_ROOT="/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset"

# ---------- 示例1: MMOE + TorchJD（默认） ----------
python -m scripts.train_rank \
    --train_path "${DATA_ROOT}/datasetsali_ccp_train.parquet" \
    --val_path   "${DATA_ROOT}/datasetsali_ccp_val.parquet" \
    --test_path  "${DATA_ROOT}/datasetsali_ccp_test.parquet" \
    --model mmoe_jd \
    --embedding_dim 32 \
    --num_experts 6 \
    --expert_hidden_dims 256 128 \
    --tower_hidden_dims 64 \
    --dropout 0.2 \
    --learning_rate 1e-3 \
    --aggregation_method upgrad \
    --batch_size 256 \
    --num_workers 4 \
    --max_epochs 100 \
    --accelerator gpu \
    --devices 4 \
    --val_check_interval 10000 \
    --limit_val_batches 100 \
    --early_stop_patience 5 \
    --exp_dir experiments/rank

# ---------- 示例2: 普通 MMOE（加权 loss） ----------
 python -m scripts.train_rank \
     --train_path "${DATA_ROOT}/datasetsali_ccp_train.parquet" \
     --val_path   "${DATA_ROOT}/datasetsali_ccp_val.parquet" \
     --test_path  "${DATA_ROOT}/datasetsali_ccp_test.parquet" \
     --model mmoe \
     --num_experts 6 \
     --ctr_weight 0.5 \
     --cvr_weight 0.5 \
     --exp_dir experiments/rank

# ---------- 示例3: MOE + TorchJD ----------
python -m scripts.train_rank \
     --model moe_jd \
     --num_experts 4 \
     --aggregation_method mgda \
     --exp_dir experiments/rank

# ---------- 示例4: Share Bottom ----------
 python -m scripts.train_rank \
     --model share_bottom \
     --shared_hidden_dims 256 128 \
     --ctr_weight 0.5 \
     --cvr_weight 0.5 \
     --exp_dir experiments/rank
