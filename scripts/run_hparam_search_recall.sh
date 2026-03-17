#!/bin/bash
#SBATCH --job-name=recall_hparam
#SBATCH --output=/work/home/maben/project/rec_sys/projects/rec_sys_pj1/rec_sys_pj1/experiments/hparam_search/logs/hparam_search.out
#SBATCH --error=/work/home/maben/project/rec_sys/projects/rec_sys_pj1/rec_sys_pj1/experiments/hparam_search/logs/hparam_search.err
#SBATCH --partition=8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=60G
#SBATCH --time=144:00:00

source ~/.bashrc
conda activate cf-design

cd /work/home/maben/project/rec_sys/projects/rec_sys_pj1/rec_sys_pj1

DATA_ROOT="/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset"
EXP_ROOT="experiments/hparam_search"
mkdir -p "${EXP_ROOT}/logs"

echo "============================================================"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES}"
echo "  Date:   $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

# ============================================================
# 超参数搜索空间
# ============================================================
EMBED_DIMS=(16 32)
HIDDEN_DIMS_LIST=("256 64" "512 128 32")
TEMPERATURES=(0.05 0.1)
NUM_EASY_NEGS=(64 512)
LEARNING_RATES=(1e-3 5e-4)

BEST_RECALL=0
BEST_EXP_DIR=""
BEST_CKPT=""

# ============================================================
# 遍历所有超参数组合
# ============================================================
for EMBED_DIM in "${EMBED_DIMS[@]}"; do
for HIDDEN_DIMS in "${HIDDEN_DIMS_LIST[@]}"; do
for TEMPERATURE in "${TEMPERATURES[@]}"; do
for NUM_EASY_NEG in "${NUM_EASY_NEGS[@]}"; do
for LR in "${LEARNING_RATES[@]}"; do

    # 构造实验名称（去掉空格）
    HIDDEN_TAG=$(echo "${HIDDEN_DIMS}" | tr ' ' '_')
    EXP_NAME="emb${EMBED_DIM}_hid${HIDDEN_TAG}_t${TEMPERATURE}_neg${NUM_EASY_NEG}_lr${LR}"
    EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
    LOG_FILE="${EXP_ROOT}/logs/${EXP_NAME}.out"

    echo ""
    echo "------------------------------------------------------------"
    echo "  实验: ${EXP_NAME}"
    echo "  embed_dim=${EMBED_DIM}, hidden_dims=${HIDDEN_DIMS}"
    echo "  temperature=${TEMPERATURE}, num_easy_neg=${NUM_EASY_NEG}, lr=${LR}"
    echo "------------------------------------------------------------"

    # ---- 训练 ----
    python -u scripts/train_recall.py \
        --train_path "${DATA_ROOT}/datasetsali_ccp_train.parquet" \
        --val_path   "${DATA_ROOT}/datasetsali_ccp_val.parquet" \
        --test_path  "${DATA_ROOT}/datasetsali_ccp_test.parquet" \
        --embed_dim  ${EMBED_DIM} \
        --hidden_dims ${HIDDEN_DIMS} \
        --dropout 0.2 \
        --temperature ${TEMPERATURE} \
        --learning_rate ${LR} \
        --training_mode listwise \
        --batch_size 1024 \
        --num_workers 8 \
        --num_easy_neg ${NUM_EASY_NEG} \
        --use_inbatch_neg true \
        --user_col 101 \
        --item_col 205 \
        --max_epochs 10 \
        --accelerator gpu \
        --devices 4 \
        --gradient_clip_val 1.0 \
        --early_stop_patience 5 \
        --exp_dir "${EXP_DIR}" \
        2>&1 | tee "${LOG_FILE}"

    # ---- 找最佳 checkpoint ----
    CKPT=$(ls -t "${EXP_DIR}/checkpoints/"*.ckpt 2>/dev/null | grep -v last | head -1)
    if [ -z "${CKPT}" ]; then
        echo "  [警告] 未找到 checkpoint，跳过测试"
        continue
    fi

    # ---- 测试（全量物料 Recall@K）----
    TEST_LOG="${EXP_ROOT}/logs/${EXP_NAME}_test.out"
    echo "  测试 checkpoint: ${CKPT}"
    python -u scripts/train_recall.py \
        --do_test \
        --test_path  "${DATA_ROOT}/datasetsali_ccp_test.parquet" \
        --checkpoint_path "${CKPT}" \
        --embed_dim  ${EMBED_DIM} \
        --hidden_dims ${HIDDEN_DIMS} \
        --batch_size 1024 \
        --num_workers 8 \
        --user_col 101 \
        --item_col 205 \
        --topk_list 50 100 \
        2>&1 | tee "${TEST_LOG}"

    # ---- 提取 test_recall@50 用于比较 ----
    RECALL=$(grep "test_recall@50" "${TEST_LOG}" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    RECALL=${RECALL:-0}
    echo "  test_recall@50 = ${RECALL}"

    # 记录到汇总文件
    echo "${EXP_NAME}  recall@50=${RECALL}  ckpt=${CKPT}" >> "${EXP_ROOT}/results_summary.txt"

    # 更新最优
    if (( $(echo "${RECALL} > ${BEST_RECALL}" | bc -l) )); then
        BEST_RECALL=${RECALL}
        BEST_EXP_DIR=${EXP_DIR}
        BEST_CKPT=${CKPT}
    fi

done
done
done
done
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================================"
echo "  超参数搜索完成！$(date)"
echo "  最佳 test_recall@50 = ${BEST_RECALL}"
echo "  最佳实验目录: ${BEST_EXP_DIR}"
echo "  最佳 checkpoint: ${BEST_CKPT}"
echo "============================================================"

# 打印完整排名
echo ""
echo "所有实验结果（按 recall@50 排序）："
sort -t= -k2 -rn "${EXP_ROOT}/results_summary.txt"
