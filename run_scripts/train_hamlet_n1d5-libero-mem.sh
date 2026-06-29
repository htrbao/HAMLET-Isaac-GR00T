#!/bin/bash
#SBATCH --job-name=groot-baseline
#SBATCH --partition=main
#SBATCH --nodelist=worker-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/mnt/data/sftp/data/baoht9/workspace/logs/%x_%3j.out
#SBATCH --error=/mnt/data/sftp/data/baoht9/workspace/logs/%x_%3j.err

source /mnt/data/sftp/data/baoht9/workspace/code/HAMLET-Isaac-GR00T/.venv/bin/activate
export WANDB_API_KEY="938a809d45af5c62586335955d205d25d0db5d04"
export WANDB_PROJECT="GR00TN1.5 MEMORY"
export HF_HOME="/mnt/data/sftp/data/baoht9/"
export PYTHONWARNINGS="ignore"

MODALITY_RIGHT_HAND_PATH="/home/baoht9/Isaac-GR00T/demo_data/modalities/baoht-libero-mem/modality.json"

DATA_CONFIG="libero_mem_hamlet"

TASK_NAME="hamlet"
DATA_PATH_1="/mnt/data/sftp/data/baoht9/workspace/data/LIBERO-MEM-lerobot"

cp $MODALITY_RIGHT_HAND_PATH $DATA_PATH_1/meta/modality.json

# HAMLET memory options
K="${K:-4}"                                      # memory window (history length); stride auto-bound to action chunk
N_MOMENT_TOKENS="${N_MOMENT_TOKENS:-4}"          # moment tokens per step (n_q)
MEM_COND_TYPE="${MEM_COND_TYPE:-cross_attn}"     # cross_attn | adaln
MEMORY_TYPE="${MEMORY_TYPE:-moment_token}"       # moment_token | vision_feature
LOAD_MOMENT_TOKENS_FROM="${LOAD_MOMENT_TOKENS_FROM:-}"  # optional Stage-1 (TCL) ckpt; see README "Moment-token initialization"
FREEZE_MOMENT_TOKENS="${FREEZE_MOMENT_TOKENS:-0}"       # 1 = freeze moment tokens (paper recipe when TCL-initialized)

MOMENT_ARGS=()
if [ "$FREEZE_MOMENT_TOKENS" = "1" ]; then MOMENT_ARGS+=(--freeze-moment-tokens); else MOMENT_ARGS+=(--no-freeze-moment-tokens); fi
[ -n "$LOAD_MOMENT_TOKENS_FROM" ] && MOMENT_ARGS+=(--load-moment-tokens-from "$LOAD_MOMENT_TOKENS_FROM")

python scripts/gr00t_finetune.py \
    --dataset-path "$DATA_PATH_1" \
    --data_config  "$DATA_CONFIG" \
    --video-backend torchvision_av \
    --num-gpus 1 \
    --output-dir /mnt/data/sftp/data/vla/vr_checkpoints/gr00tn15-$DATA_CONFIG-$TASK_NAME-$(date +"%Y-%m-%d_%H-%M-%S") \
    --report-to wandb \
    --max-steps 160000 \
    --batch-size 16 \
    --save_steps 50000 \
    --hamlet-mode finetune \
    --n-moment-tokens "$N_MOMENT_TOKENS" \
    --memory-window "$K" \
    --memory-num-layers 2 \
    --mem-cond-type "$MEM_COND_TYPE" \
    --memory-type "$MEMORY_TYPE" \
    "${MOMENT_ARGS[@]}"