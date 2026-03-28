#!/bin/bash
# Trains all 5 enhancement models sequentially on the L40S.
# Sequential so one model gets full VRAM — batch 256 runs fast anyway.
# Logs go to logs/train_<target>.log
# Run from inside the enhancement/ directory.

mkdir -p logs

TARGETS=(
    "dataset/target_clahe:model_clahe.pth"
    "dataset/target_hist:model_hist.pth"
    "dataset/target_sharpen:model_sharpen.pth"
    "dataset/target_bilateral:model_bilateral.pth"
    "dataset/target_gamma:model_gamma.pth"
)

for entry in "${TARGETS[@]}"; do

    TARGET_DIR="${entry%%:*}"
    MODEL_NAME="${entry##*:}"
    LOG_NAME="logs/train_$(basename $TARGET_DIR).log"

    echo "Training: $TARGET_DIR -> $MODEL_NAME  (log: $LOG_NAME)"

    python3 training_enhancement.py \
        --target_dir  "$TARGET_DIR" \
        --model_name  "$MODEL_NAME" \
        --epochs      50            \
        --batch_size  256           \
        --lr          0.01          \
        --num_workers 8             \
        --val_split   0.1           \
        > "$LOG_NAME" 2>&1

    echo "Done: $MODEL_NAME"

done

echo ""
echo "All 5 models trained."