#!/bin/bash
# EfficientAD異常検知訓練スクリプト（複数サブデータセット）
# bottle, cable, capsule, grid, hazelnut を順番に訓練

# 共通パラメータ
DATASET="mvtec_ad"
MODEL_SIZE="bottleneck"
BOTTLENECK_RATIO=0.4
TEACHER_WEIGHTS="models/teacher_small_bottleneck_ratio0.4_final.pth"
MVTEC_PATH="mvtec-2"
IMAGENET_PATH="archive/train"
TRAIN_STEPS=20000

# サブデータセットのリスト
SUBDATASETS=("bottle" "cable" "capsule" "grid" "hazelnut")

echo "========================================"
echo "EfficientAD Multi-Dataset Training"
echo "========================================"
echo "Model: ${MODEL_SIZE} (bottleneck_ratio=${BOTTLENECK_RATIO})"
echo "Teacher weights: ${TEACHER_WEIGHTS}"
echo "Train steps: ${TRAIN_STEPS}"
echo "Subdatasets: ${SUBDATASETS[@]}"
echo "========================================"
echo ""

# 開始時刻記録
START_TIME=$(date +%s)

# 各サブデータセットで訓練
for SUBDATASET in "${SUBDATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: ${SUBDATASET}"
    echo "========================================"

    OUTPUT_DIR="output/efficientad_result_0.4_${SUBDATASET}"

    # 訓練実行
    python efficientad.py \
        -d ${DATASET} \
        -s ${SUBDATASET} \
        -m ${MODEL_SIZE} \
        --bottleneck_ratio ${BOTTLENECK_RATIO} \
        -w ${TEACHER_WEIGHTS} \
        -a ${MVTEC_PATH} \
        -i ${IMAGENET_PATH} \
        -t ${TRAIN_STEPS} \
        -o ${OUTPUT_DIR}

    # 訓練結果チェック
    if [ $? -eq 0 ]; then
        echo "✓ ${SUBDATASET} training completed successfully"
        echo "  Results saved to: ${OUTPUT_DIR}"
    else
        echo "✗ ${SUBDATASET} training failed!"
        echo "  Stopping batch training."
        exit 1
    fi

    echo ""
done

# 終了時刻記録
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))

echo ""
echo "========================================"
echo "All trainings completed!"
echo "========================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results:"
for SUBDATASET in "${SUBDATASETS[@]}"; do
    OUTPUT_DIR="output/efficientad_result_0.9_${SUBDATASET}"
    echo "  - ${SUBDATASET}: ${OUTPUT_DIR}"
done
echo "========================================"
