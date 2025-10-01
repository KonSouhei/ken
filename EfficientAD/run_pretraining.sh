#!/bin/bash

# ImageNetのパス（ここを友達の環境に合わせて変更）
IMAGENET_PATH="/path/to/imagenet/train"

echo "4つの bottleneck_ratio で並行学習を開始します"
echo "-------------------------------------------"

# GPU 0で ratio=0.5
echo "GPU 0: bottleneck_ratio=0.5"
CUDA_VISIBLE_DEVICES=0 python pretraining.py \
    -m small_bottleneck \
    -d $IMAGENET_PATH \
    -o output/pretraining/bottleneck_0.5 \
    --bottleneck_ratio 0.5 &

# GPU 1で ratio=0.6667 (2/3)
echo "GPU 1: bottleneck_ratio=0.6667"
CUDA_VISIBLE_DEVICES=1 python pretraining.py \
    -m small_bottleneck \
    -d $IMAGENET_PATH \
    -o output/pretraining/bottleneck_0.6667 \
    --bottleneck_ratio 0.6667 &

# GPU 2で ratio=0.75 (3/4)
echo "GPU 2: bottleneck_ratio=0.75"
CUDA_VISIBLE_DEVICES=2 python pretraining.py \
    -m small_bottleneck \
    -d $IMAGENET_PATH \
    -o output/pretraining/bottleneck_0.75 \
    --bottleneck_ratio 0.75 &

# GPU 3で ratio=0.8
echo "GPU 3: bottleneck_ratio=0.8"
CUDA_VISIBLE_DEVICES=3 python pretraining.py \
    -m small_bottleneck \
    -d $IMAGENET_PATH \
    -o output/pretraining/bottleneck_0.8 \
    --bottleneck_ratio 0.8 &

# 全プロセスが終了するまで待つ
wait

echo "-------------------------------------------"
echo "全ての学習が完了しました！"
echo "結果:"
echo "  - output/pretraining/bottleneck_0.5/teacher_small_bottleneck_ratio0.5_final_state.pth"
echo "  - output/pretraining/bottleneck_0.6667/teacher_small_bottleneck_ratio0.6666666666666666_final_state.pth"
echo "  - output/pretraining/bottleneck_0.75/teacher_small_bottleneck_ratio0.75_final_state.pth"
echo "  - output/pretraining/bottleneck_0.8/teacher_small_bottleneck_ratio0.8_final_state.pth"
