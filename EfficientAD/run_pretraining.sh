#!/bin/bash

# 空いているGPUを自動選択
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | head -n 1 | cut -f 1)

echo "使用するGPU: $CUDA_VISIBLE_DEVICES"

# ImageNetのパス（ここを友達の環境に合わせて変更）
IMAGENET_PATH="/path/to/imagenet/train"

# ボトルネック圧縮率（デフォルト: 0.6667 = 2/3, 例: 0.5 = 1/2, 0.75 = 3/4）
BOTTLENECK_RATIO=0.6667

# 実行
python pretraining.py -m small_bottleneck -d $IMAGENET_PATH -o output/pretraining/bottleneck --bottleneck_ratio $BOTTLENECK_RATIO

echo "完了！"
