#!/bin/bash

# 空いているGPUを自動選択
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | head -n 1 | cut -f 1)

echo "使用するGPU: $CUDA_VISIBLE_DEVICES"

# ImageNetのパス（ここを友達の環境に合わせて変更）
IMAGENET_PATH="/path/to/imagenet/train"

# 実行
python pretraining.py -m small_bottleneck -d $IMAGENET_PATH -o output/pretraining/bottleneck

echo "完了！"
