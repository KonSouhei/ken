# EfficientAD異常検知訓練スクリプト（複数サブデータセット）
# bottle, cable, capsule, grid, hazelnut を順番に訓練

# 共通パラメータ
$DATASET = "mvtec_ad"
$MODEL_SIZE = "bottleneck"
$BOTTLENECK_RATIO = 0.6
$TEACHER_WEIGHTS = "models/teacher_small_bottleneck_ratio0.6_final_state.pth"
$MVTEC_PATH = "mvtec-2"
$IMAGENET_PATH = "archive/train"
$TRAIN_STEPS = 20000

# サブデータセットのリスト
$SUBDATASETS = @("bottle", "cable", "capsule", "grid", "hazelnut")

Write-Host "========================================"
Write-Host "EfficientAD Multi-Dataset Training"
Write-Host "========================================"
Write-Host "Model: $MODEL_SIZE (bottleneck_ratio=$BOTTLENECK_RATIO)"
Write-Host "Teacher weights: $TEACHER_WEIGHTS"
Write-Host "Train steps: $TRAIN_STEPS"
Write-Host "Subdatasets: $($SUBDATASETS -join ', ')"
Write-Host "========================================"
Write-Host ""

# 開始時刻記録
$START_TIME = Get-Date

# 各サブデータセットで訓練
foreach ($SUBDATASET in $SUBDATASETS) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Training: $SUBDATASET"
    Write-Host "========================================"

    $OUTPUT_DIR = "output/efficientad_result_0.80_$SUBDATASET"

    # 訓練実行
    python efficientad.py `
        -d $DATASET `
        -s $SUBDATASET `
        -m $MODEL_SIZE `
        --bottleneck_ratio $BOTTLENECK_RATIO `
        -w $TEACHER_WEIGHTS `
        -a $MVTEC_PATH `
        -i $IMAGENET_PATH `
        -t $TRAIN_STEPS `
        -o $OUTPUT_DIR

    # 訓練結果チェック
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $SUBDATASET training completed successfully" -ForegroundColor Green
        Write-Host "  Results saved to: $OUTPUT_DIR"
    } else {
        Write-Host "✗ $SUBDATASET training failed!" -ForegroundColor Red
        Write-Host "  Stopping batch training."
        exit 1
    }

    Write-Host ""
}

# 終了時刻記録
$END_TIME = Get-Date
$ELAPSED_TIME = $END_TIME - $START_TIME
$HOURS = [math]::Floor($ELAPSED_TIME.TotalHours)
$MINUTES = $ELAPSED_TIME.Minutes

Write-Host ""
Write-Host "========================================"
Write-Host "All trainings completed!"
Write-Host "========================================"
Write-Host "Total time: ${HOURS}h ${MINUTES}m"
Write-Host ""
Write-Host "Results:"
foreach ($SUBDATASET in $SUBDATASETS) {
    $OUTPUT_DIR = "output/efficientad_result_0.9_$SUBDATASET"
    Write-Host "  - ${SUBDATASET}: $OUTPUT_DIR"
}
Write-Host "========================================"
