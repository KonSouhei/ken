# EfficientAD GhostNetテスト訓練スクリプト
# 低精度カテゴリでGhostNetモデルをテスト

# 共通パラメータ
$DATASET = "mvtec_ad"
$MODEL_SIZE = "ghostnet"
$TEACHER_WEIGHTS = "output/pretraining/ghostnet/teacher_ghostnet_final_state.pth"
$MVTEC_PATH = "mvtec-2"
$IMAGENET_PATH = "archive/train"
$TRAIN_STEPS = 20000

# 低精度カテゴリ（5つ）
$SUBDATASETS = @("screw", "capsule", "zipper", "cable", "hazelnut")

Write-Host "========================================"
Write-Host "EfficientAD GhostNet Test Training"
Write-Host "========================================"
Write-Host "Model: $MODEL_SIZE"
Write-Host "Teacher weights: $TEACHER_WEIGHTS"
Write-Host "Train steps: $TRAIN_STEPS"
Write-Host "Test categories: $($SUBDATASETS -join ', ')"
Write-Host "========================================"
Write-Host ""

# Teacher weightsの存在確認
if (-not (Test-Path $TEACHER_WEIGHTS)) {
    Write-Host "Warning: Teacher weights not found at: $TEACHER_WEIGHTS" -ForegroundColor Yellow
    Write-Host "Please run .\run_ghostnet_pretraining.ps1 first to create teacher weights." -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue without pre-trained weights? (y/N)"
    if ($response -ne "y" -and $response -ne "Y") {
        Write-Host "Aborted." -ForegroundColor Red
        exit 1
    }
}

# 開始時刻記録
$START_TIME = Get-Date

# 各サブデータセットで訓練
foreach ($SUBDATASET in $SUBDATASETS) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Training: $SUBDATASET"
    Write-Host "========================================"

    $OUTPUT_DIR = "output/ghostnet_test"

    # 訓練実行
    python efficientad.py `
        -d $DATASET `
        -s $SUBDATASET `
        -m $MODEL_SIZE `
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
    $OUTPUT_DIR = "output/ghostnet_test"
    Write-Host "  - ${SUBDATASET}: $OUTPUT_DIR"
}
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Compare GhostNet results with DWS baseline (output/dws15)"
Write-Host "2. Analyze AUC scores for each category"
Write-Host "3. Check if screw performance improved from 59.81%"
Write-Host "========================================"
