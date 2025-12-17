# GhostNet Teacher事前学習スクリプト
# ImageNetデータでGhostNetベースのteacherモデルを事前学習

# パラメータ
$MODEL_SIZE = "ghostnet"
$DATA_PATH = "/archive/train"
$VAL_PATH = "archive/val.X"
$OUTPUT_DIR = "output/pretraining/ghostnet"
$EPOCHS = 5000
$SAVE_INTERVAL = 1000
$LOG_INTERVAL = 100
$VAL_INTERVAL = 500

Write-Host "========================================"
Write-Host "GhostNet Teacher Pre-training"
Write-Host "========================================"
Write-Host "Model: $MODEL_SIZE"
Write-Host "Training data: $DATA_PATH"
Write-Host "Validation data: $VAL_PATH"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host "Epochs: $EPOCHS"
Write-Host "========================================"
Write-Host ""

# 事前学習実行
python pretraining.py `
    -m $MODEL_SIZE `
    -d $DATA_PATH `
    --val_path $VAL_PATH `
    -o $OUTPUT_DIR `
    --epochs $EPOCHS `
    --save_interval $SAVE_INTERVAL `
    --log_interval $LOG_INTERVAL `
    --val_interval $VAL_INTERVAL

# 結果チェック
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "✓ Pre-training completed successfully" -ForegroundColor Green
    Write-Host "========================================"
    Write-Host ""
    Write-Host "Teacher weights saved to:"
    Write-Host "  $OUTPUT_DIR/teacher_${MODEL_SIZE}_final_state.pth"
    Write-Host ""
    Write-Host "To use this model for anomaly detection training, run:"
    Write-Host "  .\run_efficientad_ghost_test.ps1"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "✗ Pre-training failed!" -ForegroundColor Red
    exit 1
}
