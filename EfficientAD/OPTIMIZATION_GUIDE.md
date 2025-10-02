# エポック数60000の最適化ガイド

## 概要

このガイドでは、事前学習のイテレーション数（60000）を削減・最適化する方法を説明します。

## 実装した機能

### 1. 詳細なloss記録
- **100イテレーションごと**にlossをCSVファイルに記録
- `training_log{ratio_suffix}.csv`に保存

### 2. loss曲線の自動プロット
- 学習終了後、自動的にloss曲線をPNG画像で保存
- `loss_curve{ratio_suffix}.png`

### 3. Early Stopping（早期停止）
- lossが改善しなくなったら自動的に学習を停止
- **デフォルトでは無効**（`--early_stopping_patience 0`）

### 4. 柔軟なイテレーション数設定
- `--epochs` 引数で簡単に変更可能
- 短期実験（5000）も長期実験（60000）も可能

## 使い方

### ステップ1: 短期実験（5000イテレーション）

まずは5000イテレーションで様子を見る：

```bash
python pretraining.py \
    -m small_bottleneck \
    -d /Users/PC_User/ken/EfficientAD/imagenet100/train \
    -o output/pretraining/pilot_5000 \
    --bottleneck_ratio 0.9 \
    --epochs 5000
```

**出力ファイル**:
- `training_log_ratio0.5.csv` - loss履歴
- `loss_curve_ratio0.5.png` - loss曲線のグラフ

### ステップ2: loss曲線を確認

`loss_curve_ratio0.5.png`を開いて確認：

#### パターンA: まだ下がり続けている
```
→ もっとイテレーションが必要
→ 次は10000や20000で試す
```

#### パターンB: 平坦になってきた
```
→ 収束が近い
→ 10000-15000くらいで十分かも
```

#### パターンC: ほぼ横ばい
```
→ 設定に問題あり
→ 学習率やモデル設定を見直す
```

### ステップ3: 段階的に実験

```bash
# 10000イテレーション
python pretraining.py --epochs 10000 --bottleneck_ratio 0.5

# 20000イテレーション
python pretraining.py --epochs 20000 --bottleneck_ratio 0.5

# 最適な値が見つかったら、その値で本番実行
```

### ステップ4: Early Stoppingを使う（オプション）

最適なイテレーション数の見当がついたら、Early Stoppingを有効化：

```bash
# 例: 最大60000イテレーション、3000イテレーション改善なしで停止
python pretraining.py \
    --epochs 60000 \
    --early_stopping_patience 3000 \
    --bottleneck_ratio 0.5
```

## コマンドライン引数

### 新規追加された引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--epochs` | 60000 | 最大イテレーション数 |
| `--log_interval` | 100 | loss記録の間隔 |
| `--early_stopping_patience` | 0 | Early stoppingの猶予（0=無効） |
| `--save_interval` | 10000 | チェックポイント保存間隔 |

### 使用例

```bash
# Early Stopping無効（デフォルト）
python pretraining.py --epochs 5000

# Early Stopping有効（3000イテレーション猶予）
python pretraining.py --epochs 60000 --early_stopping_patience 3000

# ログ記録を頻繁に（50イテレーションごと）
python pretraining.py --epochs 10000 --log_interval 50

# チェックポイントを頻繁に保存（5000イテレーションごと）
python pretraining.py --epochs 20000 --save_interval 5000
```

## 実験例：4つのbottleneck_ratioを比較

### 1. まず全てを5000イテレーションで実験

```bash
# ratio 0.5
python pretraining.py --epochs 5000 --bottleneck_ratio 0.5 \
    -o output/pretraining/pilot_0.5

# ratio 0.6667
python pretraining.py --epochs 5000 --bottleneck_ratio 0.6667 \
    -o output/pretraining/pilot_0.6667

# ratio 0.75
python pretraining.py --epochs 5000 --bottleneck_ratio 0.75 \
    -o output/pretraining/pilot_0.75

# ratio 0.8
python pretraining.py --epochs 5000 --bottleneck_ratio 0.8 \
    -o output/pretraining/pilot_0.8
```

### 2. loss曲線を比較

各フォルダの`loss_curve_*.png`を見比べて：
- どのratioが最も早く収束するか
- どのratioが最も低いlossに到達するか

### 3. 最適な設定で本番実行

```bash
# 例: ratio 0.6667が最も良かった場合
python pretraining.py \
    --epochs 60000 \
    --early_stopping_patience 3000 \
    --bottleneck_ratio 0.6667 \
    -o output/pretraining/final_0.6667
```

## loss曲線の分析方法（Python）

CSVファイルを読んで詳細分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVを読み込み
df = pd.read_csv('output/pretraining/pilot_0.5/training_log_ratio0.5.csv')

# 基本統計
print(df.describe())

# 最小lossとそのイテレーション
min_loss = df['loss'].min()
min_iter = df.loc[df['loss'].idxmin(), 'iteration']
print(f'最小loss: {min_loss:.6f} (Iteration {min_iter})')

# 収束判定（lossの変化が小さくなる点を探す）
df['loss_diff'] = df['loss'].diff().abs()
print(df[['iteration', 'loss', 'loss_diff']].tail(20))

# カスタムプロット
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df['iteration'], df['loss'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df['iteration'][1:], df['loss_diff'][1:])
plt.xlabel('Iteration')
plt.ylabel('Loss Change (abs)')
plt.title('Loss Change Rate')
plt.grid(True)
plt.tight_layout()
plt.savefig('detailed_analysis.png')
```

## 期待される効果

- **時間の節約**: 60000イテレーション全て回さず、5000や10000で収束を確認
- **最適化**: 実際に必要なイテレーション数が判明（例: 60000→25000に削減）
- **自動化**: Early Stoppingで無駄な学習を自動削減
- **可視化**: loss曲線で学習の進行状況を一目で確認

## トラブルシューティング

### lossが全然下がらない
- 学習率が低すぎる可能性
- データローダーの問題
- モデル設定の問題

### lossが振動する
- 学習率が高すぎる可能性
- バッチサイズを増やすことを検討

### Early Stoppingが早すぎる
- `--early_stopping_patience`を大きくする
- または初回はEarly Stopping無効で実験

### メモリ不足
- `--save_interval`を大きくしてチェックポイント数を減らす
- バッチサイズを小さくする（pretraining.py内の`batch_size=32`を変更）
