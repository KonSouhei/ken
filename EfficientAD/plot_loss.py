#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import csv
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用
import matplotlib.pyplot as plt
import os

def plot_loss_curve(log_file, output_path=None):
    """CSVからLoss曲線を生成"""
    if not os.path.exists(log_file):
        print(f'Error: File not found: {log_file}')
        sys.exit(1)

    iterations = []
    losses = []

    print(f'Reading log file: {log_file}')
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            losses.append(float(row['loss']))

    if not iterations:
        print('Error: No data found in CSV file')
        sys.exit(1)

    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss Curve (up to iteration {iterations[-1]})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存パス決定
    if output_path is None:
        output_path = log_file.replace('.csv', '.png')

    plt.savefig(output_path, dpi=150)
    plt.close()

    # 統計情報表示
    print('\n=== Training Statistics ===')
    print(f'Loss curve saved: {output_path}')
    print(f'Total iterations: {iterations[-1]}')
    print(f'Initial loss: {losses[0]:.6f}')
    print(f'Final loss: {losses[-1]:.6f}')
    print(f'Loss reduction: {losses[0] - losses[-1]:.6f} ({(1 - losses[-1]/losses[0])*100:.1f}%)')
    print(f'Number of data points: {len(iterations)}')

    # 最低Loss
    min_loss = min(losses)
    min_iter = iterations[losses.index(min_loss)]
    print(f'Minimum loss: {min_loss:.6f} at iteration {min_iter}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_loss.py <path_to_training_log.csv> [output_path.png]')
        print('\nExample:')
        print('  python plot_loss.py output/pretraining/pilot_0.9_v2/training_log_ratio0.9.csv')
        sys.exit(1)

    log_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    plot_loss_curve(log_file, output_path)
