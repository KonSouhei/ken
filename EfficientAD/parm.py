#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
モデルのパラメータ数とFLOPsを計算・表示するスクリプト
"""
import torch
from torchinfo import summary
from common import get_pdn_small_bottleneck, get_pdn_small_dws_small

def print_separator():
    print("\n" + "="*80 + "\n")

def analyze_model(model, model_name, input_size=(1, 3, 256, 256)):
    """モデルの詳細情報を表示"""
    print(f"Model: {model_name}")
    print("-" * 80)

    stats = summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
        verbose=1
    )

    return stats

def main():
    print_separator()
    print("PyTorch Model Analysis - Parameters and FLOPs")
    print_separator()

    input_size = (1, 3, 256, 256)
    out_channels = 384

    # 結果を保存するリスト
    results = []

    # Bottleneck models (ratio 0.9, 0.8, 0.6, 0.4)
    bottleneck_ratios = [0.9, 0.8, 0.6, 0.4]

    for ratio in bottleneck_ratios:
        model = get_pdn_small_bottleneck(
            out_channels=out_channels,
            padding=False,
            bottleneck_ratio=ratio
        )
        model_name = f"Bottleneck (ratio={ratio})"

        stats = analyze_model(model, model_name, input_size)

        results.append({
            'name': model_name,
            'params': stats.total_params,
            'flops': stats.total_mult_adds
        })

        print_separator()

    # DWS model
    model_dws = get_pdn_small_dws_small(out_channels=out_channels, padding=False)
    model_name = "DWS (Depthwise Separable)"

    stats_dws = analyze_model(model_dws, model_name, input_size)

    results.append({
        'name': model_name,
        'params': stats_dws.total_params,
        'flops': stats_dws.total_mult_adds
    })

    print_separator()

    # 比較表を出力
    print("COMPARISON TABLE")
    print("-" * 80)
    print(f"{'Model':<30} {'Parameters':>15} {'FLOPs (MACs)':>20} {'Relative'}")
    print("-" * 80)

    # DWSを基準とする
    base_flops = results[-1]['flops']

    for result in results:
        params_str = f"{result['params']:,}"
        flops_str = f"{result['flops']:,}"
        relative = (result['flops'] / base_flops) * 100

        print(f"{result['name']:<30} {params_str:>15} {flops_str:>20} {relative:>7.1f}%")

    print("-" * 80)
    print(f"\nBase model (100%): {results[-1]['name']}")
    print(f"Input size: 256×256×3")
    print(f"Output channels: {out_channels}")
    print_separator()

if __name__ == '__main__':
    main()
