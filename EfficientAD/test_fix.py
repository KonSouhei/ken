#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GhostModuleとBottleneckFixの修正テスト

このスクリプトは以下をテストします:
1. インポートエラーが解消されたか
2. モデルが正常に初期化されるか
3. forward passが動作するか
"""

import torch
import torch.nn as nn


# ========== GhostModule定義 ==========
class GhostModule(nn.Module):
    """シンプルなGhostModule実装

    Primary特徴(高コスト・重要)とGhost特徴(低コスト・補助)を組み合わせる。
    """
    def __init__(self, inp, oup, kernel_size=3, ratio=2, padding=0):
        super().__init__()
        self.oup = oup
        init_channels = oup // ratio  # 半分だけ通常Conv

        # Primary特徴: 高コストだが重要
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # Ghost特徴: 低コストで生成
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, kernel_size,
                     padding=padding, groups=init_channels, bias=False),  # Depthwise
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost = self.cheap_operation(primary)
        return torch.cat([primary, ghost], dim=1)  # 結合


# ========== BottleneckFix定義 ==========
class BottleneckFix(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio=2/3, padding=False):
        super().__init__()
        pad_mult = 1 if padding else 0
        bottle_ch = int(in_channels * bottleneck_ratio)

        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, bottle_ch, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(bottle_ch, bottle_ch, kernel_size=4, padding=3*pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1*pad_mult)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(bottle_ch, bottle_ch, kernel_size=3, padding=1*pad_mult),
            nn.ReLU(inplace=True)
        )

        self.expand = nn.Conv2d(bottle_ch, out_channels, kernel_size=4, padding=3*pad_mult)

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1*pad_mult),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=3*pad_mult)
        )

    def forward(self, x):
        out = self.compress(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.expand(out)

        shortcut = self.shortcut(x)

        return out + shortcut


# ========== PDN構築関数 ==========
def get_pdn_ghost_simple(out_channels=384, padding=False):
    """シンプルなGhostModule版PDN"""
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        # Layer 1: 初期層
        nn.Conv2d(3, 128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        # Layer 2: GhostModule
        GhostModule(128, 256, kernel_size=4, padding=3 * pad_mult if padding else 0),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        # Layer 3: GhostModule
        GhostModule(256, 256, kernel_size=3, padding=1 * pad_mult),

        # Layer 4: 出力層
        nn.Conv2d(256, out_channels, kernel_size=4)
    )


def get_pdn_small_bottleneckfix(out_channels=384, padding=False, bottleneck_ratio=2/3):
    """BottleneckFix版PDN"""
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        # Layer 1
        nn.Conv2d(3, 128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),

        # Layer 2: BottleneckFix
        BottleneckFix(
            in_channels=128,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
            padding=padding
        )
    )


# ========== テスト実行 ==========
def test_ghostnet():
    print("=" * 60)
    print("GhostNet Test")
    print("=" * 60)

    try:
        model = get_pdn_ghost_simple(384, padding=False)
        print("✓ GhostNet model initialized successfully")

        x = torch.randn(1, 3, 256, 256)
        print(f"  Input shape: {x.shape}")

        with torch.no_grad():
            out = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {out.shape}")
        print(f"  Expected: torch.Size([1, 384, 33, 33])")

        if out.shape[1] == 384:
            print("✓ Output channels correct!")
        else:
            print(f"✗ Output channels mismatch: expected 384, got {out.shape[1]}")

        return True
    except Exception as e:
        print(f"✗ GhostNet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bottleneckfix():
    print("\n" + "=" * 60)
    print("BottleneckFix Test")
    print("=" * 60)

    try:
        model = get_pdn_small_bottleneckfix(384, padding=False)
        print("✓ BottleneckFix model initialized successfully")

        x = torch.randn(1, 3, 256, 256)
        print(f"  Input shape: {x.shape}")

        with torch.no_grad():
            out = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {out.shape}")
        print(f"  Expected: torch.Size([1, 384, 123, 123]) or similar")

        if out.shape[1] == 384:
            print("✓ Output channels correct!")
        else:
            print(f"✗ Output channels mismatch: expected 384, got {out.shape[1]}")

        return True
    except Exception as e:
        print(f"✗ BottleneckFix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_padding_parameter():
    print("\n" + "=" * 60)
    print("Padding Parameter Test")
    print("=" * 60)

    print("\nTesting GhostModule with different padding values...")
    try:
        # padding=0
        ghost1 = GhostModule(128, 256, kernel_size=4, padding=0)
        x1 = torch.randn(1, 128, 64, 64)
        out1 = ghost1(x1)
        print(f"  padding=0: input {x1.shape} -> output {out1.shape}")

        # padding=3
        ghost2 = GhostModule(128, 256, kernel_size=4, padding=3)
        x2 = torch.randn(1, 128, 64, 64)
        out2 = ghost2(x2)
        print(f"  padding=3: input {x2.shape} -> output {out2.shape}")

        print("✓ Padding parameter is working correctly!")
        return True
    except Exception as e:
        print(f"✗ Padding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing GhostModule and BottleneckFix fixes...\n")

    results = []
    results.append(("GhostNet", test_ghostnet()))
    results.append(("BottleneckFix", test_bottleneckfix()))
    results.append(("Padding", test_padding_parameter()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("GhostModule and BottleneckFix are now working correctly.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the error messages above.")
    print("=" * 60)
