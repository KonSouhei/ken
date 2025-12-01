#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision.datasets import ImageFolder


class BottleneckBlock(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bottleneck_ratio=2/3):
            super().__init__()

            bottle_channel = int(in_channels * bottleneck_ratio)

            self.bottleneck = nn.Sequential(
                  nn.Conv2d(in_channels=in_channels, out_channels=bottle_channel, kernel_size=1),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(in_channels=bottle_channel, out_channels=bottle_channel, kernel_size=kernel_size, stride=stride,padding='same'),
                  nn.ReLU(inplace=True),

                  nn.Conv2d(in_channels=bottle_channel, out_channels=out_channels, kernel_size=1)
                 ) 
            
            if in_channels != out_channels:
              self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride)
            else:
              self.shortcut = nn.Identity()

      def forward(self, x):
          return self.bottleneck(x) + self.shortcut(x)
      

class DWS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        
        padding_mode = padding if isinstance(padding, str) else padding
        
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding_mode, 
            groups=in_channels 
        )
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GhostModule(nn.Module):
    """シンプルなGhostModule実装

    Primary特徴（高コスト・重要）とGhost特徴（低コスト・補助）を組み合わせる。
    """
    def __init__(self, inp, oup, kernel_size=3, ratio=2, padding=1):
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


def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        #ここでサイズ調整する56か60 56がボトルネック
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )


def get_pdn_small_dws_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(

        nn.Conv2d(3, 128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        DWS(128, 256, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        DWS(256, 256, kernel_size=3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, out_channels, kernel_size=4)
    )


def get_pdn_small_bottleneck(out_channels=384, padding=False, bottleneck_ratio=2/3):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        BottleneckBlock(128,256,4,1,3 * pad_mult, bottleneck_ratio),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        BottleneckBlock(256,256,3,1,1 * pad_mult, bottleneck_ratio),

        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )


def get_pdn_ghost_simple(out_channels=384, padding=False):
    """シンプルなGhostModule版PDN

    DWSの代わりにGhostModuleを使用。
    Primary特徴とGhost特徴を組み合わせて特徴の多様性を向上。
    """
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        # Layer 1: 初期層（DWSと同じ）
        nn.Conv2d(3, 128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        # Layer 2: GhostModule（DWSの代わり）
        GhostModule(128, 256, kernel_size=4, padding=3 * pad_mult if padding else 0),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

        # Layer 3: GhostModule（DWSの代わり）
        GhostModule(256, 256, kernel_size=3, padding=1 * pad_mult),

        # Layer 4: 出力層（DWSと同じ）
        nn.Conv2d(256, out_channels, kernel_size=4)
    )


def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)