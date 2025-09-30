import torch
import torch.nn as nn

class PDNDebug(nn.Module):
    """PDNの各レイヤーでの形状変化を詳しく確認"""

    def __init__(self, out_channels=384):
        super(PDNDebug, self).__init__()

        # 各レイヤーを個別に定義
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)

    def forward(self, x):
        print(f"入力: {x.shape}")

        # Layer 1
        x = self.conv1(x)
        print(f"Conv1後: {x.shape}")
        x = self.relu1(x)
        x = self.pool1(x)
        print(f"Pool1後: {x.shape}")

        # Layer 2
        x = self.conv2(x)
        print(f"Conv2後: {x.shape}")
        x = self.relu2(x)
        x = self.pool2(x)
        print(f"Pool2後: {x.shape}")

        # Layer 3
        x = self.conv3(x)
        print(f"Conv3後: {x.shape}")
        x = self.relu3(x)

        # Layer 4
        x = self.conv4(x)
        print(f"Conv4後（最終）: {x.shape}")

        return x

# テスト
if __name__ == "__main__":
    device = 'cpu'
    print("=== PDN形状変化の詳細確認 ===\n")

    # PDN作成
    pdn = PDNDebug(out_channels=384).to(device)

    # テスト
    test_input = torch.randn(1, 3, 256, 256).to(device)
    output = pdn(test_input)

    print(f"\n最終結果: {test_input.shape} → {output.shape}")