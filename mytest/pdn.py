import torch
import torch.nn as nn

class PDN(nn.Module):
  
    def __init__(self, out_channels=384):
        super(PDN, self).__init__()

        # PDN Small構造
        self.layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),

            # Layer 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),

            # Layer 3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
        )

    def forward(self, x):
        """画像を384チャンネル特徴に変換"""
        return self.layers(x)

# テスト
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # PDN作成
    pdn = PDN(out_channels=384).to(device)

    print("PDN構造:")
    print(pdn)

    # テスト
    batch_images = torch.randn(2, 3, 256, 256).to(device)  # PDNは256x256入力
    output = pdn(batch_images)

    print(f"\n入力: {batch_images.shape}")
    print(f"PDN出力: {output.shape}")

    # パラメータ数確認
    total_params = sum(p.numel() for p in pdn.parameters())
    print(f"パラメータ数: {total_params:,}")