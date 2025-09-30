import torch
import torchvision
from torchvision.models import Wide_ResNet101_2_Weights

class Teacher:
    """EfficientADのTeacher（先生）モデル"""

    def __init__(self, device='cpu'):
        self.device = device

        # WideResNet-101をロード
        print("Teacher: WideResNet-101をロード中...")
        self.backbone = torchvision.models.wide_resnet101_2(
            weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1
        ).to(device)
        print("Teacher: ロード完了")

    def extract_features(self, image):
        """layer2とlayer3の特徴を抽出"""
        with torch.no_grad():
            x = self.backbone.conv1(image)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)

            x = self.backbone.layer2(x)
            layer2_features = x

            x = self.backbone.layer3(x)
            layer3_features = x

        return layer2_features, layer3_features

    def forward(self, batch_images):
        """バッチ画像から384チャンネル特徴を出力"""
        batch_size = batch_images.shape[0]

        # 各画像で特徴抽出
        layer2_list = []
        layer3_list = []

        for i in range(batch_size):
            single_image = batch_images[i:i+1]
            layer2, layer3 = self.extract_features(single_image)
            layer2_list.append(layer2)
            layer3_list.append(layer3)

        # バッチ結合
        layer2_batch = torch.cat(layer2_list, dim=0)
        layer3_batch = torch.cat(layer3_list, dim=0)

        # layer3を64×64にアップサンプル
        layer3_upsampled = torch.nn.functional.interpolate(
            layer3_batch, size=(64, 64), mode='bilinear', align_corners=False
        )

        # チャンネル結合: 512 + 1024 = 1536
        combined_features = torch.cat([layer2_batch, layer3_upsampled], dim=1)

        # 1536 → 384に圧縮
        batch_size, channels, height, width = combined_features.shape
        flattened = combined_features.permute(0, 2, 3, 1).reshape(-1, channels)

        compressed = torch.nn.functional.adaptive_avg_pool1d(
            flattened.unsqueeze(1), 384
        ).squeeze(1)

        final_features = compressed.reshape(batch_size, height, width, 384).permute(0, 3, 1, 2)

        return final_features

# テスト
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Teacher作成
    teacher = Teacher(device)

    # テスト
    batch_images = torch.randn(2, 3, 512, 512).to(device)
    output = teacher.forward(batch_images)

    print(f"入力: {batch_images.shape}")
    print(f"Teacher出力: {output.shape}")