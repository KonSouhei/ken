import torch
import torchvision
from torchvision.models import Wide_ResNet101_2_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob

class MVTecDataset(Dataset):
    """MVTec AD正常画像データセット"""

    def __init__(self, data_path, category='bottle', transform=None):
        self.data_path = data_path
        self.category = category
        self.transform = transform

        # 正常画像パス取得
        normal_path = os.path.join(data_path, category, 'train', 'good')
        self.image_paths = glob.glob(os.path.join(normal_path, '*.png'))

        if len(self.image_paths) == 0:
            print(f"WARNING: {normal_path} に画像が見つかりません")
            # ダミーデータで代替
            self.use_dummy = True
        else:
            self.use_dummy = False
            print(f"MVTec AD {category}: {len(self.image_paths)}枚の正常画像を読み込み")

    def __len__(self):
        if self.use_dummy:
            return 50  # ダミーデータ枚数
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.use_dummy:
            # ダミーデータ（開発用）
            image = torch.randn(3, 512, 512)
        else:
            # 実際のMVTec AD画像
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

        return image

class TeacherMVTec:
    """EfficientAD Teacher（MVTec AD対応）"""

    def __init__(self, device='cpu', data_path='./mvtec_anomaly_detection', category='bottle'):
        self.device = device
        self.category = category

        # WideResNet-101をロード
        print(f"Teacher: WideResNet-101をロード中（{category}用）...")
        self.backbone = torchvision.models.wide_resnet101_2(
            weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1
        ).to(device)
        self.backbone.eval()  # 事前学習済みモデルは評価モード
        print("Teacher: ロード完了")

        # ImageNet正規化（論文通り）
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # データセット作成
        self.dataset = MVTecDataset(data_path, category, self.transform)
        print(f"データセット準備完了: {len(self.dataset)}枚")

    def extract_features_batch(self, batch_images):
        """バッチ画像からlayer2とlayer3の特徴を抽出（効率化版）"""
        with torch.no_grad():
            # WideResNet-101の順伝播
            x = self.backbone.conv1(batch_images)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)

            # layer2特徴抽出
            x = self.backbone.layer2(x)
            layer2_features = x  # (batch, 512, 64, 64)

            # layer3特徴抽出
            x = self.backbone.layer3(x)
            layer3_features = x  # (batch, 1024, 32, 32)

        return layer2_features, layer3_features

    def forward(self, batch_images):
        """バッチ画像から384チャンネル特徴を出力（論文通り）"""
        batch_size = batch_images.shape[0]

        # バッチ特徴抽出
        layer2_batch, layer3_batch = self.extract_features_batch(batch_images)

        # layer3を64×64にアップサンプル（論文通り）
        layer3_upsampled = torch.nn.functional.interpolate(
            layer3_batch, size=(64, 64), mode='bilinear', align_corners=False
        )

        # チャンネル結合: 512 + 1024 = 1536
        combined_features = torch.cat([layer2_batch, layer3_upsampled], dim=1)

        # 1536 → 384に圧縮（論文通り）
        batch_size, channels, height, width = combined_features.shape

        # 空間次元を保持したまま圧縮
        # (batch, 1536, 64, 64) → (batch, 384, 64, 64)
        flattened = combined_features.permute(0, 2, 3, 1).reshape(-1, channels)  # (batch*64*64, 1536)

        # 1536 → 384次元圧縮
        compressed = torch.nn.functional.adaptive_avg_pool1d(
            flattened.unsqueeze(1), 384
        ).squeeze(1)  # (batch*64*64, 384)

        # 元の形状に復元
        final_features = compressed.reshape(batch_size, height, width, 384).permute(0, 3, 1, 2)

        return final_features  # (batch, 384, 64, 64)

    def create_dataloader(self, batch_size=4, shuffle=True):
        """MVTec ADデータローダー作成"""
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def test_feature_extraction(self, batch_size=4):
        """特徴抽出テスト"""
        print(f"\n=== {self.category}での特徴抽出テスト ===")

        dataloader = self.create_dataloader(batch_size=batch_size)

        for batch_idx, batch_images in enumerate(dataloader):
            batch_images = batch_images.to(self.device)

            print(f"バッチ{batch_idx+1}: 入力形状 {batch_images.shape}")

            # 特徴抽出
            features = self.forward(batch_images)

            print(f"バッチ{batch_idx+1}: 出力形状 {features.shape}")
            print(f"特徴統計: 平均={torch.mean(features).item():.6f}, 標準偏差={torch.std(features).item():.6f}")

            # 1バッチのみテスト
            break

        print("特徴抽出テスト完了")

    def extract_all_features(self, batch_size=4):
        """全正常画像の特徴抽出"""
        print(f"\n=== {self.category}の全正常画像特徴抽出 ===")

        dataloader = self.create_dataloader(batch_size=batch_size, shuffle=False)
        all_features = []

        for batch_idx, batch_images in enumerate(dataloader):
            batch_images = batch_images.to(self.device)

            # 特徴抽出
            features = self.forward(batch_images)
            all_features.append(features.cpu())

            if batch_idx % 10 == 0:
                print(f"処理済み: {batch_idx * batch_size}枚")

        # 全特徴結合
        all_features = torch.cat(all_features, dim=0)
        print(f"全特徴抽出完了: {all_features.shape}")

        return all_features

# テスト
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Teacher作成（bottleカテゴリ）
    teacher = TeacherMVTec(device, category='bottle')

    # 特徴抽出テスト
    teacher.test_feature_extraction(batch_size=2)

    # 全画像特徴抽出（小規模テスト）
    # features = teacher.extract_all_features(batch_size=2)

    print("\nMVTec AD対応Teacher完成!")