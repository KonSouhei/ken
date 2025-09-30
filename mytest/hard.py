import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm

# 既存のクラスをインポート
from pdn import PDN

class InfiniteDataloader:
    """論文通りの無限データローダー"""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # データローダーが終了したら再開
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

class HardFeatureLossTrainer:
    """PDN同士のHard Feature Loss学習クラス（EfficientAD正しい実装）"""

    def __init__(self, device='cpu', data_path='./mvtec_anomaly_detection', category='bottle'):
        self.device = device
        self.category = category

        print(f"PDN同士のHard Feature Loss学習システム初期化中（{category}）...")

        # Teacher PDN（学習済み、固定）
        self.teacher_pdn = PDN(out_channels=384).to(device)
        try:
            self.teacher_pdn.load_state_dict(torch.load(f'trained_pdn_{category}.pth', map_location=device))
            print(f"学習済みPDN（Teacher）をロードしました: trained_pdn_{category}.pth")
        except FileNotFoundError:
            print(f"WARNING: trained_pdn_{category}.pth が見つかりません。先にjou.pyを実行してください")

        self.teacher_pdn.eval()  # 固定

        # Student PDN（新規、学習対象）
        self.student_pdn = PDN(out_channels=384).to(device)
        self.student_pdn.train()

        # 最適化器（論文通り）
        self.optimizer = optim.Adam(self.student_pdn.parameters(), lr=1e-4, weight_decay=1e-5)

        # MVTec ADデータ準備
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # PDN用サイズ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 正常画像パス
        normal_path = os.path.join(data_path, category, 'train', 'good')
        self.image_paths = glob.glob(os.path.join(normal_path, '*.png'))

        if len(self.image_paths) == 0:
            print(f"WARNING: {normal_path} に画像が見つかりません")
            self.use_dummy = True
        else:
            self.use_dummy = False
            print(f"MVTec AD {category}: {len(self.image_paths)}枚の正常画像を読み込み")

        print("初期化完了")

    def hard_feature_loss(self, teacher_output, student_output, q=0.999):
        """Hard Feature Loss計算"""
        # Teacher-Student間の距離
        distance = (teacher_output - student_output) ** 2

        # 各位置での距離の平均（チャンネル方向）
        distance = torch.mean(distance, dim=1, keepdim=True)  # (batch, 1, 64, 64)

        # Hard threshold（上位0.1%の困難な位置）
        d_hard = torch.quantile(distance, q=q)

        # Hard位置のみでLoss計算
        hard_mask = distance >= d_hard
        if torch.sum(hard_mask) > 0:
            loss_hard = torch.mean(distance[hard_mask])
        else:
            loss_hard = torch.mean(distance)  # fallback

        return loss_hard, hard_mask

    def create_dataloader(self, batch_size=4):
        """MVTec ADデータローダー作成（256×256用）"""
        if self.use_dummy:
            # ダミーデータ（開発用）
            images = torch.randn(20, 3, 256, 256)
            dataset = TensorDataset(images)
        else:
            # 実際のMVTec AD画像
            images = []
            for img_path in self.image_paths:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                images.append(image)

            images = torch.stack(images)
            dataset = TensorDataset(images)

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, train_steps=1000, batch_size=4):
        """PDN同士のHard Feature Loss学習実行（論文通り、テスト用1000ステップ）"""
        print(f"PDN同士のHard Feature Loss学習開始: {train_steps}ステップ（{self.category}）")

        # 学習率スケジューラー（論文通り：95%時点で学習率×0.1）
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(0.95 * train_steps),  # 950ステップ
            gamma=0.1
        )
        print(f"学習率スケジューリング: {int(0.95 * train_steps)}ステップで学習率×0.1")

        # MVTec ADデータローダー
        dataloader = self.create_dataloader(batch_size=batch_size)

        # 無限データローダー（論文通り）
        train_loader_infinite = InfiniteDataloader(dataloader)

        total_loss = 0
        total_hard_ratio = 0

        # 論文通りの実装
        tqdm_obj = tqdm(range(train_steps))
        for step, (images,) in zip(tqdm_obj, train_loader_infinite):
            images = images.to(self.device)

            # Teacher PDN出力（学習済み、固定）
            with torch.no_grad():
                teacher_output = self.teacher_pdn(images)

            # Student PDN出力（新規、学習対象）
            student_output = self.student_pdn(images)

            # Hard Feature Loss計算
            loss_hard, hard_mask = self.hard_feature_loss(teacher_output, student_output)

            # 更新
            self.optimizer.zero_grad()
            loss_hard.backward()
            self.optimizer.step()
            self.scheduler.step()  # 学習率スケジューリング

            # 統計
            total_loss += loss_hard.item()
            hard_ratio = torch.sum(hard_mask).item() / hard_mask.numel()
            total_hard_ratio += hard_ratio

            # 最初のステップで詳細表示
            if step == 0:
                print(f"入力画像: {images.shape}")
                print(f"Teacher PDN出力: {teacher_output.shape}")
                print(f"Student PDN出力: {student_output.shape}")
                print(f"Hard位置の割合: {hard_ratio:.1%}")

            # 定期的にログ出力（論文通り）
            if step % 1000 == 0:
                avg_loss = total_loss / (step + 1)
                avg_hard_ratio = total_hard_ratio / (step + 1)
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Step {step}/{train_steps}")
                print(f"  Loss: {avg_loss:.6f}")
                print(f"  Hard位置割合: {avg_hard_ratio:.1%}")
                print(f"  学習率: {current_lr:.2e}")

            # 学習率変更時にログ
            if step == int(0.95 * train_steps):
                print(f"\n*** Step {step}: 学習率を {current_lr:.2e} に変更 ***")

        print("PDN同士のHard Feature Loss学習完了!")

        # Hard Feature Loss学習済みPDNを保存
        torch.save(self.student_pdn.state_dict(), f'hard_trained_pdn_{self.category}.pth')
        print(f"Hard Feature Loss学習済みPDNを hard_trained_pdn_{self.category}.pth に保存しました")

    def test_anomaly_detection(self):
        """PDN同士の異常検出テスト"""
        print(f"\n=== {self.category}でのPDN同士異常検出テスト ===")

        # テストモード
        self.student_pdn.eval()

        # MVTec ADデータから正常画像を取得
        test_dataloader = self.create_dataloader(batch_size=2)

        with torch.no_grad():
            for (images,) in test_dataloader:
                images = images.to(self.device)

                # Teacher PDN出力
                teacher_output = self.teacher_pdn(images)

                # Student PDN出力
                student_output = self.student_pdn(images)

                # 異常度計算（Teacher-Student差分）
                distance = torch.mean((teacher_output - student_output) ** 2, dim=1)
                anomaly_score = torch.mean(distance, dim=[1, 2])  # 画像全体の平均

                print(f"正常画像の異常度スコア: {anomaly_score.cpu().numpy()}")

                # 1バッチのみテスト
                break

        print("PDN同士異常検出テスト完了")

# 実行
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # PDN同士のHard Feature Loss学習（bottleカテゴリ）
    trainer = HardFeatureLossTrainer(device,
        data_path='C:\\Users\\PC_User\\ken\\mvtec_anomaly_detection',
        category='bottle')
    trainer.train(train_steps=10000, batch_size=2)  # テスト用に短縮（本番は70000）
    trainer.test_anomaly_detection()