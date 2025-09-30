import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 既存のクラスをインポート
from wideres_mvtec import TeacherMVTec
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

class DistillationTrainer:
    """EfficientAD蒸留学習クラス（MVTec AD対応）"""

    def __init__(self, device='cpu', data_path='./mvtec_anomaly_detection', category='bottle'):
        self.device = device
        self.category = category

        print(f"Teacher（MVTec AD {category}）と PDN を初期化中...")

        # Teacher: MVTec AD対応WideResNet-101
        self.teacher = TeacherMVTec(device, data_path, category)

        # Student: PDN
        self.student = PDN(out_channels=384).to(device)

        # 最適化器（論文通り）
        self.optimizer = optim.Adam(self.student.parameters(), lr=1e-4, weight_decay=1e-5)
        print("初期化完了")

    def create_dataloader(self, batch_size=4):
        """MVTec ADデータローダー作成"""
        return self.teacher.create_dataloader(batch_size=batch_size, shuffle=True)

    def train(self, train_steps=60000, batch_size=4):
        """EfficientAD蒸留学習実行（論文通り60,000ステップ）"""
        print(f"EfficientAD蒸留学習開始: {train_steps}ステップ（{self.category}）")

        # MVTec ADデータローダー
        dataloader = self.create_dataloader(batch_size=batch_size)

        # 無限データローダー（論文通り）
        train_loader_infinite = InfiniteDataloader(dataloader)

        total_loss = 0

        # 論文通りの実装
        tqdm_obj = tqdm(range(train_steps))
        for step, batch_images in zip(tqdm_obj, train_loader_infinite):
            # 512×512画像をデバイスに移動
            batch_images = batch_images.to(self.device)

            # Teacher出力（WideResNet-101特徴）
            with torch.no_grad():
                teacher_output = self.teacher.forward(batch_images)

            # Student入力用に256×256にリサイズ
            student_images = torch.nn.functional.interpolate(
                batch_images, size=(256, 256), mode='bilinear', align_corners=False
            )

            # Student出力（PDN特徴）
            student_output = self.student(student_images)

            # 形状確認（最初のステップのみ）
            if step == 0:
                print(f"入力画像: {batch_images.shape}")
                print(f"Teacher出力: {teacher_output.shape}")
                print(f"Student画像: {student_images.shape}")
                print(f"Student出力: {student_output.shape}")

            # 蒸留Loss計算（MSE）
            loss = nn.MSELoss()(student_output, teacher_output)

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 定期的にログ出力（論文通り）
            if step % 1000 == 0:
                avg_loss = total_loss / (step + 1)
                tqdm_obj.set_description(f"Loss: {avg_loss:.6f}")

        print("EfficientAD蒸留学習完了!")

        # 学習済みPDNを保存
        torch.save(self.student.state_dict(), f'trained_pdn_{self.category}.pth')
        print(f"学習済みPDNを trained_pdn_{self.category}.pth に保存しました")

# 実行
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # EfficientAD蒸留学習（bottleカテゴリ）
    trainer = DistillationTrainer(device,
        data_path='C:\\Users\\PC_User\\ken\\mvtec_anomaly_detection',
        category='bottle')
    trainer.train(train_steps=10000, batch_size=2)  # テスト用に短縮（本番は60000）