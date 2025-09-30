import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 既存のクラスをインポート
from wideres_mvtec import TeacherMVTec
from pdn import PDN

class HardFeatureLossTrainer:
    """EfficientAD Hard Feature Loss学習（MVTec AD対応）"""

    def __init__(self, device='cpu', data_path='./mvtec_anomaly_detection', category='bottle'):
        self.device = device
        self.category = category

        print(f"Hard Feature Loss学習システム初期化中（{category}）...")

        # Teacher: MVTec AD対応WideResNet-101（固定）
        self.teacher = TeacherMVTec(device, data_path, category)

        # Student: 新しいPDN（学習対象）
        self.student = PDN(out_channels=384).to(device)
        self.student.train()

        # 最適化器（論文通り）
        self.optimizer = optim.Adam(self.student.parameters(), lr=1e-4, weight_decay=1e-5)

        print("初期化完了")

    def hard_feature_loss(self, teacher_output, student_output, q=0.999):
        """Hard Feature Loss計算（論文通り）"""
        # Teacher-Student間の距離
        distance = (teacher_output - student_output) ** 2

        # チャンネル方向の平均
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
        """MVTec ADデータローダー作成"""
        return self.teacher.create_dataloader(batch_size=batch_size, shuffle=True)

    def train(self, epochs=5, batch_size=4):
        """Hard Feature Loss学習実行（論文通り）"""
        print(f"Hard Feature Loss学習開始: {epochs}エポック（{self.category}）")

        # MVTec ADデータローダー
        dataloader = self.create_dataloader(batch_size=batch_size)

        for epoch in range(epochs):
            total_loss = 0
            total_hard_ratio = 0
            num_batches = 0

            for batch_images in dataloader:
                # 512×512画像をデバイスに移動
                batch_images = batch_images.to(self.device)

                # Teacher出力（WideResNet-101特徴、固定）
                with torch.no_grad():
                    teacher_output = self.teacher.forward(batch_images)

                # Student入力用に256×256にリサイズ
                student_images = torch.nn.functional.interpolate(
                    batch_images, size=(256, 256), mode='bilinear', align_corners=False
                )

                # Student出力（PDN特徴）
                student_output = self.student(student_images)

                # Hard Feature Loss計算
                loss_hard, hard_mask = self.hard_feature_loss(teacher_output, student_output)

                # 更新
                self.optimizer.zero_grad()
                loss_hard.backward()
                self.optimizer.step()

                # 統計
                total_loss += loss_hard.item()
                hard_ratio = torch.sum(hard_mask).item() / hard_mask.numel()
                total_hard_ratio += hard_ratio
                num_batches += 1

                # 最初のバッチで詳細表示
                if epoch == 0 and num_batches == 1:
                    print(f"入力画像: {batch_images.shape}")
                    print(f"Teacher出力: {teacher_output.shape}")
                    print(f"Student画像: {student_images.shape}")
                    print(f"Student出力: {student_output.shape}")
                    print(f"Hard位置の割合: {hard_ratio:.1%}")

            avg_loss = total_loss / num_batches
            avg_hard_ratio = total_hard_ratio / num_batches

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"  Hard位置割合: {avg_hard_ratio:.1%}")

        print("Hard Feature Loss学習完了!")

        # Hard Feature Loss学習済みPDNを保存
        torch.save(self.student.state_dict(), f'hard_trained_pdn_{self.category}.pth')
        print(f"Hard Feature Loss学習済みPDNを hard_trained_pdn_{self.category}.pth に保存しました")

    def test_anomaly_detection(self):
        """EfficientAD異常検出テスト（Teacher vs Student）"""
        print(f"\n=== {self.category}での異常検出テスト ===")

        # テストモード
        self.student.eval()

        # MVTec ADデータから正常画像を取得
        test_dataloader = self.create_dataloader(batch_size=2)

        with torch.no_grad():
            for batch_images in test_dataloader:
                batch_images = batch_images.to(self.device)

                # Teacher出力
                teacher_output = self.teacher.forward(batch_images)

                # Student画像とStudent出力
                student_images = torch.nn.functional.interpolate(
                    batch_images, size=(256, 256), mode='bilinear', align_corners=False
                )
                student_output = self.student(student_images)

                # 異常度計算（Teacher-Student差分）
                distance = torch.mean((teacher_output - student_output) ** 2, dim=1)
                anomaly_score = torch.mean(distance, dim=[1, 2])  # 画像全体の平均

                print(f"正常画像の異常度スコア: {anomaly_score.cpu().numpy()}")

                # 1バッチのみテスト
                break

        print("異常検出テスト完了")

# 実行
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hard Feature Loss学習（bottleカテゴリ）
    trainer = HardFeatureLossTrainer(device, category='bottle')
    trainer.train(epochs=3, batch_size=2)  # 小さめのバッチサイズ
    trainer.test_anomaly_detection()