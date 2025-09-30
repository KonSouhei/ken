import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
from torchvision import transforms
from PIL import Image
import os
import glob

# 日本語フォント設定（無い場合は英語表示）
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
import matplotlib
matplotlib.use('Agg')  # GUI無効化

# 既存のクラスをインポート
from pdn import PDN

class AnomalyDetector:
    """EfficientAD異常検出評価システム"""

    def __init__(self, device='cpu', data_path='C:\\Users\\PC_User\\ken\\mvtec_anomaly_detection', category='bottle'):
        self.device = device
        self.category = category
        self.data_path = data_path

        print(f"EfficientAD異常検出システムを初期化中（{category}）...")

        # Teacher PDN（蒸留学習済み）
        self.teacher_pdn = PDN(out_channels=384).to(device)
        try:
            self.teacher_pdn.load_state_dict(torch.load(f'trained_pdn_{category}.pth', map_location=device))
            print(f"Teacher PDN（蒸留学習済み）をロードしました: trained_pdn_{category}.pth")
        except FileNotFoundError:
            print(f"WARNING: trained_pdn_{category}.pth が見つかりません")

        # Student PDN（Hard Feature Loss学習済み）
        self.student_pdn = PDN(out_channels=384).to(device)
        try:
            self.student_pdn.load_state_dict(torch.load(f'hard_trained_pdn_{category}.pth', map_location=device))
            print(f"Student PDN（Hard Feature Loss学習済み）をロードしました: hard_trained_pdn_{category}.pth")
        except FileNotFoundError:
            print(f"WARNING: hard_trained_pdn_{category}.pth が見つかりません")

        # 両方とも評価モード
        self.teacher_pdn.eval()
        self.student_pdn.eval()

        # MVTec AD用transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("異常検出システム初期化完了")

    def detect_anomaly(self, image):
        """単一画像の異常検出"""
        with torch.no_grad():
            # Teacher出力
            teacher_output = self.teacher_pdn(image)

            # Student出力
            student_output = self.student_pdn(image)

            # 出力差分計算
            difference = (teacher_output - student_output) ** 2

            return difference, teacher_output, student_output

    def calculate_anomaly_score(self, difference):
        """異常度スコア計算"""
        # チャンネル方向の平均
        anomaly_map = torch.mean(difference, dim=1, keepdim=True)  # (batch, 1, 64, 64)

        # 画像全体の異常度スコア
        anomaly_score = torch.mean(anomaly_map, dim=[1, 2, 3])  # (batch,)

        return anomaly_score, anomaly_map

    def load_mvtec_test_data(self, max_normal=50, max_abnormal=50):
        """MVTec ADテストデータロード"""
        print(f"MVTec AD {self.category} テストデータをロード中...")

        normal_images = []
        abnormal_images = []
        normal_labels = []
        abnormal_labels = []

        # 正常テスト画像
        normal_test_path = os.path.join(self.data_path, self.category, 'test', 'good')
        if os.path.exists(normal_test_path):
            normal_paths = glob.glob(os.path.join(normal_test_path, '*.png'))[:max_normal]
            for img_path in normal_paths:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                normal_images.append(image)
                normal_labels.append(0)
            print(f"正常テスト画像: {len(normal_images)}枚")

        # 異常テスト画像（全defectタイプ）
        test_path = os.path.join(self.data_path, self.category, 'test')
        if os.path.exists(test_path):
            defect_types = [d for d in os.listdir(test_path) if d != 'good' and os.path.isdir(os.path.join(test_path, d))]

            abnormal_count = 0
            for defect_type in defect_types:
                if abnormal_count >= max_abnormal:
                    break

                defect_path = os.path.join(test_path, defect_type)
                defect_paths = glob.glob(os.path.join(defect_path, '*.png'))

                for img_path in defect_paths:
                    if abnormal_count >= max_abnormal:
                        break
                    image = Image.open(img_path).convert('RGB')
                    image = self.transform(image)
                    abnormal_images.append(image)
                    abnormal_labels.append(1)
                    abnormal_count += 1

            print(f"異常テスト画像: {len(abnormal_images)}枚 (defectタイプ: {defect_types})")

        if len(normal_images) == 0 or len(abnormal_images) == 0:
            print("MVTec ADデータが見つかりません。ダミーデータを使用します。")
            return self.create_dummy_data(max_normal, max_abnormal)

        # テンソルに変換
        all_images = torch.stack(normal_images + abnormal_images)
        all_labels = torch.tensor(normal_labels + abnormal_labels, dtype=torch.float32)

        print(f"MVTec ADテストデータ準備完了: 正常{len(normal_images)}枚, 異常{len(abnormal_images)}枚")
        return all_images, all_labels

    def create_dummy_data(self, num_normal=50, num_abnormal=50):
        """ダミーテストデータ作成（MVTec ADデータが無い場合）"""
        print("ダミーテストデータを作成中...")

        # 正常画像（標準的なランダム画像）
        normal_images = torch.randn(num_normal, 3, 256, 256)
        normal_labels = torch.zeros(num_normal)  # 0 = 正常

        # 異常画像（ノイズ追加、明度変更など）
        abnormal_images = []

        # タイプ1: 高ノイズ画像
        noisy_images = torch.randn(num_abnormal//3, 3, 256, 256) * 3
        abnormal_images.append(noisy_images)

        # タイプ2: 明度異常画像
        bright_images = torch.randn(num_abnormal//3, 3, 256, 256) + 2
        abnormal_images.append(bright_images)

        # タイプ3: パターン異常画像
        pattern_images = torch.randn(num_abnormal - 2*(num_abnormal//3), 3, 256, 256)
        # チェッカーボードパターン追加
        for i in range(pattern_images.shape[0]):
            pattern_images[i, :, ::32, ::32] += 5
        abnormal_images.append(pattern_images)

        abnormal_images = torch.cat(abnormal_images, dim=0)
        abnormal_labels = torch.ones(num_abnormal)  # 1 = 異常

        # 全データ結合
        all_images = torch.cat([normal_images, abnormal_images], dim=0)
        all_labels = torch.cat([normal_labels, abnormal_labels], dim=0)

        print(f"ダミーテストデータ作成完了: 正常{num_normal}枚, 異常{num_abnormal}枚")

        return all_images, all_labels

    def evaluate(self, num_normal=50, num_abnormal=50, use_mvtec=True):
        """異常検出性能評価"""
        print(f"=== EfficientAD異常検出性能評価 ({self.category}) ===")

        # テストデータ作成
        if use_mvtec:
            test_images, test_labels = self.load_mvtec_test_data(num_normal, num_abnormal)
        else:
            test_images, test_labels = self.create_dummy_data(num_normal, num_abnormal)

        test_images = test_images.to(self.device)

        all_scores = []
        all_maps = []

        print("異常検出実行中...")

        # バッチ処理
        batch_size = 10
        for i in range(0, len(test_images), batch_size):
            batch_images = test_images[i:i+batch_size]

            # 異常検出
            difference, teacher_out, student_out = self.detect_anomaly(batch_images)
            anomaly_scores, anomaly_maps = self.calculate_anomaly_score(difference)

            all_scores.append(anomaly_scores.cpu())
            all_maps.append(anomaly_maps.cpu())

        # 結果結合
        all_scores = torch.cat(all_scores, dim=0).numpy()
        all_maps = torch.cat(all_maps, dim=0).numpy()
        test_labels = test_labels.numpy()

        # 評価メトリクス計算
        auc_score = roc_auc_score(test_labels, all_scores)

        print(f"\n=== 評価結果 ===")
        print(f"ROC-AUC スコア: {auc_score:.4f}")

        # 正常/異常の異常度スコア統計
        normal_scores = all_scores[test_labels == 0]
        abnormal_scores = all_scores[test_labels == 1]

        print(f"\n正常画像の異常度:")
        print(f"  平均: {np.mean(normal_scores):.6f}")
        print(f"  標準偏差: {np.std(normal_scores):.6f}")

        print(f"\n異常画像の異常度:")
        print(f"  平均: {np.mean(abnormal_scores):.6f}")
        print(f"  標準偏差: {np.std(abnormal_scores):.6f}")

        # 可視化
        self.visualize_results(all_scores, test_labels, all_maps)

        return auc_score

    def visualize_results(self, scores, labels, maps):
        """結果可視化"""
        print("\n結果を可視化中...")

        plt.figure(figsize=(15, 5))

        # 1. 異常度分布
        plt.subplot(1, 3, 1)
        normal_scores = scores[labels == 0]
        abnormal_scores = scores[labels == 1]

        plt.hist(normal_scores, bins=20, alpha=0.7, label='Normal Images', color='blue')
        plt.hist(abnormal_scores, bins=20, alpha=0.7, label='Abnormal Images', color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()

        # 2. ROC曲線
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        # 3. 異常マップ例
        plt.subplot(1, 3, 3)
        # 最も異常度の高い画像の異常マップ
        max_idx = np.argmax(scores)
        anomaly_map = maps[max_idx, 0]  # (64, 64)

        plt.imshow(anomaly_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Anomaly Map Example\n(Score: {scores[max_idx]:.6f})')

        plt.tight_layout()
        result_filename = f'anomaly_detection_results_{self.category}.png'
        plt.savefig(result_filename, dpi=150, bbox_inches='tight')
        # plt.show()  # GUI無効化のためコメントアウト

        print(f"可視化完了: {result_filename} に保存されました")

    def compare_models(self):
        """Teacher-Student出力比較"""
        print("\n=== Teacher-Student出力比較 ===")

        # テスト画像生成
        test_image = torch.randn(1, 3, 256, 256).to(self.device)

        with torch.no_grad():
            teacher_output = self.teacher_pdn(test_image)
            student_output = self.student_pdn(test_image)

            print(f"Teacher出力形状: {teacher_output.shape}")
            print(f"Student出力形状: {student_output.shape}")

            # 出力統計
            print(f"\nTeacher出力統計:")
            print(f"  平均: {torch.mean(teacher_output).item():.6f}")
            print(f"  標準偏差: {torch.std(teacher_output).item():.6f}")

            print(f"\nStudent出力統計:")
            print(f"  平均: {torch.mean(student_output).item():.6f}")
            print(f"  標準偏差: {torch.std(student_output).item():.6f}")

            # 差分統計
            difference = (teacher_output - student_output) ** 2
            print(f"\n出力差分統計:")
            print(f"  平均: {torch.mean(difference).item():.6f}")
            print(f"  最大: {torch.max(difference).item():.6f}")

# 実行
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 異常検出システム作成（bottleカテゴリ）
    detector = AnomalyDetector(
        device=device,
        data_path='C:\\Users\\PC_User\\ken\\mvtec_anomaly_detection',
        category='bottle'
    )

    # モデル比較
    detector.compare_models()

    # 異常検出性能評価（MVTec ADの実際のデータ使用）
    print("\n=== MVTec AD実データでのテスト ===")
    auc_score = detector.evaluate(num_normal=30, num_abnormal=30, use_mvtec=True)

    print(f"\n=== 最終結果 ({detector.category}) ===")
    print(f"EfficientAD異常検出性能: {auc_score:.4f}")

    if auc_score > 0.8:
        print("✅ 優秀な異常検出性能!")
    elif auc_score > 0.6:
        print("⚠️  まずまずの異常検出性能")
    else:
        print("❌ 異常検出性能要改善")

    print(f"\n結果画像: anomaly_detection_results_{detector.category}.png")
    print("学習済みモデルが必要です:")
    print(f"  - trained_pdn_{detector.category}.pth (jou.pyで生成)")
    print(f"  - hard_trained_pdn_{detector.category}.pth (hard.pyで生成)")