#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import torch
import torch.cuda
from torch import nn
import os
from PIL import Image
import torchvision.transforms as transforms
from common import get_pdn_small, get_autoencoder

def load_trained_models(model_dir, out_channels=384):
    """学習済みモデルを読み込み"""
    # 学習済みモデル直接読み込み（efficientad.pyではモデル全体を保存）
    teacher_path = os.path.join(model_dir, 'teacher_final.pth')
    student_path = os.path.join(model_dir, 'student_final.pth')
    autoencoder_path = os.path.join(model_dir, 'autoencoder_final.pth')

    teacher = torch.load(teacher_path, map_location='cpu')
    student = torch.load(student_path, map_location='cpu')
    autoencoder = torch.load(autoencoder_path, map_location='cpu')

    # 評価モードに設定
    teacher.eval()
    student.eval()
    autoencoder.eval()

    return teacher, student, autoencoder

def load_test_images(data_dir, category='bottle', max_images=83):
    """テスト画像を読み込み"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = []

    # 正常画像
    good_dir = os.path.join(data_dir, category, 'test', 'good')
    if os.path.exists(good_dir):
        for img_file in os.listdir(good_dir)[:max_images//4]:
            img_path = os.path.join(good_dir, img_file)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                images.append(image)

    # 異常画像も追加（全anomalyタイプを自動検出）
    test_dir = os.path.join(data_dir, category, 'test')
    if os.path.exists(test_dir):
        for anomaly_type in os.listdir(test_dir):
            if anomaly_type == 'good':
                continue
            anom_dir = os.path.join(test_dir, anomaly_type)
            if os.path.isdir(anom_dir):
                for img_file in os.listdir(anom_dir):
                    img_path = os.path.join(anom_dir, img_file)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image = Image.open(img_path).convert('RGB')
                        image = transform(image)
                        images.append(image)

    return images[:max_images]

def detailed_inference_benchmark(category='bottle'):
    """詳細な推論時間測定"""
    # GPU使用可能性確認
    gpu = torch.cuda.is_available()
    print(f"GPU available: {gpu}")
    print(f"Category: {category}")

    # モデル読み込み
    model_dir = f'./output/1/trainings/mvtec_ad/{category}'
    teacher, student, autoencoder = load_trained_models(model_dir)

    # GPU設定
    if gpu:
        teacher.half().cuda()
        student.half().cuda()
        autoencoder.half().cuda()

    # テスト画像読み込み
    data_dir = './mvtec-2'
    images = load_test_images(data_dir, category)
    print(f"Loaded {len(images)} test images")

    if not images:
        print("No images found! Using random tensor instead.")
        images = [torch.randn(3, 256, 256) for _ in range(83)]

    # 時間測定
    teacher_times = []
    student_times = []
    autoencoder_times = []
    anomaly_map_times = []
    cpu_transfer_times = []
    total_times = []

    with torch.no_grad():
        for rep in range(len(images)):
            image = images[rep].unsqueeze(0)  # バッチ次元追加

            if gpu:
                image = image.half().cuda()

            # 全体の時間測定開始
            total_start = time()

            # Teacher推論
            teacher_start = time()
            t = teacher(image)
            teacher_time = time() - teacher_start

            # Student推論
            student_start = time()
            s = student(image)
            student_time = time() - student_start

            # Autoencoder推論
            ae_start = time()
            ae = autoencoder(image)
            ae_time = time() - ae_start

            # 異常マップ計算
            map_start = time()
            st_map = torch.mean((t - s[:, :384]) ** 2, dim=1)
            ae_map = torch.mean((ae - s[:, 384:]) ** 2, dim=1)
            result_map = st_map + ae_map
            map_time = time() - map_start

            # CPU転送
            transfer_start = time()
            result_on_cpu = result_map.cpu().numpy()
            transfer_time = time() - transfer_start

            # 全体時間
            total_time = time() - total_start

            # 記録
            teacher_times.append(teacher_time * 1000)  # ms単位
            student_times.append(student_time * 1000)
            autoencoder_times.append(ae_time * 1000)
            anomaly_map_times.append(map_time * 1000)
            cpu_transfer_times.append(transfer_time * 1000)
            total_times.append(total_time * 1000)

    # 結果表示
    print("\n=== 学習済みモデル推論時間測定結果 ===")
    print(f"測定画像数: {len(images)}枚")
    print(f"GPU使用: {gpu}")
    print(f"モデルサイズ: small")
    print()
    print("平均時間 (ms):")
    print(f"1. Teacher推論:     {np.mean(teacher_times):.3f} ± {np.std(teacher_times):.3f}")
    print(f"2. Student推論:     {np.mean(student_times):.3f} ± {np.std(student_times):.3f}")
    print(f"3. Autoencoder推論: {np.mean(autoencoder_times):.3f} ± {np.std(autoencoder_times):.3f}")
    print(f"4. 異常マップ計算:   {np.mean(anomaly_map_times):.3f} ± {np.std(anomaly_map_times):.3f}")
    print(f"5. GPU→CPU転送:     {np.mean(cpu_transfer_times):.3f} ± {np.std(cpu_transfer_times):.3f}")
    print(f"6. 合計時間:        {np.mean(total_times):.3f} ± {np.std(total_times):.3f}")
    print()
    print("参考:")
    print(f"benchmark.py (ランダム重み): 13.6ms")
    print(f"FPS換算: {1000/np.mean(total_times):.1f} FPS")

if __name__ == "__main__":
    detailed_inference_benchmark()