#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import time
import os
import psutil
from common import get_pdn_small

# Constants
image_size = 256
out_channels = 384
threshold = 0.6
on_gpu = torch.cuda.is_available()

# Paths
model_dir = 'models/test_time_bottle/bottleneck0.4'
test_image_path = 'mvtec/bottle/test/broken_large/001.png'

# Image preprocessing
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_model_memory(model):
    """Calculate model memory usage in MB."""
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb, param_count

def get_memory_usage():
    """Get current system memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def load_models(model_dir):
    """Load teacher, student, and autoencoder models."""
    print(f"Loading models from {model_dir}...")

    teacher_path = os.path.join(model_dir, 'teacher_final.pth')
    student_path = os.path.join(model_dir, 'student_final.pth')
    autoencoder_path = os.path.join(model_dir, 'autoencoder_final.pth')

    # Load models
    teacher = torch.load(teacher_path, map_location='cpu')
    student = torch.load(student_path, map_location='cpu')
    autoencoder = torch.load(autoencoder_path, map_location='cpu')

    # Set to eval mode
    teacher.eval()
    student.eval()
    autoencoder.eval()

    # Move to GPU if available
    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        print("Models loaded on GPU")
    else:
        print("Models loaded on CPU")

    return teacher, student, autoencoder

@torch.no_grad()
def predict(image, teacher, student, autoencoder):
    """Run inference and return anomaly map."""
    # Forward pass
    teacher_output = teacher(image)
    student_output = student(image)
    autoencoder_output = autoencoder(image)

    # Calculate anomaly maps
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)

    # Combine maps
    map_combined = 0.5 * map_st + 0.5 * map_ae

    return map_combined

def benchmark(image, teacher, student, autoencoder, warmup_iterations=5, benchmark_iterations=1000):
    """Measure inference speed."""
    print(f"\nRunning warmup ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        _ = predict(image, teacher, student, autoencoder)

    if on_gpu:
        torch.cuda.synchronize()

    print(f"Running benchmark ({benchmark_iterations} iterations)...")
    times = []

    for _ in range(benchmark_iterations):
        start_time = time.time()
        _ = predict(image, teacher, student, autoencoder)

        if on_gpu:
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        times.append(elapsed)

    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    return avg_time, std_time, fps

def main():
    print("=" * 60)
    print("EfficientAD Inference Speed Benchmark")
    print("=" * 60)

    # Measure initial memory
    initial_memory = get_memory_usage()
    print(f"\nInitial system memory: {initial_memory:.2f} MB")

    # Load models
    teacher, student, autoencoder = load_models(model_dir)

    # Measure model memory
    teacher_mb, teacher_params = get_model_memory(teacher)
    student_mb, student_params = get_model_memory(student)
    autoencoder_mb, autoencoder_params = get_model_memory(autoencoder)
    total_model_mb = teacher_mb + student_mb + autoencoder_mb
    total_params = teacher_params + student_params + autoencoder_params

    # Measure memory after loading models
    after_load_memory = get_memory_usage()
    memory_increase = after_load_memory - initial_memory

    # GPU memory if available
    if on_gpu:
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
        gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2

    # Print memory usage
    print("\n" + "=" * 60)
    print("MODEL MEMORY USAGE")
    print("=" * 60)
    print(f"Teacher:      {teacher_mb:6.2f} MB  ({teacher_params/1e6:5.2f}M parameters)")
    print(f"Student:      {student_mb:6.2f} MB  ({student_params/1e6:5.2f}M parameters)")
    print(f"Autoencoder:  {autoencoder_mb:6.2f} MB  ({autoencoder_params/1e6:5.2f}M parameters)")
    print("-" * 60)
    print(f"Total Model:  {total_model_mb:6.2f} MB  ({total_params/1e6:5.2f}M parameters)")
    print(f"System Memory: {after_load_memory:6.2f} MB  (+{memory_increase:6.2f} MB)")
    if on_gpu:
        print(f"GPU Allocated: {gpu_memory_mb:6.2f} MB")
        print(f"GPU Reserved:  {gpu_reserved_mb:6.2f} MB")
    print("-" * 60)

    # Check if suitable for 4GB edge device
    edge_device_limit = 4096  # 4GB in MB
    # Assume OS and other processes use ~1.5GB, leaving ~2.5GB for our app
    available_for_app = edge_device_limit - 1500
    if on_gpu:
        peak_memory = max(after_load_memory, gpu_reserved_mb)
    else:
        peak_memory = after_load_memory

    if peak_memory < available_for_app:
        print(f"Edge Device (4GB): OK (using {peak_memory:.0f}/{available_for_app:.0f} MB)")
    else:
        print(f"Edge Device (4GB): NG (using {peak_memory:.0f}/{available_for_app:.0f} MB)")
    print("=" * 60)

    # Load and preprocess image
    print(f"\nLoading test image: {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = default_transform(image)
    image_tensor = image_tensor[None]  # Add batch dimension

    if on_gpu:
        image_tensor = image_tensor.cuda()

    print(f"Image shape: {image_tensor.shape}")

    # Measure peak memory during inference
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()

    # Run benchmark
    avg_time, std_time, fps = benchmark(image_tensor, teacher, student, autoencoder)

    # Get peak memory
    if on_gpu:
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
    peak_system_memory = get_memory_usage()

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Standard deviation:     {std_time*1000:.2f} ms")
    print(f"FPS (frames per sec):   {fps:.2f}")
    if on_gpu:
        print(f"Peak GPU memory:        {peak_gpu_memory:.2f} MB")
    print(f"Peak system memory:     {peak_system_memory:.2f} MB")
    print("=" * 60)

    # Run single inference for anomaly detection
    print("\nRunning anomaly detection...")
    map_combined = predict(image_tensor, teacher, student, autoencoder)
    anomaly_score = torch.max(map_combined).item()

    print(f"\nAnomaly score: {anomaly_score:.4f}")
    print(f"Threshold:     {threshold:.4f}")

    if anomaly_score > threshold:
        print("Result: ANOMALY DETECTED")
    else:
        print("Result: NORMAL")

    print("=" * 60)

if __name__ == '__main__':
    main()
