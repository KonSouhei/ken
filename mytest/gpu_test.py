import torch
import torchvision
import time
import gc

def check_gpu_basic():
    """基本的なGPU情報確認"""
    print("=== GPU基本情報 ===")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU数: {torch.cuda.device_count()}")
        print(f"現在のGPU: {torch.cuda.current_device()}")
        print(f"GPU名: {torch.cuda.get_device_name()}")

        # メモリ情報
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"総メモリ: {total_memory:.2f} GB")

        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"使用中メモリ: {allocated:.2f} GB")
        print(f"キャッシュメモリ: {cached:.2f} GB")
    else:
        print("CUDA未対応またはGPU未検出")

def test_basic_operations():
    """基本的なテンソル演算テスト"""
    print("\n=== 基本演算テスト ===")

    if not torch.cuda.is_available():
        print("GPUが利用できません")
        return

    device = torch.device('cuda')

    # CPU vs GPU速度比較
    size = 1000

    # CPU
    start = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start

    # GPU
    start = time.time()
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    torch.cuda.synchronize()  # GPU処理完了を待機

    start_compute = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_compute

    print(f"CPU時間: {cpu_time:.4f}秒")
    print(f"GPU時間: {gpu_time:.4f}秒")
    print(f"GPU高速化率: {cpu_time/gpu_time:.2f}倍")

    # メモリ使用量
    allocated = torch.cuda.memory_allocated() / 1024**2
    print(f"GPU メモリ使用量: {allocated:.2f} MB")

def test_wideres_model():
    """WideResNet-101のGPU動作テスト"""
    print("\n=== WideResNet-101 テスト ===")

    if not torch.cuda.is_available():
        print("GPUが利用できません")
        return

    device = torch.device('cuda')

    try:
        # メモリクリア
        torch.cuda.empty_cache()

        print("WideResNet-101をロード中...")
        model = torchvision.models.wide_resnet101_2(pretrained=True)
        model = model.to(device)
        model.eval()

        print("モデルロード完了")

        # メモリ使用量確認
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"モデルロード後メモリ: {allocated:.2f} MB")

        # テスト画像作成
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            try:
                print(f"\nバッチサイズ {batch_size} でテスト中...")

                # ダミー画像（512x512）
                test_images = torch.randn(batch_size, 3, 512, 512, device=device)

                # 推論実行
                start = time.time()
                with torch.no_grad():
                    output = model(test_images)
                torch.cuda.synchronize()
                inference_time = time.time() - start

                # メモリ使用量
                allocated = torch.cuda.memory_allocated() / 1024**2

                print(f"  推論時間: {inference_time:.4f}秒")
                print(f"  出力形状: {output.shape}")
                print(f"  メモリ使用: {allocated:.2f} MB")

                # メモリクリア
                del test_images, output
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  バッチサイズ {batch_size}: メモリ不足")
                    torch.cuda.empty_cache()
                else:
                    print(f"  エラー: {e}")

        print("WideResNet-101テスト完了")

    except Exception as e:
        print(f"WideResNet-101テストエラー: {e}")
    finally:
        # メモリクリア
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def test_memory_limits():
    """メモリ限界テスト"""
    print("\n=== メモリ限界テスト ===")

    if not torch.cuda.is_available():
        print("GPUが利用できません")
        return

    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # 段階的にメモリを使用
    sizes = [100, 500, 1000, 2000, 5000]

    for size in sizes:
        try:
            print(f"サイズ {size}x{size} のテンソル作成中...")
            tensor = torch.randn(size, size, device=device)

            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"  成功: {allocated:.2f} MB使用")

            del tensor
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  サイズ {size}: メモリ不足")
                torch.cuda.empty_cache()
                break
            else:
                print(f"  エラー: {e}")

def main():
    """メイン実行"""
    print("GPU動作確認テスト開始\n")

    # 基本情報
    check_gpu_basic()

    # 基本演算
    test_basic_operations()

    # WideResNet-101
    test_wideres_model()

    # メモリ限界
    test_memory_limits()

    print("\n=== 最終メモリ状況 ===")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"使用中メモリ: {allocated:.2f} MB")
        print(f"キャッシュメモリ: {cached:.2f} MB")

    print("\nGPU動作確認テスト完了!")

if __name__ == "__main__":
    main()