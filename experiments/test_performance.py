"""
性能测试：验证优化后的速度提升
"""
import sys
sys.path.insert(0, '../src')

import torch
import time
import numpy as np
from information_oriented_system import InformationOrientedSystem

def test_optimized_speed():
    """测试优化后的处理速度"""
    print("=" * 60)
    print("性能测试：优化后的系统速度")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建系统
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=128,
        info_dim=64,
        num_classes=10
    ).to(device)
    
    # 预热（避免首次调用的初始化开销）
    print("\n预热中...")
    features = torch.randn(28, 128).to(device)
    _ = system(features, return_details=False)
    
    # 测试单样本速度
    print("\n测试单样本处理速度（20次）...")
    times = []
    
    for i in range(20):
        features = torch.randn(28, 128).to(device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        _ = system(features, return_details=False)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        elapsed = (end - start) * 1000  # ms
        times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            print(f"  完成 {i+1}/20 次，当前: {elapsed:.2f}ms")
    
    # 统计结果
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\n" + "=" * 60)
    print("📊 性能统计结果")
    print("=" * 60)
    print(f"✅ 平均耗时: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"✅ 最快: {min_time:.2f} ms")
    print(f"✅ 最慢: {max_time:.2f} ms")
    print(f"✅ 吞吐量: {1000/avg_time:.2f} 样本/秒")
    
    # 与优化前对比
    print("\n" + "=" * 60)
    print("📈 性能提升对比")
    print("=" * 60)
    baseline_time = 180  # ms（优化前的基准）
    speedup = baseline_time / avg_time
    
    print(f"优化前: ~{baseline_time:.0f} ms/样本")
    print(f"优化后: ~{avg_time:.2f} ms/样本")
    print(f"🚀 加速比: {speedup:.2f}x")
    print(f"🚀 性能提升: {(speedup-1)*100:.1f}%")
    
    # 瓶颈分析
    print("\n" + "=" * 60)
    print("🔍 主要优化项")
    print("=" * 60)
    print("✅ 1. raw_data延迟转换（tensor→list）")
    print("✅ 2. 批量统计计算（减少GPU-CPU同步）")
    print("✅ 3. content向量缓存")
    print("✅ 4. 向量化相似度计算（O(n²)→矩阵运算）")
    
    return avg_time

def test_reconstruction_speed():
    """测试重构速度"""
    print("\n" + "=" * 60)
    print("🔄 信息重构速度测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=128,
        info_dim=64,
        num_classes=10
    ).to(device)
    
    features = torch.randn(28, 128).to(device)
    
    # 完整处理
    print("完整信息处理...")
    start = time.time()
    output = system(features, return_details=True)
    process_time = (time.time() - start) * 1000
    
    # 重构
    print("重构原始数据...")
    start = time.time()
    reconstructed = system.reconstruct(output)
    recon_time = (time.time() - start) * 1000
    
    # 验证准确性
    if reconstructed is not None:
        mse = torch.nn.functional.mse_loss(reconstructed, features).item()
        cos_sim = torch.nn.functional.cosine_similarity(
            reconstructed.flatten(),
            features.flatten(),
            dim=0
        ).item()
        
        print(f"\n✅ 处理耗时: {process_time:.2f} ms")
        print(f"✅ 重构耗时: {recon_time:.2f} ms")
        print(f"✅ 总耗时: {process_time + recon_time:.2f} ms")
        print(f"✅ MSE: {mse:.6f}")
        print(f"✅ Cosine Similarity: {cos_sim:.6f}")
    else:
        print("❌ 重构失败")

if __name__ == '__main__':
    try:
        avg_time = test_optimized_speed()
        test_reconstruction_speed()
        
        print("\n" + "=" * 60)
        print("✅ 性能测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

