"""
基础使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

演示信息化球面系统的基本功能
"""

import sys
sys.path.append('..')

import torch
from src.information_oriented_system import InformationOrientedSystem


def example1_basic_processing():
    """示例1: 基础信息处理"""
    print("\n" + "="*60)
    print("示例1: 基础信息处理")
    print("="*60)
    
    # 创建系统
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=32,
        info_dim=32
    )
    
    # 创建输入数据 (模拟文本/图像特征)
    data = torch.randn(1, 20, 128)  # [batch, seq_len, dim]
    
    # 处理
    output = system(data, return_details=True)
    
    # 查看结果
    print(f"\n输入形状: {data.shape}")
    print(f"提取的信息元数量: {len(output['elements'])}")
    print(f"构建的信息组数量: {len(output['groups'])}")
    
    # 查看信息元详情
    if output['elements']:
        elem = output['elements'][0]
        print(f"\n信息元示例:")
        print(f"  类型: {elem.element_type.value}")
        print(f"  语义角色: {elem.semantic_role.value}")
        print(f"  空间位置: {elem.spatial.numpy()}")
        print(f"  时间戳: {elem.temporal.item():.4f}")
        print(f"  重要性: {elem.importance:.3f}")
        print(f"  确定性: {elem.certainty:.3f}")


def example2_decoding():
    """示例2: 信息解码"""
    print("\n" + "="*60)
    print("示例2: 信息解码")
    print("="*60)
    
    system = InformationOrientedSystem(input_dim=128, content_dim=32)
    
    # 处理数据
    data = torch.randn(1, 30, 128)
    output = system(data, return_details=True)
    
    # 解码
    decoded = system.decode(output)
    
    print(f"\n原始输入: [batch=1, seq_len=30, dim=128]")
    print(f"\n解码结果:")
    print(f"  {decoded}")
    
    print(f"\n说明:")
    print(f"  - 可以看到系统识别出的语义结构")
    print(f"  - 每个信息组的类型和角色")
    print(f"  - 在球面上的位置")


def example3_interpretability():
    """示例3: 可解释性"""
    print("\n" + "="*60)
    print("示例3: 可解释性分析")
    print("="*60)
    
    system = InformationOrientedSystem(input_dim=128, content_dim=32)
    
    # 模拟两个不同的场景
    scene1 = torch.randn(1, 20, 128) + torch.tensor([1.0, 0.0, 0.0])
    scene2 = torch.randn(1, 20, 128) + torch.tensor([0.0, 1.0, 0.0])
    
    output1 = system(scene1, return_details=True)
    output2 = system(scene2, return_details=True)
    
    print("\n场景1:")
    print(f"  信息元: {len(output1['elements'])}个")
    print(f"  信息组: {len(output1['groups'])}个")
    if output1['groups']:
        r, theta, phi = output1['groups'][0].sphere_coords or (0,0,0)
        print(f"  球面位置: r={r:.3f}, θ={theta:.3f}, φ={phi:.3f}")
    
    print("\n场景2:")
    print(f"  信息元: {len(output2['elements'])}个")
    print(f"  信息组: {len(output2['groups'])}个")
    if output2['groups']:
        r, theta, phi = output2['groups'][0].sphere_coords or (0,0,0)
        print(f"  球面位置: r={r:.3f}, θ={theta:.3f}, φ={phi:.3f}")
    
    print(f"\n说明:")
    print(f"  - 不同场景在球面上有不同的位置")
    print(f"  - 可以追踪信息处理的全过程")
    print(f"  - 每个维度都有明确含义")


def example4_batch_processing():
    """示例4: 批处理"""
    print("\n" + "="*60)
    print("示例4: 批处理")
    print("="*60)
    
    import time
    
    system = InformationOrientedSystem(input_dim=128, content_dim=32)
    
    # 批处理
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        data = torch.randn(batch_size, 20, 128)
        
        start = time.time()
        output = system(data, return_details=False)  # 不返回详情，更快
        elapsed = (time.time() - start) * 1000
        
        print(f"\nbatch_size={batch_size}:")
        print(f"  处理时间: {elapsed:.2f} ms")
        print(f"  平均/样本: {elapsed/batch_size:.2f} ms")


def example5_visualization():
    """示例5: 可视化结构"""
    print("\n" + "="*60)
    print("示例5: 可视化信息结构")
    print("="*60)
    
    system = InformationOrientedSystem(input_dim=128, content_dim=32)
    
    data = torch.randn(1, 25, 128)
    output = system(data, return_details=True)
    
    # 使用系统内置的可视化
    system.visualize_structure(output)


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("信息化球面系统 - 基础使用示例")
    print("="*80)
    
    example1_basic_processing()
    example2_decoding()
    example3_interpretability()
    example4_batch_processing()
    example5_visualization()
    
    print("\n" + "="*80)
    print("所有示例完成！")
    print("="*80)


if __name__ == "__main__":
    main()

