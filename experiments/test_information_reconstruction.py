"""
信息重构测试 - 验证信息化系统的核心能力

评估指标：
1. 信息保留度 - 原始信息能否完整保留
2. 重构准确度 - 能否从结构逆向重构
3. 结构一致性 - 相似输入 → 相似结构

不评估分类准确率！这不是分类器！
"""

import sys
sys.path.insert(0, '../src')

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from information_oriented_system import InformationOrientedSystem
import numpy as np


def prepare_mnist_features(images, device):
    """准备MNIST特征：28x28 → 28个序列，每个128维"""
    batch_size = images.shape[0]
    images = images.view(batch_size, 28, 28)
    
    features_list = []
    for i in range(batch_size):
        img = images[i]  # [28, 28]
        # 每行作为一个时间步，扩展到128维
        row_features = []
        for row in img:
            # 每行28个像素，扩展到128维
            feature = torch.zeros(128, device=device)
            feature[:28] = row
            # 添加位置编码
            feature[28:32] = torch.tensor([i/28.0] * 4, device=device)
            row_features.append(feature)
        features_list.append(torch.stack(row_features))
    
    return torch.stack(features_list)


def test_information_preservation():
    """测试1：信息保留度"""
    print("\n" + "="*80)
    print("测试1: 信息保留度")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化系统（增加content_dim到128）
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=128,  # 增加到128以保留更多信息
        info_dim=64,
        num_classes=10
    ).to(device)
    
    # 加载MNIST数据
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # 取3个样本
    samples = []
    for i in range(3):
        img, label = dataset[i]
        samples.append((img, label))
    
    print(f"\n测试{len(samples)}个样本的信息保留度...")
    
    total_mse = 0.0
    total_cosine = 0.0
    
    for idx, (img, label) in enumerate(samples):
        # 准备输入
        features = prepare_mnist_features(img.unsqueeze(0), device)  # [1, 28, 128]
        features = features.squeeze(0)  # [28, 128]
        
        # 信息化处理
        output = system(features)
        
        # 重构
        reconstructed = system.reconstruct(output)
        
        if reconstructed is None:
            print(f"  样本{idx+1}: 重构失败")
            continue
        
        # 计算重构误差
        # 原始输入: [28, 128]
        # 重构输出: [num_elements, 128]
        
        # 调整重构序列的长度以匹配原始输入（28步）
        if reconstructed.shape[0] != features.shape[0]:
            # 使用插值调整序列长度
            reconstructed = F.interpolate(
                reconstructed.unsqueeze(0).transpose(1, 2),  # [1, 128, num_elements]
                size=features.shape[0],  # 目标长度28
                mode='linear',
                align_corners=True
            ).transpose(1, 2).squeeze(0)  # [28, 128]
        
        # MSE误差（逐元素比较）
        mse = F.mse_loss(reconstructed, features).item()
        
        # 余弦相似度（整体序列）
        original_flat = features.flatten()
        reconstructed_flat = reconstructed.flatten()
        cosine = F.cosine_similarity(original_flat.unsqueeze(0), reconstructed_flat.unsqueeze(0), dim=-1).item()
        
        total_mse += mse
        total_cosine += cosine
        
        print(f"  样本{idx+1} (标签{label}):")
        print(f"    MSE: {mse:.4f}")
        print(f"    余弦相似度: {cosine:.4f}")
        print(f"    信息元数: {len(output['elements'])}")
        print(f"    信息组数: {len(output['groups'])}")
    
    avg_mse = total_mse / len(samples)
    avg_cosine = total_cosine / len(samples)
    
    print(f"\n总体信息保留度:")
    print(f"  平均MSE: {avg_mse:.4f} (越小越好)")
    print(f"  平均余弦相似度: {avg_cosine:.4f} (越接近1越好)")
    
    # 评估
    if avg_cosine > 0.8:
        print(f"  ✅ 信息保留度优秀")
    elif avg_cosine > 0.6:
        print(f"  ⚠️  信息保留度一般")
    else:
        print(f"  ❌ 信息保留度较差")


def test_structure_consistency():
    """测试2：结构一致性 - 相似输入应该产生相似结构"""
    print("\n" + "="*80)
    print("测试2: 结构一致性")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=128,
        info_dim=64,
        num_classes=10
    ).to(device)
    
    # 加载MNIST数据
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # 找两个相同数字的样本
    samples_by_label = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label not in samples_by_label:
            samples_by_label[label] = []
        if len(samples_by_label[label]) < 2:
            samples_by_label[label].append((img, label))
        if len(samples_by_label[label]) == 2:
            break
    
    print(f"\n测试相同标签样本的结构一致性...")
    
    for label, pairs in samples_by_label.items():
        if len(pairs) < 2:
            continue
        
        # 处理两个样本
        outputs = []
        for img, _ in pairs:
            features = prepare_mnist_features(img.unsqueeze(0), device).squeeze(0)
            output = system(features)
            outputs.append(output)
        
        # 比较结构
        num_elements_1 = len(outputs[0]['elements'])
        num_elements_2 = len(outputs[1]['elements'])
        num_groups_1 = len(outputs[0]['groups'])
        num_groups_2 = len(outputs[1]['groups'])
        
        print(f"\n标签{label}的两个样本:")
        print(f"  样本1: {num_elements_1}个信息元, {num_groups_1}个信息组")
        print(f"  样本2: {num_elements_2}个信息元, {num_groups_2}个信息组")
        
        # 计算结构相似度
        element_diff = abs(num_elements_1 - num_elements_2) / max(num_elements_1, num_elements_2)
        group_diff = abs(num_groups_1 - num_groups_2) / max(num_groups_1, num_groups_2)
        
        print(f"  信息元差异: {element_diff:.2%}")
        print(f"  信息组差异: {group_diff:.2%}")
        
        # 比较球面坐标
        if outputs[0]['sphere_coords'] and outputs[1]['sphere_coords']:
            coords_1 = np.array([c for c in outputs[0]['sphere_coords']])
            coords_2 = np.array([c for c in outputs[1]['sphere_coords']])
            
            # 取平均球面位置
            avg_1 = coords_1.mean(axis=0) if len(coords_1) > 0 else np.zeros(3)
            avg_2 = coords_2.mean(axis=0) if len(coords_2) > 0 else np.zeros(3)
            
            spatial_distance = np.linalg.norm(avg_1 - avg_2)
            print(f"  球面位置距离: {spatial_distance:.4f}")
            
            if spatial_distance < 0.5:
                print(f"  ✅ 结构一致性好")
            else:
                print(f"  ⚠️  结构一致性一般")
        
        break  # 只测试第一个标签


def test_decodability():
    """测试3：可解码性 - 能否从结构解码出语义信息"""
    print("\n" + "="*80)
    print("测试3: 可解码性")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=128,
        info_dim=64,
        num_classes=10
    ).to(device)
    
    # 加载MNIST数据
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    print(f"\n测试解码能力...")
    
    for i in range(3):
        img, label = dataset[i]
        features = prepare_mnist_features(img.unsqueeze(0), device).squeeze(0)
        
        # 信息化处理
        output = system(features)
        
        # 解码
        decoded = system.decode(output)
        
        print(f"\n样本{i+1} (真实标签: {label}):")
        print(f"  解码结果: {decoded}")
        
        # 显示详细结构
        print(f"  结构详情:")
        print(f"    - {len(output['elements'])}个信息元")
        print(f"    - {len(output['groups'])}个信息组")
        if output['groups']:
            avg_coherence = np.mean([g.coherence for g in output['groups']])
            print(f"    - 平均内聚性: {avg_coherence:.3f}")


def test_processing_efficiency():
    """测试4：处理效率"""
    print("\n" + "="*80)
    print("测试4: 处理效率")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=128,  # 必须与信息元的content维度一致
        info_dim=64,
        num_classes=10
    ).to(device)
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    import time
    
    # 测试不同批次大小
    batch_sizes = [1, 10, 50]
    
    for batch_size in batch_sizes:
        # 准备批次数据
        images = []
        for i in range(batch_size):
            img, _ = dataset[i]
            images.append(img)
        
        # 测试处理时间
        start_time = time.time()
        
        for img in images:
            features = prepare_mnist_features(img.unsqueeze(0), device).squeeze(0)
            output = system(features)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / batch_size
        
        print(f"\n批次大小: {batch_size}")
        print(f"  总时间: {total_time:.3f}秒")
        print(f"  平均时间: {avg_time:.4f}秒/样本")
        print(f"  吞吐量: {batch_size/total_time:.2f}样本/秒")
        
        if avg_time < 0.1:
            print(f"  ✅ 处理效率优秀")
        elif avg_time < 0.5:
            print(f"  ⚠️  处理效率一般")
        else:
            print(f"  ❌ 处理效率较低")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("信息化系统重构测试")
    print("="*80)
    print("\n核心评估:")
    print("1. 信息保留度 - 信息是否完整保留")
    print("2. 结构一致性 - 相似输入 → 相似结构")
    print("3. 可解码性 - 结构 → 语义描述")
    print("4. 处理效率 - 低延迟要求")
    print("\n注意: 我们不评估分类准确率！这不是分类器！")
    
    try:
        test_information_preservation()
        test_structure_consistency()
        test_decodability()
        test_processing_efficiency()
        
        print("\n" + "="*80)
        print("✅ 所有测试完成！")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

