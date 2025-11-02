"""
MNIST最终实验 - 调优版本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用优化的参数配置进行训练和评估
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.information_oriented_system import InformationOrientedSystem


def load_balanced_mnist(train_per_class=200, test_per_class=50):
    """加载平衡的MNIST数据"""
    print(f"\n加载MNIST数据...")
    print(f"  训练: 每类{train_per_class}个")
    print(f"  测试: 每类{test_per_class}个")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 训练集
    train_data_list = []
    train_labels_list = []
    
    for digit in range(10):
        indices = (train_dataset.targets == digit).nonzero(as_tuple=True)[0][:train_per_class]
        train_data_list.append(train_dataset.data[indices].float() / 255.0)
        train_labels_list.append(train_dataset.targets[indices])
    
    train_data = torch.cat(train_data_list)
    train_labels = torch.cat(train_labels_list)
    
    # 测试集
    test_data_list = []
    test_labels_list = []
    
    for digit in range(10):
        indices = (test_dataset.targets == digit).nonzero(as_tuple=True)[0][:test_per_class]
        test_data_list.append(test_dataset.data[indices].float() / 255.0)
        test_labels_list.append(test_dataset.targets[indices])
    
    test_data = torch.cat(test_data_list)
    test_labels = torch.cat(test_labels_list)
    
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    
    return train_data, train_labels, test_data, test_labels


def prepare_features(images):
    """准备特征 - 保留更多细节"""
    batch_size = images.shape[0]
    device = images.device
    
    # 28x28 → 分成7个patch (每个4行)，每个patch展平为112维
    patches = images.view(batch_size, 7, 4, 28)
    patches = patches.reshape(batch_size, 7, -1)  # [B, 7, 112]
    
    # 扩展到128维
    pad = torch.zeros(batch_size, 7, 16, device=device)
    features = torch.cat([patches, pad], dim=2)
    
    return features


def train_system(system, train_data, train_labels, epochs=10, lr=0.005):
    """训练系统"""
    print(f"\n{'='*80}")
    print(f"开始训练")
    print(f"{'='*80}")
    
    device = system.device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    
    train_features = prepare_features(train_data)
    
    optimizer = torch.optim.Adam(system.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 32
    num_samples = len(train_data)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        system.train()
        perm = torch.randperm(num_samples)
        epoch_loss = 0
        epoch_correct = 0
        num_processed = 0
        
        pbar = tqdm(range(0, num_samples, batch_size), 
                   desc=f"Epoch {epoch+1}/{epochs}",
                   ncols=100)
        
        for i in pbar:
            indices = perm[i:i+batch_size]
            batch_features = train_features[indices]
            batch_labels = train_labels[indices]
            
            optimizer.zero_grad()
            
            # 收集批次预测
            batch_logits = []
            for j in range(len(batch_features)):
                sample = batch_features[j:j+1]
                output = system(sample, return_details=False)
                
                if output['predictions'] is not None:
                    batch_logits.append(output['predictions'])
            
            if not batch_logits:
                continue
            
            logits = torch.cat(batch_logits, dim=0)
            actual_labels = batch_labels[:len(logits)]
            
            loss = criterion(logits, actual_labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            epoch_correct += (pred == actual_labels).sum().item()
            num_processed += len(actual_labels)
            
            # 更新进度条
            current_acc = epoch_correct / num_processed * 100
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{current_acc:.1f}%'})
        
        # Epoch总结
        avg_loss = epoch_loss / (num_samples // batch_size)
        train_acc = epoch_correct / num_processed * 100
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, 准确率={train_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time/60:.2f}分钟")
    
    return system


def evaluate_system(system, test_data, test_labels):
    """评估系统"""
    print(f"\n{'='*80}")
    print(f"评估")
    print(f"={'*80}")
    
    device = system.device
    test_features = prepare_features(test_data).to(device)
    test_labels = test_labels.to(device)
    
    system.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_features)), desc="评估", ncols=100):
            sample = test_features[i:i+1]
            output = system(sample, return_details=False)
            
            if output['predictions'] is not None:
                pred = output['predictions'].argmax(dim=1)
                all_preds.append(pred.item())
                all_true.append(test_labels[i].item())
    
    # 计算准确率
    all_preds = torch.tensor(all_preds)
    all_true = torch.tensor(all_true)
    
    correct = (all_preds == all_true).sum().item()
    accuracy = correct / len(all_true) * 100
    
    print(f"\n总体准确率: {accuracy:.2f}% ({correct}/{len(all_true)})")
    
    # 每类准确率
    print(f"\n每类准确率:")
    for digit in range(10):
        mask = (all_true == digit)
        if mask.sum() > 0:
            digit_acc = (all_preds[mask] == digit).sum().item() / mask.sum().item() * 100
            print(f"  数字 {digit}: {digit_acc:.1f}%")
    
    return accuracy


def visualize_results(system, test_data, test_labels, save_path='../assets/mnist_results.png'):
    """可视化结果"""
    print(f"\n生成可视化...")
    
    device = system.device
    test_features = prepare_features(test_data[:20]).to(device)
    
    system.eval()
    
    preds = []
    with torch.no_grad():
        for i in range(20):
            sample = test_features[i:i+1]
            output = system(sample, return_details=False)
            if output['predictions'] is not None:
                pred = output['predictions'].argmax(dim=1).item()
            else:
                pred = -1
            preds.append(pred)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < 20:
            ax.imshow(test_data[i].cpu().numpy(), cmap='gray')
            true_label = test_labels[i].item()
            pred_label = preds[i]
            
            if pred_label == true_label:
                color = 'green'
                symbol = '✓'
            else:
                color = 'red'
                symbol = '✗'
            
            ax.set_title(f'{symbol} True:{true_label} | Pred:{pred_label}', 
                        color=color, fontsize=12, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ 保存到: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("MNIST最终实验 - 信息化球面系统")
    print("="*80)
    
    # 设置种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 加载数据
    train_data, train_labels, test_data, test_labels = load_balanced_mnist(
        train_per_class=200,
        test_per_class=50
    )
    
    # 初始化系统（使用更大的维度）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    system = InformationOrientedSystem(
        input_dim=128,
        hidden_dim=256,      # 增大隐藏层
        content_dim=32,      # 增大内容维度
        info_dim=32,         # 增大信息维度
        num_classes=10,
        spatial_threshold=0.3,  # 降低聚类阈值，产生更多组
        device=device
    )
    
    print(f"\n系统配置:")
    print(f"  输入维度: 128")
    print(f"  隐藏维度: 256")
    print(f"  内容维度: 32")
    print(f"  信息维度: 32")
    
    # 训练
    system = train_system(system, train_data, train_labels, epochs=15, lr=0.003)
    
    # 评估
    accuracy = evaluate_system(system, test_data, test_labels)
    
    # 可视化
    try:
        visualize_results(system, test_data, test_labels)
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 保存模型
    model_path = '../assets/mnist_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': system.state_dict(),
        'accuracy': accuracy,
    }, model_path)
    print(f"\n✓ 模型已保存: {model_path}")
    
    # 最终总结
    print("\n" + "="*80)
    print("实验完成")
    print("="*80)
    print(f"\n✓ 最终准确率: {accuracy:.2f}%")
    print(f"✓ 训练样本: {len(train_data)}")
    print(f"✓ 测试样本: {len(test_data)}")
    print(f"\n核心特性:")
    print(f"  • 信息元提取 - 直接构建语义单位")
    print(f"  • 信息组构建 - 时空语义聚类")
    print(f"  • 球面映射 - 结构化组织")
    print(f"  • 完全可解释 - 每一步都透明")
    print(f"  • 完全可解码 - 可逆向重构语义")


if __name__ == "__main__":
    main()

