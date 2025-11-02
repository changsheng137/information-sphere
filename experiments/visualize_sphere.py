"""
球面分布可视化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

创建多种可视化：
1. 3D球面分布
2. 2D投影图
3. 层次分布直方图
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from src.information_oriented_system import InformationOrientedSystem


def create_sample_data(num_samples=200, num_classes=5):
    """创建示例数据"""
    data = []
    labels = []
    
    for cls in range(num_classes):
        # 每个类别在特征空间的不同位置
        center = torch.randn(128) * 2
        samples = torch.randn(num_samples // num_classes, 30, 128) * 0.5 + center
        data.append(samples)
        labels.extend([cls] * (num_samples // num_classes))
    
    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels)
    
    return data, labels


def collect_sphere_coords(system, data, labels):
    """收集所有样本的球面坐标"""
    coords_by_class = {}
    
    print("处理样本...")
    for i in range(len(data)):
        sample = data[i:i+1]
        label = labels[i].item()
        
        output = system(sample, return_details=True)
        
        if output['groups'] and output['groups'][0].sphere_coords:
            r, theta, phi = output['groups'][0].sphere_coords
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            if label not in coords_by_class:
                coords_by_class[label] = []
            coords_by_class[label].append({
                'r': r, 'theta': theta, 'phi': phi,
                'x': x, 'y': y, 'z': z
            })
    
    return coords_by_class


def plot_3d_sphere(coords_by_class, save_path='../assets/sphere_3d.png'):
    """绘制3D球面分布"""
    print("\n生成3D球面分布图...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制球面网格
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v)) * 0.5
    y_sphere = np.outer(np.sin(u), np.sin(v)) * 0.5
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * 0.5
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # 绘制数据点
    colors = plt.cm.tab10(np.linspace(0, 1, len(coords_by_class)))
    
    for idx, (cls, coords) in enumerate(sorted(coords_by_class.items())):
        coords_array = np.array([[c['x'], c['y'], c['z']] for c in coords])
        ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2],
                  c=[colors[idx]], label=f'Class {cls}',
                  alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Information Distribution on Sphere', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # 设置相同的坐标轴范围
    max_range = 1.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到: {save_path}")
    plt.close()


def plot_2d_projections(coords_by_class, save_path='../assets/sphere_2d.png'):
    """绘制2D投影图"""
    print("\n生成2D投影图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(coords_by_class)))
    
    # XY平面
    ax = axes[0, 0]
    for idx, (cls, coords) in enumerate(sorted(coords_by_class.items())):
        coords_array = np.array([[c['x'], c['y']] for c in coords])
        ax.scatter(coords_array[:, 0], coords_array[:, 1],
                  c=[colors[idx]], label=f'Class {cls}',
                  alpha=0.6, s=30)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title('XY Projection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # XZ平面
    ax = axes[0, 1]
    for idx, (cls, coords) in enumerate(sorted(coords_by_class.items())):
        coords_array = np.array([[c['x'], c['z']] for c in coords])
        ax.scatter(coords_array[:, 0], coords_array[:, 1],
                  c=[colors[idx]], label=f'Class {cls}',
                  alpha=0.6, s=30)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Z', fontsize=11)
    ax.set_title('XZ Projection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # YZ平面
    ax = axes[1, 0]
    for idx, (cls, coords) in enumerate(sorted(coords_by_class.items())):
        coords_array = np.array([[c['y'], c['z']] for c in coords])
        ax.scatter(coords_array[:, 0], coords_array[:, 1],
                  c=[colors[idx]], label=f'Class {cls}',
                  alpha=0.6, s=30)
    ax.set_xlabel('Y', fontsize=11)
    ax.set_ylabel('Z', fontsize=11)
    ax.set_title('YZ Projection', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 极坐标投影 (θ, φ)
    ax = axes[1, 1]
    for idx, (cls, coords) in enumerate(sorted(coords_by_class.items())):
        coords_array = np.array([[c['theta'], c['phi']] for c in coords])
        ax.scatter(coords_array[:, 0], coords_array[:, 1],
                  c=[colors[idx]], label=f'Class {cls}',
                  alpha=0.6, s=30)
    ax.set_xlabel('θ (Polar)', fontsize=11)
    ax.set_ylabel('φ (Azimuthal)', fontsize=11)
    ax.set_title('Angular Distribution (θ, φ)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到: {save_path}")
    plt.close()


def plot_radial_distribution(coords_by_class, save_path='../assets/radial_dist.png'):
    """绘制径向分布（抽象层次）"""
    print("\n生成径向分布图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    for cls, coords in sorted(coords_by_class.items()):
        r_values = [c['r'] for c in coords]
        ax1.hist(r_values, bins=20, alpha=0.6, label=f'Class {cls}')
    
    ax1.set_xlabel('Radius r (Abstraction Level)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Radial Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 箱线图
    data_for_box = []
    labels_for_box = []
    for cls, coords in sorted(coords_by_class.items()):
        r_values = [c['r'] for c in coords]
        data_for_box.append(r_values)
        labels_for_box.append(f'Class {cls}')
    
    ax2.boxplot(data_for_box, labels=labels_for_box)
    ax2.set_ylabel('Radius r', fontsize=11)
    ax2.set_title('Abstraction Level by Class', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到: {save_path}")
    plt.close()


def plot_architecture_diagram(save_path='../assets/architecture.png'):
    """绘制架构图"""
    print("\n生成架构图...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 定义各层的位置
    layers = [
        {'name': '原始输入\n(Text/Image/Sensor)', 'y': 0.9, 'color': '#E8F4F8'},
        {'name': '信息元提取\n• 语义边界识别\n• 提取时空内容', 'y': 0.7, 'color': '#B8E6F0'},
        {'name': '信息组构建\n• 时空聚类\n• 计算内聚性', 'y': 0.5, 'color': '#88D8E8'},
        {'name': '球面映射\n• 映射到球面(r,θ,φ)\n• 径向=抽象层次', 'y': 0.3, 'color': '#58CAE0'},
        {'name': '结构化输出\n• 可解释\n• 可解码', 'y': 0.1, 'color': '#28BCD8'},
    ]
    
    for i, layer in enumerate(layers):
        # 绘制方框
        rect = plt.Rectangle((0.2, layer['y'] - 0.05), 0.6, 0.08,
                            facecolor=layer['color'], edgecolor='black',
                            linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # 添加文字
        ax.text(0.5, layer['y'], layer['name'],
               ha='center', va='center', fontsize=11,
               fontweight='bold', transform=ax.transAxes)
        
        # 添加箭头
        if i < len(layers) - 1:
            ax.annotate('', xy=(0.5, layers[i+1]['y'] + 0.03),
                       xytext=(0.5, layer['y'] - 0.05),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                       transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("信息化球面系统 - 可视化生成")
    print("="*60)
    
    # 创建示例数据
    print("\n创建示例数据...")
    data, labels = create_sample_data(num_samples=200, num_classes=5)
    
    # 初始化系统
    print("初始化系统...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    system = InformationOrientedSystem(
        input_dim=128,
        content_dim=32,
        info_dim=32,
        device=device
    )
    
    # 收集球面坐标
    coords_by_class = collect_sphere_coords(system, data, labels)
    
    # 生成各种可视化
    plot_3d_sphere(coords_by_class)
    plot_2d_projections(coords_by_class)
    plot_radial_distribution(coords_by_class)
    plot_architecture_diagram()
    
    print("\n" + "="*60)
    print("所有可视化生成完成！")
    print("="*60)
    print("\n文件保存在 assets/ 目录:")
    print("  • sphere_3d.png - 3D球面分布")
    print("  • sphere_2d.png - 2D投影")
    print("  • radial_dist.png - 径向分布")
    print("  • architecture.png - 架构图")


if __name__ == "__main__":
    main()

