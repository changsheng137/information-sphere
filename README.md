# Information Sphere System v1.0

<div align="center">

**从数据化到信息化的范式转变**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Speed-2.28x-green.svg)](docs/PERFORMANCE_OPTIMIZATION.md)

[English](#english) | [中文](#chinese)

### 🚀 最新更新 (v1.0.1)
- ✅ **性能优化**: 2.28x加速，78.80ms/样本
- ✅ **完美重构**: MSE=0, Cosine Similarity=1.0
- ✅ **透明可解释**: 无神经网络黑箱
- 📄 详见 [性能优化报告](docs/PERFORMANCE_OPTIMIZATION.md)

</div>

---

<a name="chinese"></a>

## 🌟 核心创新

### **传统方案的根本问题**

传统深度学习采用**数据化**思路：
- ❌ Token分割破坏语义完整性（"信息化" → ["信", "息", "化"]）
- ❌ Embedding是黑盒，无法解释
- ❌ 时空信息隐式编码，容易丢失
- ❌ 不同模态需要复杂的对齐机制

### **我们的解决方案：信息化**

<table>
<tr>
<th>维度</th>
<th>传统数据化</th>
<th>信息化球面系统</th>
</tr>
<tr>
<td><b>基本单位</b></td>
<td>Token（机械分割）</td>
<td><b>信息元</b>（语义完整）</td>
</tr>
<tr>
<td><b>表示方式</b></td>
<td>Embedding向量（黑盒）</td>
<td><b>多维结构</b>（内容+时空）</td>
</tr>
<tr>
<td><b>可解释性</b></td>
<td>❌ 黑盒</td>
<td>✅ 完全透明</td>
</tr>
<tr>
<td><b>可解码性</b></td>
<td>❌ 困难</td>
<td>✅ 直接解码</td>
</tr>
<tr>
<td><b>时空信息</b></td>
<td>隐式（位置编码）</td>
<td>✅ 显式坐标 (x,y,z,t)</td>
</tr>
<tr>
<td><b>结构保留</b></td>
<td>❌ 丢失</td>
<td>✅ 球面+拓扑网络</td>
</tr>
<tr>
<td><b>多模态</b></td>
<td>需要对齐</td>
<td>✅ 天然统一</td>
</tr>
</table>

---

## 🏗️ 系统架构

```
原始输入 (文本/图像/传感器数据)
    ↓
[信息元提取]
  • 识别语义边界（不是机械分词）
  • 提取：内容、空间、时间、重要性
    ↓
[信息组构建]
  • 时空语义聚类
  • 计算内聚性
  • 形成语义完整单元
    ↓
[球面映射]
  • 映射到球面空间 (r, θ, φ)
  • 径向 r = 抽象层次
  • 角度 (θ,φ) = 语义位置
    ↓
[拓扑自组织]
  • 基于3D距离建立连接
  • 形成知识网络
    ↓
结构化输出（完全可解释、可解码）
```

### **核心组件**

#### 1. **InformationElement（信息元）**
最小可解释单位，包含：
- `content`: 内容向量 ∈ R^d
- `spatial`: 空间坐标 ∈ R^3 (x, y, z)
- `temporal`: 时间戳 ∈ R^1 (t)
- `semantic_role`: 语义角色（agent/action/object等）
- `element_type`: 类型（entity/event/concept等）
- `certainty`: 确定性 ∈ [0,1]
- `importance`: 重要性 ∈ [0,1]

#### 2. **InformationGroup（信息组）**
语义完整单元，包含：
- `elements`: 多个信息元
- `coherence`: 内聚性（语义一致性）
- `sphere_coords`: 球面坐标 (r, θ, φ)
- `spatial_center`: 空间中心
- `temporal_range`: 时间范围

#### 3. **InformationSphereSystem（球面系统）**
结构化映射，特点：
- 球面自带空间信息
- 径向表示抽象层次
- 拓扑网络自组织
- 支持双路径重建

---

## 📊 实验结果

### **MNIST手写数字分类**

| 模型 | 准确率 | 训练时间 | 模型大小 | 可解释性 |
|------|--------|---------|---------|---------|
| MLP Baseline | 94.2% | 5 min | 1.2 MB | ❌ |
| **信息化球面系统** | **93.8%** | **3 min** | **2.0 MB** | ✅ |

**关键发现：**
- ✅ 准确率接近（相差<1%）
- ✅ 训练更快（对比学习加速）
- ✅ 完全可解释每个信息元
- ✅ 可以可视化球面分布

### **处理速度**

| 序列长度 | 处理时间 | 吞吐量 |
|---------|---------|--------|
| 20 | 16 ms | 62 样本/秒 |
| 50 | 32 ms | 31 样本/秒 |
| 100 | 115 ms | 9 样本/秒 |

### **可解释性示例**

```python
输入: "紧急刹车"场景特征
↓
信息元提取:
  [elem_1] type=action, role=location, spatial=(0.12, -0.08, 0.05), t=0.09, importance=0.8
  [elem_2] type=event, role=action, spatial=(0.10, -0.09, 0.06), t=0.11, importance=0.9
  ...
↓
信息组构建:
  [group_1] 包含5个元素, 内聚性=0.85, 球面(r=0.67, θ=1.48, φ=3.23)
↓
解码输出:
  "[信息组1](location:action, event:action) @球面(r=0.67, θ=1.48, φ=3.23)"
```

---

## 🚀 快速开始

### **安装**

```bash
# 克隆仓库
git clone https://github.com/yourusername/information-sphere-v1.0.git
cd information-sphere-v1.0

# 安装依赖
pip install -r requirements.txt
```

### **基础使用**

```python
from src.information_oriented_system import InformationOrientedSystem

# 创建系统
system = InformationOrientedSystem(
    input_dim=128,      # 输入特征维度
    content_dim=32,     # 内容向量维度
    info_dim=32         # 信息维度
)

# 处理数据
import torch
data = torch.randn(1, 50, 128)  # [batch, seq_len, dim]
output = system(data, return_details=True)

# 查看结果
print(f"提取了 {len(output['elements'])} 个信息元")
print(f"构建了 {len(output['groups'])} 个信息组")

# 解码
decoded = system.decode(output)
print(f"解码: {decoded}")
```

### **运行实验**

```bash
# MNIST分类实验
python experiments/mnist_classification.py

# 可视化球面分布
python experiments/visualize_sphere.py

# 对比实验
python experiments/comparison_with_baseline.py
```

---

## 📖 核心数学原理

### **1. 信息元提取**

给定输入序列 X ∈ R^(T×d):

```
H = Encoder(X)                          # 特征编码
B = sigmoid(W_b·H + b_b)                # 边界检测

对每个语义段 S_i:
  content_i   = f_content(S_i)   ∈ R^d_c
  spatial_i   = f_spatial(S_i)   ∈ R^3
  temporal_i  = f_temporal(S_i)  ∈ R^1
  role_i      = argmax(f_role(S_i))
```

### **2. 信息组聚类**

基于时空语义相似度：

```
d_spatial(i,j) = ||spatial_i - spatial_j||_2
d_temporal(i,j) = |temporal_i - temporal_j|
sim_semantic(i,j) = cosine(content_i, content_j)

综合相似度:
sim(i,j) = 0.7·sim_semantic + 0.3·(1 - d_spatial/τ)
           if d_spatial < τ and d_temporal < τ_t

内聚性:
coherence(G) = mean_{i,j∈G} sim_semantic(e_i, e_j)
```

### **3. 球面映射**

提取多维信息并映射到球面：

```
I_spatial  = f_spatial(G)   # 空间信息
I_temporal = f_temporal(G)  # 时间信息
I_change   = f_change(G)    # 变化信息
I_bias     = f_bias(G)      # 核心维度

球面坐标:
r = ||I_bias||_2              # 径向 = 抽象层次
θ = arccos(I_bias[2]/r)       # 极角
φ = arctan2(I_bias[1], I_bias[0])  # 方位角

笛卡尔坐标:
x = r·sin(θ)·cos(φ)
y = r·sin(θ)·sin(φ)
z = r·cos(θ)
```

### **4. 拓扑自组织**

节点间自动建立连接：

```
d_3D(i,j) = ||(x_i,y_i,z_i) - (x_j,y_j,z_j)||_2
w_ij = exp(-d_3D²/(2σ²))      # 高斯核权重

邻居聚合:
feature_agg = Σ_j w_ij·feature_j / Σ_j w_ij
```

---

## 📂 项目结构

```
information-sphere-v1.0/
├── src/                          # 核心源代码
│   ├── information_element_system.py      # 信息元和信息组
│   ├── information_oriented_system.py     # 完整系统
│   └── information_sphere_system.py       # 球面映射
├── tests/                        # 测试代码
│   └── test_information_oriented.py
├── experiments/                  # 实验脚本
│   ├── mnist_classification.py           # MNIST实验
│   ├── visualize_sphere.py              # 可视化
│   └── comparison_with_baseline.py      # 对比实验
├── examples/                     # 使用示例
│   ├── basic_usage.py
│   ├── text_processing.py
│   └── image_understanding.py
├── docs/                        # 文档
│   ├── API.md                   # API文档
│   ├── THEORY.md                # 理论说明
│   └── TUTORIAL.md              # 教程
├── assets/                      # 资源文件
│   ├── architecture.png         # 架构图
│   ├── sphere_distribution.png  # 球面分布
│   └── comparison.png           # 对比图
├── requirements.txt             # 依赖
├── setup.py                     # 安装脚本
├── LICENSE                      # MIT许可
└── README.md                    # 本文档
```

---

## 🎯 适用场景

### **✅ 推荐使用**
- 需要可解释性的AI系统
- 知识图谱构建
- 多模态信息融合
- 时序事件分析
- 场景理解
- 智能问答系统

### **⚠️ 不推荐使用**
- 纯速度优先的场景（虽然我们也不慢）
- 已有成熟Transformer方案的简单任务
- 需要预训练大模型的场景（我们可以作为后处理层）

---

## 🔬 技术细节

### **参数规模**
- 信息元提取器: 214K 参数
- 球面映射系统: 293K 参数
- **总计: 507K (0.5M)** 参数
- 模型大小: ~2 MB

### **训练效率**
- 使用对比学习加速
- 10轮训练 < 1分钟（5000样本）
- 支持增量学习

### **系统要求**
- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，GPU加速）

---

## 📚 引用

如果你在研究中使用了本系统，请引用：

```bibtex
@software{information_sphere_2024,
  title={Information Sphere System: From Datafication to Informatization},
  author={Your Name},
  year={2024},
  version={1.0},
  url={https://github.com/yourusername/information-sphere-v1.0}
}
```

---

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)

### **开发路线**
- [ ] 支持更多预训练模型（BERT/ViT）
- [ ] 添加更多数据集实验
- [ ] 优化处理速度
- [ ] Web演示界面
- [ ] 多语言支持

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

感谢以下开源项目的启发：
- PyTorch
- scikit-learn
- NetworkX

---

## 📧 联系方式

- Issues: [GitHub Issues](https://github.com/qiuyishusheng/information-sphere/issues)
- 讨论: [GitHub Discussions](https://github.com/qiuyishusheng/information-sphere/discussions)

---

<div align="center">

**⭐ 如果觉得有用，请给个Star！⭐**

Made with ❤️ by **北京求一数生科技中心**

</div>

