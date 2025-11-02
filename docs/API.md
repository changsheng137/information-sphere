# API 文档

## 核心类

### InformationOrientedSystem

完整的信息化处理系统。

#### 初始化

```python
InformationOrientedSystem(
    input_dim: int = 128,           # 输入特征维度
    hidden_dim: int = 256,          # 隐藏层维度
    content_dim: int = 32,          # 内容向量维度
    info_dim: int = 32,             # 信息维度
    num_classes: int = 10,          # 分类数（如需要）
    spatial_threshold: float = 0.5, # 空间聚类阈值
    temporal_threshold: float = 1.0,# 时间聚类阈值
    semantic_threshold: float = 0.6,# 语义相似度阈值
    device: str = None              # 'cuda' 或 'cpu'
)
```

#### 方法

##### forward()

```python
def forward(
    self, 
    raw_input: torch.Tensor,        # [batch, seq_len, input_dim]
    context: Optional[Dict] = None, # 可选上下文
    return_details: bool = True     # 是否返回详细信息
) -> Dict[str, Any]
```

**返回值：**
```python
{
    'elements': List[InformationElement],  # 信息元列表
    'groups': List[InformationGroup],      # 信息组列表
    'sphere_nodes': List[InformationNode], # 球面节点
    'features': torch.Tensor,              # 聚合特征
    'sphere_coords': List[Tuple],          # 球面坐标
    'predictions': torch.Tensor,           # 预测（如有）
    'interpretable': bool,                 # 是否可解释
    'decodable': bool,                     # 是否可解码
    'statistics': Dict                     # 统计信息
}
```

##### decode()

```python
def decode(
    self, 
    output: Dict[str, Any]  # forward()的输出
) -> str                    # 解码后的语义描述
```

##### visualize_structure()

```python
def visualize_structure(
    self, 
    output: Dict[str, Any]  # forward()的输出
)
```

---

### InformationElement

信息元 - 最小可解释单位。

#### 属性

```python
content: torch.Tensor           # 内容向量 [d_content]
spatial: torch.Tensor           # 空间坐标 [3] (x, y, z)
temporal: torch.Tensor          # 时间戳 [1]
element_id: str                 # 唯一ID
element_type: ElementType       # 类型（entity/action/relation/...）
modality: Modality              # 模态（text/image/audio/...）
semantic_role: SemanticRole     # 语义角色（agent/action/object/...）
certainty: float                # 确定性 [0,1]
importance: float               # 重要性 [0,1]
```

#### 方法

```python
def to_tensor() -> torch.Tensor
    """转换为统一张量表示"""

def is_compatible(
    other: InformationElement,
    spatial_threshold: float = 0.5,
    temporal_threshold: float = 1.0
) -> bool
    """判断两个信息元是否可以组合"""

def get_spatial_position() -> np.ndarray
    """获取空间位置"""

def get_temporal_position() -> float
    """获取时间位置"""
```

---

### InformationGroup

信息组 - 语义完整单元。

#### 属性

```python
elements: List[InformationElement]  # 包含的信息元
group_id: str                        # 组ID
structure_type: str                  # 结构类型
coherence: float                     # 内聚性 [0,1]
semantic_summary: torch.Tensor       # 语义摘要
spatial_center: torch.Tensor         # 空间中心
spatial_radius: float                # 空间半径
temporal_range: Tuple[float, float]  # 时间范围
sphere_coords: Tuple[float, float, float]  # (r, θ, φ)
abstraction_level: float             # 抽象层次
```

#### 方法

```python
def aggregate() -> Dict[str, torch.Tensor]
    """
    聚合所有信息元
    返回: {
        'content': 平均内容,
        'spatial': 平均空间,
        'temporal': 平均时间,
        'variance': 内容方差
    }
    """

def to_tensor() -> torch.Tensor
    """转换为张量表示"""

def compute_statistics() -> Dict[str, float]
    """
    计算统计信息
    返回: {
        'num_elements': 元素数量,
        'coherence': 内聚性,
        'spatial_radius': 空间半径,
        'temporal_span': 时间跨度,
        'avg_importance': 平均重要性,
        'avg_certainty': 平均确定性
    }
    """
```

---

### InformationSphereSystem

球面结构化系统。

#### 初始化

```python
InformationSphereSystem(
    input_dim: int = 128,     # 输入维度
    info_dim: int = 32,       # 信息维度
    num_classes: int = 10     # 分类数
)
```

#### 方法

```python
def predict(
    self,
    data: torch.Tensor,       # 输入数据
    use_neighbors: bool = True # 是否使用邻居信息
) -> torch.Tensor             # 预测结果
```

```python
def add_information(
    self,
    data: torch.Tensor,
    label: Optional[int] = None
) -> InformationNode          # 添加的节点
```

---

## 枚举类型

### ElementType

信息元类型：
- `ENTITY`: 实体
- `ACTION`: 动作
- `RELATION`: 关系
- `STATE`: 状态
- `EVENT`: 事件
- `CONCEPT`: 概念

### SemanticRole

语义角色：
- `AGENT`: 施事者
- `ACTION`: 动作
- `OBJECT`: 受事者
- `LOCATION`: 位置
- `TIME`: 时间
- `STATE`: 状态
- `ATTRIBUTE`: 属性
- `RELATION`: 关系

### Modality

信息模态：
- `TEXT`: 文本
- `IMAGE`: 图像
- `AUDIO`: 音频
- `SENSOR`: 传感器数据
- `MULTI`: 多模态

---

## 工具函数

### efficient_contrastive_train()

高效训练函数。

```python
def efficient_contrastive_train(
    system: InformationSphereSystem,  # 要训练的系统
    train_data: torch.Tensor,         # 训练数据
    train_labels: torch.Tensor,       # 标签
    epochs: int = 10,                 # 训练轮数
    lr: float = 0.001,                # 学习率
    batch_size: int = 32              # 批大小
)
```

**特点：**
- 结合监督学习和对比学习
- 快速收敛（10轮<1分钟）
- 提升节点区分度

---

## 使用示例

### 基础使用

```python
from src.information_oriented_system import InformationOrientedSystem
import torch

# 初始化
system = InformationOrientedSystem(
    input_dim=128,
    content_dim=32
)

# 处理数据
data = torch.randn(1, 50, 128)
output = system(data, return_details=True)

# 查看结果
print(f"信息元: {len(output['elements'])}")
print(f"信息组: {len(output['groups'])}")

# 解码
decoded = system.decode(output)
print(decoded)
```

### 训练

```python
from src.information_sphere_system import efficient_contrastive_train

# 准备数据
train_data = torch.randn(1000, 50, 128)
train_labels = torch.randint(0, 10, (1000,))

# 训练
efficient_contrastive_train(
    system.sphere_mapper,
    train_data,
    train_labels,
    epochs=10,
    lr=0.001
)
```

### 批处理

```python
# 批量处理
batch_data = torch.randn(8, 50, 128)
output = system(batch_data, return_details=False)

print(f"处理了 {batch_data.shape[0]} 个样本")
```

---

## 性能指标

### 处理速度

| 序列长度 | 处理时间 | 吞吐量 |
|---------|---------|--------|
| 20 | ~16ms | 62 样本/秒 |
| 50 | ~32ms | 31 样本/秒 |
| 100 | ~115ms | 9 样本/秒 |

### 内存占用

- 模型大小: ~2 MB
- 推理内存: ~100 MB (batch_size=8)
- 训练内存: ~500 MB (batch_size=32)

### 参数规模

- 信息元提取器: 214K
- 球面映射系统: 293K
- **总计**: 507K (0.5M)

---

## 常见问题

### Q: 如何处理不同长度的序列？

A: 系统自动处理，会根据边界检测分割为语义段。

### Q: 如何加速处理？

A: 
1. 使用GPU（`device='cuda'`）
2. 增大batch_size
3. 设置`return_details=False`

### Q: 如何提升准确率？

A: 使用`efficient_contrastive_train()`进行训练。

### Q: 支持哪些模态？

A: 理论上支持所有模态，只需提供特征表示。已测试：文本特征、图像patch特征、传感器数据。

---

## 扩展开发

### 自定义信息元提取器

```python
class CustomExtractor(nn.Module):
    def forward(self, features):
        # 你的逻辑
        return elements

# 替换提取器
system.element_extractor = CustomExtractor()
```

### 自定义信息组构建器

```python
builder = InformationGroupBuilder(
    spatial_threshold=1.0,  # 调整阈值
    temporal_threshold=2.0,
    semantic_threshold=0.7
)
system.group_builder = builder
```

---

## 引用

```bibtex
@software{information_sphere_2024,
  title={Information Sphere System},
  author={Your Name},
  year={2024},
  version={1.0}
}
```

