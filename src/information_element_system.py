"""
信息元系统 - Information Element System

核心创新：从数据化到信息化的范式转变
- 信息元（InformationElement）：最小可解释单位
- 信息组（InformationGroup）：语义完整单元
- 绕过传统Token化，保留完整语义和时空结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum
import time


class SemanticRole(Enum):
    """语义角色"""
    AGENT = "agent"          # 施事者
    ACTION = "action"        # 动作
    OBJECT = "object"        # 受事者
    LOCATION = "location"    # 位置
    TIME = "time"           # 时间
    STATE = "state"         # 状态
    ATTRIBUTE = "attribute" # 属性
    RELATION = "relation"   # 关系


class ElementType(Enum):
    """信息元类型"""
    ENTITY = "entity"       # 实体
    ACTION = "action"       # 动作
    RELATION = "relation"   # 关系
    STATE = "state"         # 状态
    EVENT = "event"         # 事件
    CONCEPT = "concept"     # 概念


class Modality(Enum):
    """信息模态"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    SENSOR = "sensor"
    MULTI = "multi"


@dataclass
class InformationElement:
    """
    信息元 - 信息的原子单位
    
    核心革新：多维度显式信息结构
    - 每个维度独立、可解释、可验证
    - 不使用抽象向量，而是直接存储真实信息
    - 支持完全无损的逆向重构
    """
    
    # === 1. 空间信息（Spatial Information）===
    spatial_info: Dict[str, Any]  # {
        # 'coordinates': (x, y, z) - 真实坐标
        # 'coordinate_type': 'image_pixel' | 'geographic' | 'abstract'
        # 'region_size': (w, h, d) - 区域大小
        # 'reference_frame': 坐标系参考
    # }
    
    # === 2. 时间信息（Temporal Information）===
    temporal_info: Dict[str, Any]  # {
        # 'timestamp': t - 真实时间戳或序列位置
        # 'time_type': 'absolute' | 'relative' | 'sequence'
        # 'duration': Δt - 持续时间
        # 'sequence_index': 在序列中的位置
    # }
    
    # === 3. 变化信息（Change Information）===
    change_info: Dict[str, Any]  # {
        # 'trend': Δ - 变化趋势值
        # 'rate': dΔ/dt - 变化率
        # 'change_type': 'increasing' | 'decreasing' | 'stable'
        # 'magnitude': 变化幅度
    # }
    
    # === 4. 语义信息（Semantic Information）===
    semantic_info: Dict[str, Any]  # {
        # 'role': SemanticRole - 语义角色
        # 'type': ElementType - 元素类型
        # 'abstraction_level': [0-1] - 抽象程度
        # 'category': 类别标签
    # }
    
    # === 5. 内容信息（Content Information）===
    content_info: Dict[str, Any]  # {
        # 'modality': Modality - 模态
        # 'raw_data': 原始数据（像素值、文本、传感器读数等）
        # 'parsed_data': 解析后的结构化数据
        # 'statistics': 统计信息（均值、方差等）
    # }
    
    # === 元信息 ===
    element_id: str                 # 唯一标识
    certainty: float = 1.0          # 确定性 [0,1]
    importance: float = 0.5         # 重要性 [0,1]
    
    # === 关系属性 ===
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)
    
    # === 附加信息 ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # === 性能优化：缓存 ===
    _content_cache: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    
    # === 兼容性：保留旧接口（动态计算+缓存）===
    @property
    def content(self) -> torch.Tensor:
        """
        内容向量（从content_info的raw_data提取压缩表示）
        
        优化：使用缓存避免重复计算
        
        策略：将raw_data压缩为固定维度（128维）
        - 如果raw_data维度>128：平均池化
        - 如果raw_data维度<128：零填充
        """
        # 使用缓存
        if self._content_cache is not None:
            return self._content_cache
        
        target_dim = 128  # 固定目标维度
        
        if 'raw_data' in self.content_info:
            data = self.content_info['raw_data']
            
            # 转换为tensor
            if isinstance(data, torch.Tensor):
                tensor_data = data.flatten()
            elif isinstance(data, (list, np.ndarray)):
                tensor_data = torch.tensor(data, dtype=torch.float32).flatten()
            else:
                return torch.zeros(target_dim)
            
            current_dim = tensor_data.shape[0]
            
            if current_dim == target_dim:
                result = tensor_data
            elif current_dim > target_dim:
                # 平均池化压缩
                pool_size = current_dim // target_dim
                remainder = current_dim % target_dim
                
                # 简单的分组平均
                compressed = []
                for i in range(target_dim):
                    start = i * pool_size
                    end = start + pool_size
                    if i == target_dim - 1:  # 最后一组包含余数
                        end = current_dim
                    compressed.append(tensor_data[start:end].mean())
                
                result = torch.tensor(compressed, dtype=torch.float32)
            else:
                # 零填充
                padded = torch.zeros(target_dim, dtype=torch.float32)
                padded[:current_dim] = tensor_data
                result = padded
            
            # 缓存结果
            self._content_cache = result
            return result
        
        # 默认值
        result = torch.zeros(target_dim)
        self._content_cache = result
        return result
    
    @property
    def spatial(self) -> torch.Tensor:
        """空间坐标向量"""
        coords = self.spatial_info.get('coordinates', (0, 0, 0))
        return torch.tensor(coords, dtype=torch.float32)
    
    @property
    def temporal(self) -> torch.Tensor:
        """时间向量"""
        t = self.temporal_info.get('timestamp', 0)
        return torch.tensor([t], dtype=torch.float32)
    
    @property
    def modality(self) -> Modality:
        """模态"""
        return self.content_info.get('modality', Modality.TEXT)
    
    @property
    def element_type(self) -> ElementType:
        """元素类型"""
        return self.semantic_info.get('type', ElementType.ENTITY)
    
    @property
    def semantic_role(self) -> SemanticRole:
        """语义角色"""
        return self.semantic_info.get('role', SemanticRole.OBJECT)
    
    def to_tensor(self) -> torch.Tensor:
        """转换为统一张量表示"""
        return torch.cat([
            self.content.flatten(),
            self.spatial.flatten(),
            self.temporal.flatten(),
            torch.tensor([self.certainty, self.importance])
        ])
    
    def is_compatible(self, other: 'InformationElement', 
                     spatial_threshold: float = 0.5,
                     temporal_threshold: float = 1.0) -> bool:
        """判断两个信息元是否可以组合"""
        # 时空接近
        spatial_dist = torch.norm(self.spatial - other.spatial).item()
        temporal_dist = abs(self.temporal.item() - other.temporal.item())
        
        # 语义兼容（同一模态且类型兼容）
        modality_match = self.modality == other.modality
        
        return (spatial_dist < spatial_threshold and 
                temporal_dist < temporal_threshold and
                modality_match)
    
    def get_spatial_position(self) -> np.ndarray:
        """获取空间位置"""
        return self.spatial.cpu().numpy()
    
    def get_temporal_position(self) -> float:
        """获取时间位置"""
        return self.temporal.item()


@dataclass
class InformationGroup:
    """
    信息组 - 由多个信息元组成的语义单元
    
    核心设计：
    - 语义完整性
    - 内在结构
    - 时空连续性
    - 可映射到球面
    """
    
    # === 组成元素 ===
    elements: List[InformationElement]  # 包含的信息元
    group_id: str                        # 组ID
    
    # === 结构属性 ===
    structure_type: str = 'graph'        # 'linear', 'hierarchical', 'graph'
    coherence: float = 0.0               # 内聚性 [0,1]
    
    # === 语义属性 ===
    semantic_summary: Optional[torch.Tensor] = None  # 语义摘要向量
    key_elements: List[str] = field(default_factory=list)  # 关键元素IDs
    
    # === 时空范围 ===
    spatial_center: Optional[torch.Tensor] = None    # 空间中心
    spatial_radius: float = 0.0                      # 空间半径
    temporal_range: Tuple[float, float] = (0.0, 0.0) # 时间范围
    
    # === 球面映射 ===
    sphere_coords: Optional[Tuple[float, float, float]] = None  # (r, θ, φ)
    abstraction_level: float = 0.0  # 抽象层次
    
    # === 附加信息 ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """聚合所有信息元"""
        if not self.elements:
            raise ValueError("信息组为空，无法聚合")
        
        all_content = torch.stack([e.content for e in self.elements])
        all_spatial = torch.stack([e.spatial for e in self.elements])
        all_temporal = torch.stack([e.temporal for e in self.elements])
        
        # 修复std计算问题
        if len(self.elements) > 1:
            variance = all_content.std(dim=0)
        else:
            variance = torch.zeros_like(all_content[0])
        
        return {
            'content': all_content.mean(dim=0),
            'spatial': all_spatial.mean(dim=0),
            'temporal': all_temporal.mean(dim=0),
            'variance': variance,
            'content_all': all_content
        }
    
    def to_tensor(self) -> torch.Tensor:
        """转换为张量表示"""
        agg = self.aggregate()
        return torch.cat([
            agg['content'].flatten(),
            agg['spatial'].flatten(),
            agg['temporal'].flatten(),
            agg['variance'].flatten()
        ])
    
    def compute_statistics(self) -> Dict[str, float]:
        """计算信息组统计信息"""
        return {
            'num_elements': len(self.elements),
            'coherence': self.coherence,
            'spatial_radius': self.spatial_radius,
            'temporal_span': self.temporal_range[1] - self.temporal_range[0],
            'avg_importance': np.mean([e.importance for e in self.elements]),
            'avg_certainty': np.mean([e.certainty for e in self.elements])
        }


class InformationElementExtractor(nn.Module):
    """
    信息元提取器 - 从原始输入直接提取信息元
    
    核心功能：
    - 绕过传统tokenization
    - 识别语义边界
    - 提取多维信息
    - 分配语义角色
    """
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 content_dim: int = 32,
                 num_roles: int = 8,
                 num_types: int = 6):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim
        
        # === 特征编码器 ===
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === 语义边界检测器 ===
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # === 信息元分解器 ===
        self.content_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, content_dim)
        )
        
        self.spatial_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # (x, y, z)
        )
        
        self.temporal_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # timestamp
        )
        
        # === 属性预测器 ===
        self.certainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # === 语义角色分类器 ===
        self.role_classifier = nn.Linear(hidden_dim, num_roles)
        self.type_classifier = nn.Linear(hidden_dim, num_types)
        
    def forward(self, 
                features: torch.Tensor,
                context: Optional[Dict] = None) -> List[InformationElement]:
        """
        从特征直接提取显式信息元
        
        核心革新：不使用神经网络预测，而是直接计算真实信息
        
        Args:
            features: [seq_len, input_dim] - 原始输入特征
            context: 可选的上下文信息
            
        Returns:
            List[InformationElement]: 提取的信息元列表
        """
        if features.dim() == 3:
            features = features.squeeze(0)  # [seq_len, input_dim]
        elif features.dim() == 1:
            features = features.unsqueeze(0)  # [1, input_dim]
        
        seq_len, input_dim = features.shape
        device = features.device
        
        # 直接按固定粒度分割（之后可以改成基于内容的智能分割）
        segment_size = 4  # 每4个时间步作为一个信息元
        elements_list = []
        
        for seg_idx in range(0, seq_len, segment_size):
            seg_end = min(seg_idx + segment_size, seq_len)
            segment_data = features[seg_idx:seg_end]  # [segment_size, input_dim]
            
            # 直接从原始数据计算各维度信息
            element = self._create_element_from_raw_data(
                segment_data=segment_data,
                seg_idx=seg_idx,
                seq_len=seq_len,
                device=device,
                context=context
            )
            elements_list.append(element)
        
        return elements_list
    
    def _segment_by_boundaries(self, 
                                encoded: torch.Tensor, 
                                boundary_scores: torch.Tensor,
                                threshold: float = 0.5) -> List[torch.Tensor]:
        """基于边界分数分割序列"""
        seq_len = encoded.shape[0]
        
        # 找到边界位置
        boundary_indices = torch.where(boundary_scores > threshold)[0].cpu().numpy()
        
        if len(boundary_indices) == 0:
            # 没有明显边界，按固定长度分割
            segment_size = max(1, seq_len // 5)
            segments = [encoded[i:i+segment_size].mean(dim=0) 
                       for i in range(0, seq_len, segment_size)]
        else:
            # 按边界分割
            segments = []
            start = 0
            for end in boundary_indices:
                if end > start:
                    segments.append(encoded[start:end].mean(dim=0))
                start = end
            if start < seq_len:
                segments.append(encoded[start:].mean(dim=0))
        
        return segments if segments else [encoded.mean(dim=0)]
    
    def _create_element_from_raw_data(self,
                                      segment_data: torch.Tensor,
                                      seg_idx: int,
                                      seq_len: int,
                                      device: str,
                                      context: Optional[Dict] = None) -> InformationElement:
        """
        从原始数据段直接计算信息元的各维度信息
        
        核心：不用神经网络预测，而是直接计算真实信息
        """
        # segment_data: [segment_size, input_dim]
        segment_size = segment_data.shape[0]
        
        # === 优化：批量计算所有统计量（减少GPU-CPU同步）===
        # 一次性计算所有需要的统计信息
        with torch.no_grad():
            stats = torch.stack([
                segment_data.mean(),
                segment_data.std(),
                segment_data.max(),
                segment_data.min()
            ]).cpu()  # 一次性传输到CPU
        
        mean_val, std_val, max_val, min_val = stats.tolist()
        
        # 计算变化趋势
        if segment_size > 1:
            values = segment_data.mean(dim=1)
            trend = (values[-1] - values[0]).item()
            rate = trend / segment_size
            
            if trend > 0.01:
                change_type = 'increasing'
            elif trend < -0.01:
                change_type = 'decreasing'
            else:
                change_type = 'stable'
        else:
            trend = 0.0
            rate = 0.0
            change_type = 'stable'
        
        # === 1. 空间信息（Spatial Information）===
        row_idx = seg_idx  # 行索引（时间步）
        col_idx = mean_val  # 列索引（内容平均值）
        depth = std_val  # 深度（内容方差）
        
        spatial_info = {
            'coordinates': (float(row_idx), float(col_idx * 10), float(depth * 10)),
            'coordinate_type': 'sequence_mapping',
            'region_size': (segment_size, self.input_dim, 1),
            'reference_frame': 'input_sequence'
        }
        
        # === 2. 时间信息（Temporal Information）===
        temporal_info = {
            'timestamp': seg_idx / seq_len,
            'time_type': 'sequence',
            'duration': segment_size / seq_len,
            'sequence_index': seg_idx
        }
        
        # === 3. 变化信息（Change Information）===
        change_info = {
            'trend': float(trend),
            'rate': float(rate),
            'change_type': change_type,
            'magnitude': abs(float(trend))
        }
        
        # === 4. 语义信息（Semantic Information）===
        
        # 简单的语义分类规则
        if std_val > 0.3:
            sem_type = ElementType.EVENT
            sem_role = SemanticRole.ACTION
        elif mean_val > 0.5:
            sem_type = ElementType.ENTITY
            sem_role = SemanticRole.OBJECT
        else:
            sem_type = ElementType.STATE
            sem_role = SemanticRole.ATTRIBUTE
        
        semantic_info = {
            'role': sem_role,
            'type': sem_type,
            'abstraction_level': std_val,  # 方差越大越抽象
            'category': f'{sem_type.value}_{sem_role.value}'
        }
        
        # === 5. 内容信息（Content Information）===
        # 优化：保持tensor格式，延迟转换到真正需要时
        content_info = {
            'modality': Modality.IMAGE,
            'raw_data': segment_data.detach(),  # 保持tensor格式（快！）
            'parsed_data': {
                'mean': float(mean_val),
                'std': float(std_val),
                'max': float(max_val),  # 使用已计算的值
                'min': float(min_val)   # 使用已计算的值
            },
            'statistics': {
                'mean': float(mean_val),
                'std': float(std_val),
                'non_zero_ratio': float((segment_data != 0).float().mean().item())
            }
        }
        
        # === 计算确定性和重要性 ===
        certainty = 1.0 - std_val  # 方差小=确定性高
        importance = mean_val  # 均值大=重要性高
        
        # 创建信息元
        element = InformationElement(
            spatial_info=spatial_info,
            temporal_info=temporal_info,
            change_info=change_info,
            semantic_info=semantic_info,
            content_info=content_info,
            element_id=f"elem_{seg_idx}",
            certainty=float(max(0.0, min(1.0, certainty))),
            importance=float(max(0.0, min(1.0, importance)))
        )
        
        return element
    
    def _create_element_from_segment(self,
                                     seg_features: torch.Tensor,
                                     seg_idx: int,
                                     batch_idx: int = 0,
                                     device: str = 'cpu') -> InformationElement:
        """从段特征创建信息元"""
        # 确保是1D张量
        if seg_features.dim() == 0:
            seg_features = seg_features.unsqueeze(0)
        if seg_features.dim() > 1:
            seg_features = seg_features.flatten()
        
        # 如果维度不匹配，调整
        if seg_features.shape[0] != self.hidden_dim:
            # 插值或截断到正确维度
            if seg_features.shape[0] < self.hidden_dim:
                padding = torch.zeros(self.hidden_dim - seg_features.shape[0], device=device)
                seg_features = torch.cat([seg_features, padding])
            else:
                seg_features = seg_features[:self.hidden_dim]
        
        seg_features = seg_features.unsqueeze(0)  # [1, hidden_dim]
        
        # 提取各维度
        content = self.content_extractor(seg_features).squeeze(0)  # [content_dim]
        spatial = self.spatial_extractor(seg_features).squeeze(0)   # [3]
        temporal = self.temporal_extractor(seg_features).squeeze(0) # [1]
        
        # 预测属性
        certainty = self.certainty_predictor(seg_features).squeeze().item()
        importance = self.importance_predictor(seg_features).squeeze().item()
        
        # 分类角色和类型
        role_logits = self.role_classifier(seg_features).squeeze(0)
        type_logits = self.type_classifier(seg_features).squeeze(0)
        
        role_idx = role_logits.argmax().item()
        type_idx = type_logits.argmax().item()
        
        # 映射到枚举
        roles = list(SemanticRole)
        types = list(ElementType)
        semantic_role = roles[role_idx % len(roles)]
        element_type = types[type_idx % len(types)]
        
        # 创建信息元
        element = InformationElement(
            content=content.detach(),
            spatial=spatial.detach(),
            temporal=temporal.detach(),
            element_id=f"elem_{batch_idx}_{seg_idx}_{int(time.time()*1000000) % 1000000}",
            element_type=element_type,
            modality=Modality.TEXT,
            semantic_role=semantic_role,
            certainty=certainty,
            importance=importance,
            metadata={'segment_index': seg_idx, 'batch_index': batch_idx}
        )
        
        return element


class InformationGroupBuilder:
    """
    信息组构建器 - 将信息元组合成信息组
    
    核心功能：
    - 时空语义聚类
    - 计算内聚性
    - 识别关键元素
    - 建立组间关系
    """
    
    def __init__(self,
                 spatial_threshold: float = 2.0,  # 增大阈值，更容易聚类
                 temporal_threshold: float = 5.0,
                 semantic_threshold: float = 0.3,  # 降低阈值，更容易聚类
                 min_group_size: int = 1,
                 max_group_size: int = 10):
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.semantic_threshold = semantic_threshold
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
    
    def build_groups(self, elements: List[InformationElement]) -> List[InformationGroup]:
        """构建信息组"""
        if not elements:
            return []
        
        # 1. 基于时空和语义聚类
        clusters = self._cluster_elements(elements)
        
        # 2. 为每个聚类构建信息组
        groups = []
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) >= self.min_group_size:
                group = self._create_group(cluster, cluster_idx)
                groups.append(group)
        
        # 3. 建立组间关系
        groups = self._link_groups(groups)
        
        return groups
    
    def _cluster_elements(self, elements: List[InformationElement]) -> List[List[InformationElement]]:
        """基于时空语义聚类 - 优化：向量化计算"""
        n = len(elements)
        
        if n == 0:
            return []
        if n == 1:
            return [elements]
        
        # 优化：批量提取所有特征
        spatials = torch.stack([e.spatial for e in elements])  # [n, 3]
        temporals = torch.stack([e.temporal for e in elements])  # [n, 1]
        contents = torch.stack([e.content for e in elements])  # [n, 128]
        
        # 批量计算空间距离矩阵
        spatial_dists = torch.cdist(spatials, spatials, p=2)  # [n, n]
        
        # 批量计算时间距离矩阵
        temporal_dists = torch.abs(temporals - temporals.T)  # [n, n]
        
        # 批量计算语义相似度矩阵
        # cosine_similarity: [n, 1, 128] vs [1, n, 128] -> [n, n]
        semantic_sim = F.cosine_similarity(
            contents.unsqueeze(1),  # [n, 1, 128]
            contents.unsqueeze(0),  # [1, n, 128]
            dim=2
        )  # [n, n]
        
        # 构建相似度矩阵
        # 只保留时空都满足阈值的
        time_space_mask = (spatial_dists < self.spatial_threshold) & (temporal_dists < self.temporal_threshold)
        
        # 综合相似度
        similarity = torch.where(
            time_space_mask,
            0.7 * semantic_sim + 0.3 * (1.0 - spatial_dists / self.spatial_threshold).clamp(min=0),
            torch.zeros_like(semantic_sim)
        )
        
        # 只保留上三角（避免重复）
        similarity = torch.triu(similarity, diagonal=1)
        similarity = similarity + similarity.T  # 对称化
        
        # 简单的贪心聚类
        clusters = []
        used = set()
        
        for i in range(n):
            if i in used:
                continue
            
            cluster = [elements[i]]
            used.add(i)
            
            # 找相似的元素
            for j in range(i+1, n):
                if j not in used and similarity[i, j] > self.semantic_threshold:
                    if len(cluster) < self.max_group_size:
                        cluster.append(elements[j])
                        used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _create_group(self, elements: List[InformationElement], group_idx: int) -> InformationGroup:
        """创建信息组"""
        # 计算空间中心和范围
        all_spatial = torch.stack([e.spatial for e in elements])
        spatial_center = all_spatial.mean(dim=0)
        spatial_radius = torch.norm(all_spatial - spatial_center, dim=1).max().item()
        
        # 计算时间范围
        all_temporal = [e.temporal.item() for e in elements]
        temporal_range = (min(all_temporal), max(all_temporal))
        
        # 计算语义摘要
        all_content = torch.stack([e.content for e in elements])
        semantic_summary = all_content.mean(dim=0)
        
        # 计算内聚性
        coherence = self._compute_coherence(elements)
        
        # 识别关键元素（重要性最高的）
        importances = [(e.element_id, e.importance) for e in elements]
        importances.sort(key=lambda x: x[1], reverse=True)
        key_elements = [elem_id for elem_id, _ in importances[:min(3, len(importances))]]
        
        return InformationGroup(
            elements=elements,
            group_id=f"group_{group_idx}_{int(time.time()*1000000) % 1000000}",
            structure_type='graph',
            coherence=coherence,
            semantic_summary=semantic_summary,
            key_elements=key_elements,
            spatial_center=spatial_center,
            spatial_radius=spatial_radius,
            temporal_range=temporal_range,
            metadata={'cluster_index': group_idx}
        )
    
    def _compute_coherence(self, elements: List[InformationElement]) -> float:
        """计算信息组内聚性"""
        if len(elements) <= 1:
            return 1.0
        
        # 计算内部相似度
        similarities = []
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                if elements[i].content.shape == elements[j].content.shape:
                    sim = F.cosine_similarity(
                        elements[i].content.unsqueeze(0),
                        elements[j].content.unsqueeze(0)
                    ).item()
                    similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _link_groups(self, groups: List[InformationGroup]) -> List[InformationGroup]:
        """建立组间关系（基于时空接近性）"""
        # 简单实现：标记在时空上接近的组
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                # 检查时空接近性
                if group1.spatial_center is not None and group2.spatial_center is not None:
                    spatial_dist = torch.norm(group1.spatial_center - group2.spatial_center).item()
                    
                    t1_center = (group1.temporal_range[0] + group1.temporal_range[1]) / 2
                    t2_center = (group2.temporal_range[0] + group2.temporal_range[1]) / 2
                    temporal_dist = abs(t1_center - t2_center)
                    
                    if spatial_dist < 1.0 and temporal_dist < 2.0:
                        # 记录关系
                        group1.metadata.setdefault('related_groups', []).append(group2.group_id)
                        group2.metadata.setdefault('related_groups', []).append(group1.group_id)
        
        return groups


def print_element(element: InformationElement, indent: int = 0):
    """打印信息元信息"""
    prefix = "  " * indent
    print(f"{prefix}信息元 {element.element_id}:")
    print(f"{prefix}  类型: {element.element_type.value}, 角色: {element.semantic_role.value}")
    print(f"{prefix}  空间: {element.spatial.cpu().numpy()}")
    print(f"{prefix}  时间: {element.temporal.item():.4f}")
    print(f"{prefix}  确定性: {element.certainty:.3f}, 重要性: {element.importance:.3f}")


def print_group(group: InformationGroup, indent: int = 0):
    """打印信息组信息"""
    prefix = "  " * indent
    print(f"{prefix}信息组 {group.group_id}:")
    print(f"{prefix}  元素数量: {len(group.elements)}")
    print(f"{prefix}  内聚性: {group.coherence:.3f}")
    print(f"{prefix}  空间中心: {group.spatial_center.cpu().numpy() if group.spatial_center is not None else 'N/A'}")
    print(f"{prefix}  时间范围: [{group.temporal_range[0]:.4f}, {group.temporal_range[1]:.4f}]")
    if group.sphere_coords:
        print(f"{prefix}  球面坐标: r={group.sphere_coords[0]:.3f}, θ={group.sphere_coords[1]:.3f}, φ={group.sphere_coords[2]:.3f}")
    stats = group.compute_statistics()
    print(f"{prefix}  统计: 平均重要性={stats['avg_importance']:.3f}, 平均确定性={stats['avg_certainty']:.3f}")

