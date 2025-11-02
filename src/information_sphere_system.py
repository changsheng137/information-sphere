"""
完整的信息化球面系统
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心设计：
1. 数据信息化：
   - 球面自带空间信息
   - 解析：空间/时间/变化/偏向信息
   - 结合核心维度 → 球结构化

2. 解析方式：
   - 从核心维度触发
   - 双线逆向重建信息

3. 信息演化：
   - 信息在球面上积累
   - 拓扑连接形成
   - 涌现新知识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import math
from tqdm import tqdm


# ============================================================================
# 高效训练工具函数
# ============================================================================

def efficient_contrastive_train(system, train_data, train_labels, 
                               epochs=10, lr=0.001, batch_size=32):
    """
    高效训练 - 对比学习 + 监督学习
    
    为什么快速？
    1. 对比学习让编码器快速学会区分不同类别
    2. 小批量训练，每轮只需几秒
    3. 只比较邻近样本，避免O(N²)复杂度
    """
    system.train()
    optimizer = torch.optim.Adam(system.parameters(), lr=lr)
    
    num_samples = len(train_data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"⚡ 高效训练: {num_samples}样本, {epochs}轮, {batch_size}批大小")
    
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", ncols=100)
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            indices = perm[start:end]
            
            batch_data = train_data[indices].to(system.device)
            batch_labels = train_labels[indices].to(system.device)
            
            # 预测（训练时不用邻居，避免依赖）
            logits = system.predict(batch_data, use_neighbors=False)
            
            # 分类损失
            cls_loss = F.cross_entropy(logits, batch_labels)
            
            # 对比损失（让同类特征接近）
            info_dict = system.info_extractor(batch_data)
            bias_features = info_dict['bias']  # 核心维度最重要
            
            contrastive_loss = 0.0
            for i in range(len(batch_labels)):
                # 只比较邻近5个样本，O(N)复杂度
                for j in range(i+1, min(i+5, len(batch_labels))):
                    sim = F.cosine_similarity(bias_features[i:i+1], bias_features[j:j+1])
                    if batch_labels[i] == batch_labels[j]:
                        contrastive_loss += (1 - sim)  # 同类靠近
                    else:
                        contrastive_loss += F.relu(sim - 0.5)  # 异类远离
            
            if len(batch_labels) > 1:
                contrastive_loss /= len(batch_labels)
            
            # 总损失
            loss = cls_loss + 0.1 * contrastive_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1)
            acc = (pred == batch_labels).float().mean().item()
            epoch_acc += acc
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{acc:.2%}'})
        
        print(f"  → Loss: {epoch_loss/num_batches:.4f}, Acc: {epoch_acc/num_batches:.2%}")
    
    print("✓ 训练完成\n")
    system.eval()


@dataclass
class SpatialStructure:
    """空间信息化结构"""
    position: Tuple[float, float, float]  # 球坐标 (r, θ, φ)
    cartesian: Tuple[float, float, float]  # 笛卡尔坐标 (x, y, z)
    local_density: float  # 局部密度（周围节点数）
    geodesic_center: float  # 到球心的测地线距离
    spatial_coherence: float  # 空间一致性
    layer_depth: int  # 所在层次（0=核心, 5=外层）


@dataclass
class TemporalStructure:
    """时间信息化结构 - 参考trainer的三重时间"""
    observation_time: float  # 观察时间（创建时刻）
    access_time: float  # 最后访问时间
    evolution_time: float  # 演化时间（累计存在时间）
    temporal_stability: float  # 时间稳定性
    change_rate: float  # 变化速率
    temporal_coherence: float  # 时间一致性


@dataclass
class InformationNode:
    """信息节点 - 球面上的结构化信息"""
    node_id: str
    
    # === 空间信息化结构 ===
    spatial: SpatialStructure
    
    # === 时间信息化结构 ===
    temporal: TemporalStructure
    
    # === 原始数据的多维信息 ===
    spatial_info: torch.Tensor    # 空间信息编码
    temporal_info: torch.Tensor   # 时间信息编码
    change_info: torch.Tensor     # 变化信息编码
    bias_info: torch.Tensor       # 偏向信息（核心维度）
    
    # === 元信息 ===
    importance: float = 1.0
    abstraction_level: float = 0.0  # 抽象层次（半径相关）
    certainty: float = 0.5
    complexity: float = 0.5
    
    # === 演化信息 ===
    connections: List[str] = None
    access_count: int = 0
    evolution_stage: int = 0
    
    # === 分类信息 ===
    label: Optional[int] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []


class MultiDimensionInfoExtractor(nn.Module):
    """多维度信息提取器 - 从原始数据中提取结构化信息"""
    
    def __init__(self, input_dim=784, info_dim=32):
        super().__init__()
        
        # 1. 空间信息提取
        self.spatial_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, info_dim),
            nn.Tanh()
        )
        
        # 2. 时间信息提取（从数据的序列/演化模式）
        self.temporal_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, info_dim),
            nn.Tanh()
        )
        
        # 3. 变化信息提取（数据的动态特性）
        self.change_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, info_dim),
            nn.Sigmoid()  # 变化幅度 [0,1]
        )
        
        # 4. 偏向信息提取（核心维度）- 最重要
        self.bias_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, info_dim),
            nn.Tanh()
        )
        
    def forward(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取多维度信息"""
        return {
            'spatial': self.spatial_extractor(data),
            'temporal': self.temporal_extractor(data),
            'change': self.change_extractor(data),
            'bias': self.bias_extractor(data)  # 核心维度
        }


class CoreDimensionMapper(nn.Module):
    """
    真正的球面+球内三轴映射器
    
    设计理念：
    1. 球面(Surface): 拓扑映射真实世界的空间位置
       - spatial信息 → 球面坐标(θ, φ)
       - 保持空间邻近性
    
    2. 球内三轴(Interior Axes):
       - X轴: 时间维度 (temporal信息)
       - Y轴: 变化趋势维度 (change信息)
       - Z轴: 语义/含义维度 (bias/semantic信息)
    
    3. 球内位置 = 球面位置 + 三轴偏移
    """
    
    def __init__(self, info_dim=32):
        super().__init__()
        
        # === 球面拓扑映射：spatial → (θ, φ) ===
        self.surface_mapper = nn.Sequential(
            nn.Linear(info_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # (θ, φ)
        )
        
        # === 球内三轴编码 ===
        # X轴：时间维度
        self.temporal_axis_encoder = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1]: 负=过去, 0=现在, 正=未来
        )
        
        # Y轴：变化趋势
        self.change_axis_encoder = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1]: 负=减少, 0=稳定, 正=增加
        )
        
        # Z轴：语义/含义
        self.semantic_axis_encoder = nn.Sequential(
            nn.Linear(info_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1]: 负=抽象, 0=中性, 正=具体
        )
        
    def forward(self, info_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        真正的球面+球内映射
        
        输入：
        - spatial: 空间信息
        - temporal: 时间信息
        - change: 变化信息
        - bias: 核心语义信息
        
        输出：
        - theta, phi: 球面位置
        - x_axis: 时间轴坐标
        - y_axis: 变化轴坐标
        - z_axis: 语义轴坐标
        - interior_position: (x, y, z) 球内笛卡尔坐标
        """
        # 1. 空间信息 → 球面位置
        surface_raw = self.surface_mapper(info_dict['spatial'])
        theta = torch.sigmoid(surface_raw[:, 0]) * math.pi  # [0, π]
        phi = torch.sigmoid(surface_raw[:, 1]) * 2 * math.pi  # [0, 2π]
        
        # 2. 时间信息 → X轴
        x_axis = self.temporal_axis_encoder(info_dict['temporal']).squeeze(-1)
        
        # 3. 变化信息 → Y轴
        y_axis = self.change_axis_encoder(info_dict['change']).squeeze(-1)
        
        # 4. 语义信息 → Z轴
        z_axis = self.semantic_axis_encoder(info_dict['bias']).squeeze(-1)
        
        # 5. 计算球内笛卡尔坐标
        # 基础半径（0.5为球面基准）
        base_r = 0.5
        
        # 球面上的笛卡尔坐标
        sphere_x = base_r * torch.sin(theta) * torch.cos(phi)
        sphere_y = base_r * torch.sin(theta) * torch.sin(phi)
        sphere_z = base_r * torch.cos(theta)
        
        # 加上三轴偏移（扩展到球内）
        interior_x = sphere_x + 0.3 * x_axis  # 时间偏移
        interior_y = sphere_y + 0.3 * y_axis  # 变化偏移
        interior_z = sphere_z + 0.3 * z_axis  # 语义偏移
        
        # 计算实际半径和角度（用于兼容现有代码）
        r = torch.sqrt(interior_x**2 + interior_y**2 + interior_z**2)
        
        return {
            'theta': theta,
            'phi': phi,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'interior_x': interior_x,
            'interior_y': interior_y,
            'interior_z': interior_z,
            'r': r,  # 兼容性
            'abstraction': (z_axis + 1) / 2  # 兼容性：语义轴映射到抽象度
        }


class SpherePositionEncoder(nn.Module):
    """球体位置编码器 - 将球坐标编码为高维向量"""
    
    def __init__(self, encoding_dim=16):
        super().__init__()
        self.encoding_dim = encoding_dim
        # 简单但有效的编码：直接用MLP
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
    
    def forward(self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        输入: r, theta, phi (batch_size,)
        输出: position_encoding (batch_size, encoding_dim)
        """
        # 拼接三个坐标
        coords = torch.stack([r, theta, phi], dim=-1)  # (batch, 3)
        # 通过MLP编码
        position_emb = self.encoder(coords)  # (batch, encoding_dim)
        return position_emb


class SphereAwareReconstructor(nn.Module):
    """
    球体感知的重建器 - 基于球面+球内三轴的双线重建
    
    改进：
    1. 利用球内三轴信息（时间、变化、语义）
    2. 保持空间拓扑结构
    3. 增强表达能力
    """
    
    def __init__(self, info_dim=32, num_classes=10):
        super().__init__()
        
        # === 信息线1：从核心到表层（由内而外） ===
        self.path1_core = nn.Sequential(
            nn.Linear(info_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # === 信息线2：从聚合信息（由粗到细） ===
        self.path2_aggregate = nn.Sequential(
            nn.Linear(info_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # === 球内三轴信息融合 ===
        self.interior_axes_fusion = nn.Sequential(
            nn.Linear(3, 32),  # (x_axis, y_axis, z_axis)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # === 位置信息融合 ===
        self.position_fusion = nn.Sequential(
            nn.Linear(16, 32),  # 位置编码16维
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # === 四路融合（新增三轴信息） ===
        self.final_fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64, 256),  # path1 + path2 + axes + position
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # === 分类器 ===
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, info_dict: Dict[str, torch.Tensor], 
                position_dict: Dict[str, torch.Tensor],
                position_emb: torch.Tensor) -> torch.Tensor:
        """
        球体感知的双线重建 - 包含球内三轴
        
        Args:
            info_dict: 包含spatial, temporal, change, bias的信息字典
            position_dict: 包含theta, phi, x_axis, y_axis, z_axis等的位置信息
            position_emb: 球体位置编码 (batch, 16)
        """
        # 路径1：从核心维度
        path1 = self.path1_core(info_dict['bias'])
        
        # 路径2：从聚合信息
        global_info = torch.cat([
            info_dict['spatial'],
            info_dict['temporal'],
            info_dict['change'],
            info_dict['bias']
        ], dim=-1)
        path2 = self.path2_aggregate(global_info)
        
        # 新增：球内三轴信息
        if 'x_axis' in position_dict:
            axes_info = torch.stack([
                position_dict['x_axis'],
                position_dict['y_axis'],
                position_dict['z_axis']
            ], dim=-1)  # (batch, 3)
            axes_features = self.interior_axes_fusion(axes_info)
        else:
            # 兼容旧版本
            axes_features = torch.zeros(path1.shape[0], 64, device=path1.device)
        
        # 位置信息
        position_features = self.position_fusion(position_emb)
        
        # 四路融合（增加三轴）
        combined = torch.cat([path1, path2, axes_features, position_features], dim=-1)
        reconstructed = self.final_fusion(combined)
        
        # 分类
        logits = self.classifier(reconstructed)
        
        return logits


class InformationReconstructorDualPath(nn.Module):
    """双线逆向信息重建器 - 从核心维度触发的两条信息线（旧版，保留兼容）"""
    
    def __init__(self, info_dim=32, num_classes=10):
        super().__init__()
        
        # === 信息线1：从核心到表层（由内而外） ===
        self.path1_core_to_surface = nn.Sequential(
            nn.Linear(info_dim, 64),  # 核心偏向
            nn.ReLU(),
            nn.Linear(64, 128),       # 展开到中层
            nn.ReLU(),
            nn.Linear(128, 256),      # 展开到表层
            nn.ReLU()
        )
        
        # === 信息线2：从整体到细节（由粗到细） ===
        self.path2_global_to_local = nn.Sequential(
            nn.Linear(info_dim * 4, 128),  # 全局信息
            nn.ReLU(),
            nn.Linear(128, 256),           # 细化
            nn.ReLU()
        )
        
        # === 双线融合 ===
        self.dual_fusion = nn.Sequential(
            nn.Linear(512, 256),  # 融合两条信息线
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # === 最终分类器 ===
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, info_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """双线逆向重建"""
        # 信息线1：从核心维度出发
        path1 = self.path1_core_to_surface(info_dict['bias'])
        
        # 信息线2：从全局出发
        global_info = torch.cat([
            info_dict['spatial'],
            info_dict['temporal'],
            info_dict['change'],
            info_dict['bias']
        ], dim=-1)
        path2 = self.path2_global_to_local(global_info)
        
        # 融合两条线
        fused = torch.cat([path1, path2], dim=-1)
        reconstructed = self.dual_fusion(fused)
        
        # 分类
        logits = self.classifier(reconstructed)
        
        return logits


class InformationSphereSystem(nn.Module):
    """完整的信息化球面系统 - 球体几何感知版本"""
    
    def __init__(self, input_dim=784, info_dim=32, num_classes=10):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 信息提取器
        self.info_extractor = MultiDimensionInfoExtractor(input_dim, info_dim).to(self.device)
        
        # 2. 核心维度映射器
        self.core_mapper = CoreDimensionMapper(info_dim).to(self.device)
        
        # 3. 球体位置编码器
        self.position_encoder = SpherePositionEncoder(encoding_dim=16).to(self.device)
        
        # 4. 双线重建器（增强版，包含位置信息）
        self.reconstructor = SphereAwareReconstructor(info_dim, num_classes).to(self.device)
        
        # 5. 信息节点存储
        self.info_nodes: Dict[str, InformationNode] = {}
        self.node_count = 0
        
        # 6. 几何参数
        self.neighbor_radius = 0.3  # 邻居搜索半径
        self.gaussian_sigma = 0.15  # 高斯核标准差
        
    def add_information(self, data: torch.Tensor, label: int) -> str:
        """添加信息到球面"""
        with torch.no_grad():
            data = data.to(self.device)
            if data.dim() == 1:
                data = data.unsqueeze(0)  # 添加batch维度
            
            # 1. 提取多维度信息
            info_dict = self.info_extractor(data)
            
            # 2. 基于核心维度映射到球面
            position_dict = self.core_mapper(info_dict)
            
            # 3. 计算球面坐标
            r = position_dict['r'][0].item() if position_dict['r'].dim() > 0 else position_dict['r'].item()
            theta = position_dict['theta'][0].item() if position_dict['theta'].dim() > 0 else position_dict['theta'].item()
            phi = position_dict['phi'][0].item() if position_dict['phi'].dim() > 0 else position_dict['phi'].item()
            
            # 4. 转换为笛卡尔坐标
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            
            # 5. 检查相似信息
            similar = self._find_similar_information(info_dict, (r, theta, phi))
            
            if similar:
                # 信息融合
                similar_node = self.info_nodes[similar]
                similar_node.access_count += 1
                similar_node.importance += 0.1
                return similar
            else:
                # 创建新信息节点
                node_id = f"info_{self.node_count}"
                current_time = time.time()
                
                # 计算层次深度（基于半径）
                layer_depth = int(r * 6)  # 0-5层
                
                # 创建空间结构
                spatial_struct = SpatialStructure(
                    position=(r, theta, phi),
                    cartesian=(x, y, z),
                    local_density=0.0,  # 稍后计算
                    geodesic_center=r,
                    spatial_coherence=1.0,
                    layer_depth=layer_depth
                )
                
                # 创建时间结构
                temporal_struct = TemporalStructure(
                    observation_time=current_time,
                    access_time=current_time,
                    evolution_time=0.0,
                    temporal_stability=1.0,
                    change_rate=0.0,
                    temporal_coherence=1.0
                )
                
                node = InformationNode(
                    node_id=node_id,
                    spatial=spatial_struct,
                    temporal=temporal_struct,
                    spatial_info=info_dict['spatial'][0].cpu(),
                    temporal_info=info_dict['temporal'][0].cpu(),
                    change_info=info_dict['change'][0].cpu(),
                    bias_info=info_dict['bias'][0].cpu(),
                    abstraction_level=position_dict['abstraction'][0].item() if position_dict['abstraction'].dim() > 0 else position_dict['abstraction'].item(),
                    label=label
                )
                
                self.info_nodes[node_id] = node
                self.node_count += 1
                
                # 建立拓扑连接
                self._establish_topology(node_id)
                
                # 更新局部密度
                self._update_spatial_density(node_id)
                
                return node_id
    
    def predict(self, data: torch.Tensor, use_neighbors: bool = True) -> torch.Tensor:
        """
        球体几何感知的预测 - 使用球面+球内三轴
        
        Args:
            data: 输入数据
            use_neighbors: 是否使用邻居聚合（默认True）
        """
        data = data.to(self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # 1. 提取查询的信息
        info_dict = self.info_extractor(data)
        
        # 2. 映射到球体位置（球面+球内三轴）
        position_dict = self.core_mapper(info_dict)
        r = position_dict['r']
        theta = position_dict['theta']
        phi = position_dict['phi']
        
        # 3. 球体位置编码
        position_emb = self.position_encoder(r, theta, phi)
        
        # 4. 如果有节点且使用邻居聚合
        if use_neighbors and len(self.info_nodes) > 0:
            # 邻居聚合
            aggregated_info = self._aggregate_neighbors(
                position_dict, info_dict, k=5
            )
            # 使用聚合后的信息
            for key in info_dict:
                info_dict[key] = 0.5 * info_dict[key] + 0.5 * aggregated_info[key]
        
        # 5. 球体感知的双线重建（包含三轴信息）
        logits = self.reconstructor(info_dict, position_dict, position_emb)
        
        return logits
    
    def _aggregate_neighbors(self, position_dict: Dict, info_dict: Dict, k: int = 5) -> Dict[str, torch.Tensor]:
        """基于球体几何的邻居聚合
        
        Args:
            position_dict: 查询位置 {r, theta, phi}
            info_dict: 查询信息
            k: 最多聚合k个邻居
        
        Returns:
            聚合后的信息字典
        """
        if len(self.info_nodes) == 0:
            return info_dict
        
        # 获取查询位置
        r_q = position_dict['r'][0].item()
        theta_q = position_dict['theta'][0].item()
        phi_q = position_dict['phi'][0].item()
        
        # 转换为笛卡尔坐标
        x_q = r_q * math.sin(theta_q) * math.cos(phi_q)
        y_q = r_q * math.sin(theta_q) * math.sin(phi_q)
        z_q = r_q * math.cos(theta_q)
        
        # 计算所有节点的距离
        distances = []
        for node_id, node in self.info_nodes.items():
            r, theta, phi = node.spatial.position
            x = r * math.sin(theta) * math.cos(phi)
            y = r * math.sin(theta) * math.sin(phi)
            z = r * math.cos(theta)
            
            # 3D欧氏距离
            dist = math.sqrt((x_q - x)**2 + (y_q - y)**2 + (z_q - z)**2)
            distances.append((dist, node_id, node))
        
        # 按距离排序，取最近的k个
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        
        if not neighbors:
            return info_dict
        
        # 高斯核权重计算
        weights = []
        for dist, _, _ in neighbors:
            weight = math.exp(-(dist ** 2) / (2 * self.gaussian_sigma ** 2))
            weights.append(weight)
        
        # 归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(neighbors)] * len(neighbors)
        
        # 加权聚合
        aggregated = {
            'spatial': torch.zeros_like(info_dict['spatial']),
            'temporal': torch.zeros_like(info_dict['temporal']),
            'change': torch.zeros_like(info_dict['change']),
            'bias': torch.zeros_like(info_dict['bias']),
        }
        
        for (dist, node_id, node), weight in zip(neighbors, weights):
            aggregated['spatial'] += weight * node.spatial_info.to(self.device).unsqueeze(0)
            aggregated['temporal'] += weight * node.temporal_info.to(self.device).unsqueeze(0)
            aggregated['change'] += weight * node.change_info.to(self.device).unsqueeze(0)
            aggregated['bias'] += weight * node.bias_info.to(self.device).unsqueeze(0)
        
        return aggregated
    
    def _find_similar_information(self, info_dict: Dict[str, torch.Tensor], 
                                  position: Tuple, threshold: float = 0.95) -> Optional[str]:
        """查找相似信息（语义+空间）- 阈值提高到0.95避免过度合并"""
        if len(self.info_nodes) == 0:
            return None
        
        best_sim = -1.0
        best_node = None
        
        for node_id, node in self.info_nodes.items():
            # 语义相似度（核心维度最重要）
            bias_sim = F.cosine_similarity(
                info_dict['bias'].to(self.device),
                node.bias_info.to(self.device).unsqueeze(0)
            ).item()
            
            spatial_sim = F.cosine_similarity(
                info_dict['spatial'].to(self.device),
                node.spatial_info.to(self.device).unsqueeze(0)
            ).item()
            
            # 空间距离
            r1, theta1, phi1 = position
            r2, theta2, phi2 = node.spatial.position
            spatial_dist = math.sqrt((r1-r2)**2 + (theta1-theta2)**2 + (phi1-phi2)**2)
            
            # 综合相似度（核心维度权重最高）
            similarity = 0.5 * bias_sim + 0.2 * spatial_sim + 0.3 * (1 - spatial_dist/3.0)
            
            if similarity > best_sim and similarity > threshold:
                best_sim = similarity
                best_node = node_id
        
        return best_node
    
    def _establish_topology(self, new_node_id: str):
        """建立拓扑连接"""
        new_node = self.info_nodes[new_node_id]
        r1, theta1, phi1 = new_node.spatial.position
        
        for node_id, node in self.info_nodes.items():
            if node_id == new_node_id:
                continue
            
            # 计算球面距离
            r2, theta2, phi2 = node.spatial.position
            dist = math.sqrt((r1-r2)**2 + (theta1-theta2)**2 + (phi1-phi2)**2)
            
            if dist < 0.4:  # 邻近阈值
                new_node.connections.append(node_id)
                node.connections.append(new_node_id)
    
    def _update_spatial_density(self, node_id: str):
        """更新空间局部密度"""
        node = self.info_nodes[node_id]
        # 局部密度 = 连接数 / 最大可能连接数
        if len(self.info_nodes) > 1:
            node.spatial.local_density = len(node.connections) / (len(self.info_nodes) - 1)
    
    def update_temporal_states(self):
        """更新所有节点的时间状态"""
        current_time = time.time()
        for node in self.info_nodes.values():
            # 更新演化时间
            node.temporal.evolution_time = current_time - node.temporal.observation_time
            
            # 计算时间稳定性（基于访问频率）
            if node.access_count > 0:
                time_since_last = current_time - node.temporal.access_time
                node.temporal.temporal_stability = 1.0 / (1.0 + time_since_last / 3600.0)  # 小时衰减
            
            # 更新访问时间
            node.temporal.access_time = current_time
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if len(self.info_nodes) == 0:
            return {'total_nodes': 0}
        
        # 按层统计
        layers = {
            'inner': [],   # r < 0.5
            'middle': [],  # 0.5 <= r < 0.7
            'outer': []    # r >= 0.7
        }
        
        for node in self.info_nodes.values():
            r = node.spatial.position[0]
            if r < 0.5:
                layers['inner'].append(node)
            elif r < 0.7:
                layers['middle'].append(node)
            else:
                layers['outer'].append(node)
        
        return {
            'total_nodes': len(self.info_nodes),
            'layer_distribution': {
                'inner': len(layers['inner']),
                'middle': len(layers['middle']),
                'outer': len(layers['outer'])
            },
            'avg_connections': sum(len(n.connections) for n in self.info_nodes.values()) / len(self.info_nodes),
            'avg_abstraction': sum(n.abstraction_level for n in self.info_nodes.values()) / len(self.info_nodes)
        }


def demonstrate_information_sphere():
    """演示信息化球面系统"""
    print("="*70)
    print("信息化球面系统演示")
    print("="*70)
    
    # 创建系统
    system = InformationSphereSystem(input_dim=784, info_dim=32, num_classes=3)
    
    # 优化器
    optimizer = torch.optim.Adam(system.parameters(), lr=0.001)
    
    # 生成测试数据（3类，每类有明显特征）
    def make_data(class_id, num_samples):
        X = torch.randn(num_samples, 784)
        # 在特定位置添加强特征
        X[:, class_id*250:(class_id+1)*250] += 3.0
        y = torch.full((num_samples,), class_id, dtype=torch.long)
        return X, y
    
    # === 阶段1：训练（信息积累+双线学习） ===
    print("\n【阶段1】训练：信息积累+双线学习...")
    system.train()
    
    num_epochs = 10
    samples_per_class = 100
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # 进度条
        pbar = tqdm(range(3), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        
        for class_id in pbar:
            X, y = make_data(class_id, samples_per_class)
            for i in range(len(X)):
                # 前向
                logits = system.predict(X[i])
                loss = F.cross_entropy(logits, y[i:i+1].to(system.device))
                
                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                pred = logits.argmax().item()
                if pred == y[i].item():
                    correct += 1
                total += 1
                
                # 积累信息（最后一轮）
                if epoch == num_epochs - 1:
                    system.add_information(X[i], y[i].item())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{total_loss/total:.3f}',
                    'acc': f'{correct/total*100:.1f}%'
                })
        
        # 每个epoch后更新时间状态
        system.update_temporal_states()
        
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}: Loss={total_loss/total:.3f}, Acc={acc:.1f}%")
    
    # 统计
    stats = system.get_statistics()
    print(f"\n【统计】信息节点分布:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  层分布: 内层={stats['layer_distribution']['inner']}, "
          f"中层={stats['layer_distribution']['middle']}, "
          f"外层={stats['layer_distribution']['outer']}")
    print(f"  平均连接数: {stats['avg_connections']:.2f}")
    print(f"  平均抽象层次: {stats['avg_abstraction']:.3f}")
    
    # === 阶段2：双线重建测试 ===
    print("\n【阶段2】双线逆向重建测试...")
    
    system.eval()
    test_results = []
    
    for test_class in range(3):
        X_test, y_test = make_data(test_class, 50)
        correct = 0
        
        pbar = tqdm(range(len(X_test)), desc=f"测试类别{test_class}", ncols=100)
        with torch.no_grad():
            for i in pbar:
                logits = system.predict(X_test[i])
                pred = logits.argmax().item()
                if pred == y_test[i].item():
                    correct += 1
                pbar.set_postfix({'acc': f'{correct/(i+1)*100:.1f}%'})
        
        accuracy = correct / len(X_test) * 100
        test_results.append(accuracy)
        print(f"  类别{test_class}准确率: {accuracy:.1f}%")
    
    avg_accuracy = sum(test_results) / len(test_results)
    print(f"\n  平均测试准确率: {avg_accuracy:.1f}%")
    
    # === 阶段3：展示信息化的价值 ===
    print("\n【阶段3】信息化效果:")
    print(f"  ✓ 数据→多维信息: 空间/时间/变化/偏向")
    print(f"  ✓ 核心维度驱动: 偏向信息决定球面位置")
    print(f"  ✓ 拓扑自组织: {stats['total_nodes']}个节点, 平均{stats['avg_connections']:.1f}个连接")
    print(f"  ✓ 双线重建: 从核心维度双向解析")
    
    total_samples = num_epochs * samples_per_class * 3
    compression_ratio = total_samples / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
    print(f"  ✓ 质变基础: {total_samples}个样本→{stats['total_nodes']}个信息节点（{compression_ratio:.1f}:1压缩）")
    
    # 展示空间和时间结构
    print("\n【阶段4】空间和时间信息化结构:")
    sample_nodes = list(system.info_nodes.values())[:min(3, len(system.info_nodes))]
    for node in sample_nodes:
        print(f"\n  节点 {node.node_id}:")
        print(f"    空间结构:")
        print(f"      - 位置: r={node.spatial.position[0]:.3f}, 层次={node.spatial.layer_depth}")
        print(f"      - 局部密度: {node.spatial.local_density:.3f}")
        print(f"      - 空间一致性: {node.spatial.spatial_coherence:.3f}")
        print(f"    时间结构:")
        print(f"      - 演化时间: {node.temporal.evolution_time:.2f}秒")
        print(f"      - 时间稳定性: {node.temporal.temporal_stability:.3f}")
        print(f"      - 访问次数: {node.access_count}")
        print(f"    元信息:")
        print(f"      - 抽象层次: {node.abstraction_level:.3f}")
        print(f"      - 连接数: {len(node.connections)}")
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)


if __name__ == '__main__':
    demonstrate_information_sphere()

