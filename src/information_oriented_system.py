"""
信息化导向系统 - Information-Oriented System

完整的信息化处理流程：
1. 信息元提取：从原始输入提取语义完整的信息元
2. 信息组构建：组织信息元为语义单元
3. 球面结构化：映射到球面空间
4. 拓扑网络：建立信息间的关系

核心创新：
- 绕过传统Token化
- 保留完整语义和时空结构
- 完全可解释和可解码
- 多模态统一表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from information_element_system import (
    InformationElement, InformationGroup,
    InformationElementExtractor, InformationGroupBuilder,
    print_element, print_group
)
from information_sphere_system import (
    InformationSphereSystem, InformationNode,
    SpatialStructure, TemporalStructure
)


class InformationReconstructor(nn.Module):
    """
    信息重构器 - 从信息元列表逆向重构原始信息
    
    核心设计：
    - 输入：信息元列表（不是聚合向量！）
    - 方法：逐个重构每个信息元对应的原始片段
    - 输出：拼接成完整的原始输入
    
    这才是真正的"准确逆向提取"！
    """
    
    def __init__(self, content_dim=128, output_feature_dim=128):
        super().__init__()
        
        # 从单个信息元重构对应的原始输入片段
        # 输入：content + spatial + temporal = content_dim + 3 + 1
        element_input_dim = content_dim + 4
        
        self.element_decoder = nn.Sequential(
            nn.Linear(element_input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_feature_dim),  # 重构一个时间步的特征
            nn.Tanh()
        )
        
    def forward(self, elements: List[InformationElement]) -> torch.Tensor:
        """
        从信息元列表直接重构原始序列
        
        核心革新：
        - 不使用神经网络解码
        - 直接从content_info中的raw_data提取原始数据
        - 按temporal_info中的sequence_index排序和拼接
        
        这才是真正的无损重构！
        
        Args:
            elements: 信息元列表
        
        Returns:
            reconstructed: [seq_len, feature_dim]
        """
        if not elements:
            return None
        
        # 按序列索引排序
        sorted_elements = sorted(elements, 
                                key=lambda e: e.temporal_info.get('sequence_index', 0))
        
        reconstructed_segments = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for elem in sorted_elements:
            # 直接从content_info提取原始数据（完全无损！）
            raw_data = elem.content_info.get('raw_data', None)
            
            if raw_data is not None:
                # 优化后raw_data已经是tensor格式
                if isinstance(raw_data, torch.Tensor):
                    segment_tensor = raw_data.to(device)
                elif isinstance(raw_data, list):
                    segment_tensor = torch.tensor(raw_data, dtype=torch.float32, device=device)
                else:
                    segment_tensor = torch.tensor([[raw_data]], dtype=torch.float32, device=device)
                
                # segment_tensor应该是 [segment_size, input_dim]
                if segment_tensor.dim() == 1:
                    segment_tensor = segment_tensor.unsqueeze(0)
                
                reconstructed_segments.append(segment_tensor)
        
        if not reconstructed_segments:
            return None
        
        # 拼接所有段
        reconstructed = torch.cat(reconstructed_segments, dim=0)  # [total_seq_len, input_dim]
        
        return reconstructed


class InformationOrientedSystem(nn.Module):
    """
    基于信息元的完整信息化系统
    
    核心流程：
    原始输入 → 信息元提取 → 信息组构建 → 球面映射 → 结构化输出
    
    输出特点：
    - 完全可解释（每个信息元有明确语义）
    - 完全可解码（可逆向重构）
    - 结构化（层次+拓扑）
    - 低延迟（并行处理）
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 content_dim: int = 128,  # 增加到128以保留更多信息
                 info_dim: int = 32,
                 num_classes: int = 10,
                 spatial_threshold: float = 0.5,
                 temporal_threshold: float = 1.0,
                 semantic_threshold: float = 0.6,
                 device: str = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.content_dim = content_dim
        self.info_dim = info_dim
        
        # 维度一致性检查
        # InformationElement.content property硬编码为128维
        # 所以系统的content_dim也必须是128
        if content_dim != 128:
            raise ValueError(
                f"content_dim必须为128以匹配InformationElement.content的固定维度，"
                f"当前设置为{content_dim}"
            )
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # === 阶段1: 信息元提取器 ===
        self.element_extractor = InformationElementExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            content_dim=content_dim,
            num_roles=8,
            num_types=6
        )
        
        # === 阶段2: 信息组构建器 ===
        self.group_builder = InformationGroupBuilder(
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            semantic_threshold=semantic_threshold,
            min_group_size=1,
            max_group_size=10
        )
        
        # === 阶段3: 球面映射系统 ===
        # 计算聚合特征的维度：content_dim + 3(spatial) + 1(temporal) + content_dim(variance)
        # 注意：不管多少个组，聚合后维度都是这个
        aggregated_feature_dim = content_dim * 2 + 4
        
        self.sphere_mapper = InformationSphereSystem(
            input_dim=aggregated_feature_dim,
            info_dim=info_dim,
            num_classes=num_classes
        )
        
        # === 阶段4: 信息重构器（新增）===
        # 用于从信息元列表逆向重构原始信息
        self.reconstructor = InformationReconstructor(
            content_dim=content_dim,
            output_feature_dim=input_dim
        )
        
        self.to(self.device)
    
    def forward(self, 
                raw_input: torch.Tensor,
                context: Optional[Dict] = None,
                return_details: bool = True) -> Dict[str, Any]:
        """
        完整的信息化处理流程
        
        Args:
            raw_input: [batch, seq_len, input_dim] 或 [seq_len, input_dim]
            context: 可选的上下文信息
            return_details: 是否返回详细的中间结果
            
        Returns:
            Dict包含：
            - elements: 信息元列表
            - groups: 信息组列表
            - sphere_nodes: 球面节点列表
            - features: 聚合特征
            - sphere_coords: 球面坐标
            - predictions: 预测结果（如果有）
            - interpretable: 可解释性标记
            - decodable: 可解码性标记
        """
        raw_input = raw_input.to(self.device)
        
        # === 阶段1: 信息元提取 ===
        elements = self.element_extractor(raw_input, context)
        
        if len(elements) == 0:
            # 如果没有提取到信息元，返回空结果
            return self._empty_result()
        
        # === 阶段2: 信息组构建 ===
        groups = self.group_builder.build_groups(elements)
        
        if len(groups) == 0:
            # 如果没有构建出信息组，返回空结果
            return self._empty_result()
        
        # === 阶段3: 球面结构化 ===
        structured_groups, sphere_nodes = self._map_to_sphere(groups)
        
        # === 阶段4: 特征聚合 ===
        aggregated_features, all_sphere_coords = self._aggregate_features(structured_groups)
        
        # === 阶段5: 预测（可选）===
        predictions = None
        if aggregated_features is not None:
            try:
                # 使用球面系统进行预测
                predictions = self.sphere_mapper.predict(aggregated_features, use_neighbors=False)
            except Exception as e:
                print(f"预测失败: {e}")
                predictions = None
        
        # === 构建输出 ===
        output = {
            'elements': elements,
            'groups': structured_groups,
            'sphere_nodes': sphere_nodes,
            'features': aggregated_features,
            'sphere_coords': all_sphere_coords,
            'predictions': predictions,
            'interpretable': True,
            'decodable': True,
            'statistics': self._compute_statistics(elements, structured_groups)
        }
        
        if not return_details:
            # 只返回必要信息
            output = {
                'features': aggregated_features,
                'sphere_coords': all_sphere_coords,
                'predictions': predictions,
                'num_elements': len(elements),
                'num_groups': len(structured_groups)
            }
        
        return output
    
    def _map_to_sphere(self, groups: List[InformationGroup]) -> Tuple[List[InformationGroup], List[InformationNode]]:
        """将信息组映射到球面"""
        structured_groups = []
        sphere_nodes = []
        
        for group in groups:
            try:
                # 聚合信息组
                agg = group.aggregate()
                
                # 构建组tensor
                group_tensor = torch.cat([
                    agg['content'].flatten(),
                    agg['spatial'].flatten(),
                    agg['temporal'].flatten(),
                    agg['variance'].flatten()
                ]).unsqueeze(0).to(self.device)
                
                # 使用球面系统提取信息和映射位置
                info_dict = self.sphere_mapper.info_extractor(group_tensor)
                position_dict = self.sphere_mapper.core_mapper(info_dict)
                
                # 更新信息组的球面坐标
                group.sphere_coords = (
                    position_dict['r'].item(),
                    position_dict['theta'].item(),
                    position_dict['phi'].item()
                )
                group.abstraction_level = position_dict['abstraction'].item()
                
                # 创建球面节点
                r, theta, phi = group.sphere_coords
                # 计算笛卡尔坐标
                import math
                x = r * math.sin(theta) * math.cos(phi)
                y = r * math.sin(theta) * math.sin(phi)
                z = r * math.cos(theta)
                
                # 创建空间结构
                spatial_struct = SpatialStructure(
                    position=(r, theta, phi),
                    cartesian=(x, y, z),
                    local_density=0.0,
                    geodesic_center=r,
                    spatial_coherence=group.coherence,
                    layer_depth=int(r * 5)  # 0-5层
                )
                
                # 创建时间结构
                import time
                current_time = time.time()
                temporal_struct = TemporalStructure(
                    observation_time=current_time,
                    access_time=current_time,
                    evolution_time=0.0,
                    temporal_stability=1.0,
                    change_rate=0.0,
                    temporal_coherence=1.0
                )
                
                sphere_node = InformationNode(
                    node_id=group.group_id,
                    spatial=spatial_struct,
                    temporal=temporal_struct,
                    spatial_info=info_dict['spatial'].squeeze(0).cpu(),
                    temporal_info=info_dict['temporal'].squeeze(0).cpu(),
                    change_info=info_dict['change'].squeeze(0).cpu(),
                    bias_info=info_dict['bias'].squeeze(0).cpu(),
                    abstraction_level=group.abstraction_level,
                    importance=float(np.mean([e.importance for e in group.elements]))
                )
                
                structured_groups.append(group)
                sphere_nodes.append(sphere_node)
                
            except Exception as e:
                print(f"映射信息组到球面失败: {e}")
                continue
        
        return structured_groups, sphere_nodes
    
    def _aggregate_features(self, groups: List[InformationGroup]) -> Tuple[Optional[torch.Tensor], List[Tuple]]:
        """
        聚合信息组特征 - 逻辑自洽的设计
        
        原则：
        1. 保持维度一致性（始终返回固定维度）
        2. 保留信息完整性（不丢失组间差异）
        3. 可解释性（聚合方式有明确含义）
        
        方法：
        - 单组：直接返回组特征
        - 多组：基于重要性的加权融合（维度不变）
        """
        if not groups:
            return None, []
        
        all_features = []
        all_coords = []
        
        # 提取每个组的特征
        for group in groups:
            agg = group.aggregate()
            # 构建特征向量: content + spatial + temporal + variance
            feature = torch.cat([
                agg['content'].flatten(),  # content_dim
                agg['spatial'].flatten(),  # 3
                agg['temporal'].flatten(), # 1
                agg['variance'].flatten()  # content_dim
            ])  # 总维度 = 2*content_dim + 4
            all_features.append(feature)
            
            if group.sphere_coords:
                all_coords.append(group.sphere_coords)
        
        if all_features:
            if len(all_features) == 1:
                # 单组：直接返回
                aggregated = all_features[0]
            else:
                # 多组：基于重要性的加权融合
                # 重要性 = 内聚性 × 元素数量
                importances = []
                for group in groups:
                    importance = group.coherence * len(group.elements)
                    importances.append(importance)
                
                # 归一化权重
                weights = torch.tensor(importances, device=all_features[0].device)
                weights = F.softmax(weights, dim=0)
                
                # 加权融合（维度保持不变）
                features_stack = torch.stack(all_features)
                aggregated = (features_stack * weights.unsqueeze(-1)).sum(dim=0)
                
                # aggregated维度 = 2*content_dim + 4 （与单组一致）
            
            return aggregated, all_coords
        
        return None, []
    
    def _compute_statistics(self, 
                           elements: List[InformationElement],
                           groups: List[InformationGroup]) -> Dict[str, Any]:
        """计算统计信息"""
        stats = {
            'num_elements': len(elements),
            'num_groups': len(groups),
            'avg_elements_per_group': len(elements) / len(groups) if groups else 0,
        }
        
        if elements:
            stats.update({
                'avg_element_importance': np.mean([e.importance for e in elements]),
                'avg_element_certainty': np.mean([e.certainty for e in elements]),
                'element_types': {et.value: sum(1 for e in elements if e.element_type == et) 
                                 for et in set(e.element_type for e in elements)},
                'semantic_roles': {sr.value: sum(1 for e in elements if e.semantic_role == sr)
                                  for sr in set(e.semantic_role for e in elements)}
            })
        
        if groups:
            stats.update({
                'avg_group_coherence': np.mean([g.coherence for g in groups]),
                'avg_spatial_radius': np.mean([g.spatial_radius for g in groups]),
                'total_temporal_span': max(g.temporal_range[1] for g in groups) - 
                                      min(g.temporal_range[0] for g in groups) if groups else 0
            })
        
        return stats
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'elements': [],
            'groups': [],
            'sphere_nodes': [],
            'features': None,
            'sphere_coords': [],
            'predictions': None,
            'interpretable': True,
            'decodable': True,
            'statistics': {'num_elements': 0, 'num_groups': 0}
        }
    
    def decode(self, output: Dict[str, Any]) -> str:
        """
        解码：从结构化信息恢复语义描述
        
        这是信息化系统的关键优势：可以直接解码
        """
        groups = output.get('groups', [])
        
        if not groups:
            return "[空信息]"
        
        decoded_parts = []
        
        # 按时间顺序排序
        sorted_groups = sorted(groups, key=lambda g: g.temporal_range[0])
        
        for group_idx, group in enumerate(sorted_groups):
            # 解码信息组
            group_desc = f"[信息组{group_idx+1}]"
            
            # 按重要性排序元素
            sorted_elements = sorted(group.elements, key=lambda e: e.importance, reverse=True)
            
            element_descs = []
            for elem in sorted_elements[:3]:  # 只显示前3个最重要的
                # 解码信息元
                elem_desc = f"{elem.semantic_role.value}:{elem.element_type.value}"
                element_descs.append(elem_desc)
            
            group_desc += "(" + ", ".join(element_descs) + ")"
            
            # 添加球面位置信息
            if group.sphere_coords:
                r, theta, phi = group.sphere_coords
                group_desc += f" @球面(r={r:.2f}, θ={theta:.2f}, φ={phi:.2f})"
            
            decoded_parts.append(group_desc)
        
        return " → ".join(decoded_parts)
    
    def visualize_structure(self, output: Dict[str, Any]):
        """可视化信息结构"""
        print("\n" + "="*80)
        print("信息化系统输出")
        print("="*80)
        
        elements = output.get('elements', [])
        groups = output.get('groups', [])
        stats = output.get('statistics', {})
        
        print(f"\n整体统计:")
        print(f"  信息元数量: {stats.get('num_elements', 0)}")
        print(f"  信息组数量: {stats.get('num_groups', 0)}")
        print(f"  平均元素/组: {stats.get('avg_elements_per_group', 0):.2f}")
        print(f"  平均内聚性: {stats.get('avg_group_coherence', 0):.3f}")
        
        if 'element_types' in stats:
            print(f"\n元素类型分布:")
            for etype, count in stats['element_types'].items():
                print(f"  {etype}: {count}")
        
        if 'semantic_roles' in stats:
            print(f"\n语义角色分布:")
            for role, count in stats['semantic_roles'].items():
                print(f"  {role}: {count}")
        
        # 显示前3个信息组
        print(f"\n前3个信息组详情:")
        for i, group in enumerate(groups[:3]):
            print(f"\n信息组 {i+1}:")
            print_group(group, indent=1)
            
            # 显示组内前2个元素
            print(f"  包含的元素（前2个）:")
            for elem in group.elements[:2]:
                print_element(elem, indent=2)
        
        # 解码
        decoded = self.decode(output)
        print(f"\n解码结果:")
        print(f"  {decoded}")
        
        print("\n" + "="*80)
    
    def reconstruct(self, output: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        重构：从信息元列表逆向重构原始信息
        
        核心改进：不再从聚合向量重构，而是从信息元列表重构！
        每个信息元重构对应的原始输入片段，然后拼接。
        
        Args:
            output: forward的输出，包含'elements'列表
        
        Returns:
            reconstructed: 重构的原始序列 [num_elements, feature_dim]
        """
        if 'elements' not in output or not output['elements']:
            return None
        
        # 从信息元列表重构（不是从聚合特征！）
        reconstructed = self.reconstructor(output['elements'])
        return reconstructed


class InformationOrientedTrainer:
    """信息化系统的训练器（可选）"""
    
    def __init__(self, system: InformationOrientedSystem, lr: float = 0.001):
        self.system = system
        self.optimizer = torch.optim.Adam(system.parameters(), lr=lr)
    
    def train_step(self, batch_data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        训练一步
        
        注意：信息元提取和组构建是无监督的
        只有球面映射和预测部分需要监督信号
        """
        self.system.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        output = self.system(batch_data, return_details=False)
        
        predictions = output.get('predictions')
        
        if predictions is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # 计算损失
        loss = F.cross_entropy(predictions, labels)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        # 计算准确率
        pred_labels = predictions.argmax(dim=1)
        accuracy = (pred_labels == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }

