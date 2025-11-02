"""
Information Sphere System v1.0

从数据化到信息化的范式转变
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .information_element_system import (
    InformationElement,
    InformationGroup,
    InformationElementExtractor,
    InformationGroupBuilder,
    ElementType,
    SemanticRole,
    Modality
)

from .information_oriented_system import (
    InformationOrientedSystem,
    InformationOrientedTrainer
)

from .information_sphere_system import (
    InformationSphereSystem,
    InformationNode,
    SpatialStructure,
    TemporalStructure,
    efficient_contrastive_train
)

__all__ = [
    # Core system
    'InformationOrientedSystem',
    'InformationOrientedTrainer',
    
    # Information elements
    'InformationElement',
    'InformationGroup',
    'InformationElementExtractor',
    'InformationGroupBuilder',
    
    # Sphere system
    'InformationSphereSystem',
    'InformationNode',
    'SpatialStructure',
    'TemporalStructure',
    
    # Training
    'efficient_contrastive_train',
    
    # Enums
    'ElementType',
    'SemanticRole',
    'Modality',
]

