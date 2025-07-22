"""
自适应阈值加载器
用于避免加载预训练模型时motion_far阈值的突变问题
"""
import torch
import yaml
from omegaconf import OmegaConf
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AdaptiveThresholdLoader:
    """
    智能加载和适应motion_far阈值，避免奖励突变
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        
    def extract_threshold_from_checkpoint(self, checkpoint_path: str) -> Optional[float]:
        """
        从checkpoint中提取当前的motion_far阈值
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 尝试从不同可能的位置提取阈值
            possible_keys = [
                'env_state_dict',
                'env_info', 
                'terminate_when_motion_far_threshold',
                'training_state'
            ]
            
            for key in possible_keys:
                if key in checkpoint:
                    env_state = checkpoint[key]
                    if isinstance(env_state, dict) and 'terminate_when_motion_far_threshold' in env_state:
                        threshold = float(env_state['terminate_when_motion_far_threshold'])
                        logger.info(f"从checkpoint中提取到motion_far阈值: {threshold}")
                        return threshold
                        
            logger.warning("无法从checkpoint中提取motion_far阈值")
            return None
            
        except Exception as e:
            logger.error(f"提取阈值时出错: {e}")
            return None
    
    def create_adaptive_config(self, 
                              base_config: Dict[str, Any],
                              pretrained_threshold: float,
                              target_min: float = 0.3,
                              target_max: float = 2.0,
                              adaptation_rate: float = 0.00001) -> Dict[str, Any]:
        """
        创建自适应配置，从预训练阈值平滑过渡到目标范围
        
        Args:
            base_config: 基础配置
            pretrained_threshold: 预训练模型的阈值
            target_min: 目标最小阈值  
            target_max: 目标最大阈值
            adaptation_rate: 适应速率
        """
        
        # 计算合理的初始范围
        # 如果预训练阈值过小，允许适度放宽
        if pretrained_threshold < 0.5:
            initial_max = min(pretrained_threshold * 2, target_max)
        else:
            initial_max = min(pretrained_threshold * 1.5, target_max)
            
        # 如果预训练阈值过大，允许适度收紧  
        if pretrained_threshold > 1.5:
            initial_min = max(pretrained_threshold * 0.8, target_min)
        else:
            initial_min = target_min
            
        # 更新配置
        adaptive_config = base_config.copy()
        
        curriculum_config = {
            'terminate_when_motion_far_curriculum': True,
            'terminate_when_motion_far_initial_threshold': pretrained_threshold,
            'terminate_when_motion_far_threshold_max': initial_max,
            'terminate_when_motion_far_threshold_min': initial_min,
            'terminate_when_motion_far_curriculum_degree': adaptation_rate,
            'terminate_when_motion_far_curriculum_level_down_threshold': 35,
            'terminate_when_motion_far_curriculum_level_up_threshold': 45
        }
        
        if 'env' not in adaptive_config:
            adaptive_config['env'] = {}
        if 'config' not in adaptive_config['env']:
            adaptive_config['env']['config'] = {}
        if 'termination_curriculum' not in adaptive_config['env']['config']:
            adaptive_config['env']['config']['termination_curriculum'] = {}
            
        adaptive_config['env']['config']['termination_curriculum'].update(curriculum_config)
        
        logger.info(f"创建自适应配置:")
        logger.info(f"  初始阈值: {pretrained_threshold}")
        logger.info(f"  允许范围: [{initial_min:.3f}, {initial_max:.3f}]")
        logger.info(f"  目标范围: [{target_min}, {target_max}]")
        logger.info(f"  适应速率: {adaptation_rate}")
        
        return adaptive_config
    
    def load_with_smooth_transition(self,
                                   checkpoint_path: str,
                                   config: Dict[str, Any],
                                   fallback_threshold: float = 0.7) -> Dict[str, Any]:
        """
        加载预训练模型并创建平滑过渡配置
        
        Args:
            checkpoint_path: 预训练模型路径
            config: 当前配置
            fallback_threshold: 如果无法提取阈值时的后备值
        """
        
        # 尝试从checkpoint提取阈值
        extracted_threshold = self.extract_threshold_from_checkpoint(checkpoint_path)
        
        # 使用提取的阈值或后备值
        start_threshold = extracted_threshold if extracted_threshold is not None else fallback_threshold
        
        # 创建自适应配置
        adaptive_config = self.create_adaptive_config(
            config, 
            start_threshold,
            target_min=0.3,
            target_max=2.0,
            adaptation_rate=0.00001
        )
        
        return adaptive_config

def smooth_load_pretrained_model(checkpoint_path: str, 
                                config: Dict[str, Any],
                                fallback_threshold: float = 0.7) -> Dict[str, Any]:
    """
    便捷函数：平滑加载预训练模型，避免阈值突变
    
    使用示例:
    ```python
    from humanoidverse.utils.adaptive_threshold_loader import smooth_load_pretrained_model
    
    # 在加载预训练模型前调用
    config = smooth_load_pretrained_model(
        checkpoint_path="path/to/pretrained/model.pth",
        config=original_config,
        fallback_threshold=0.7659  # 您之前训练的阈值
    )
    ```
    """
    loader = AdaptiveThresholdLoader()
    return loader.load_with_smooth_transition(checkpoint_path, config, fallback_threshold) 