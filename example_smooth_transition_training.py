#!/usr/bin/env python3
"""
平滑过渡训练示例
展示如何避免加载预训练模型时motion_far阈值的突变问题
"""

import hydra
from omegaconf import OmegaConf
import torch
import logging
from pathlib import Path

# 导入自适应阈值加载器
from humanoidverse.utils.adaptive_threshold_loader import smooth_load_pretrained_model

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    """
    主训练函数，演示平滑过渡的使用
    """
    
    # 预训练模型路径
    pretrained_checkpoint = "path/to/your/pretrained_model.pth"
    
    # 检查是否需要加载预训练模型
    if config.get("pretrained_checkpoint", None):
        pretrained_checkpoint = config.pretrained_checkpoint
    
    if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
        logger.info(f"准备加载预训练模型: {pretrained_checkpoint}")
        
        # 方法1: 使用便捷函数（推荐）
        logger.info("使用自适应阈值加载器...")
        config = smooth_load_pretrained_model(
            checkpoint_path=pretrained_checkpoint,
            config=OmegaConf.to_container(config, resolve=True),
            fallback_threshold=0.7659  # 您之前训练的阈值
        )
        config = OmegaConf.create(config)
        
        # 打印关键配置信息
        curriculum_config = config.env.config.termination_curriculum
        logger.info("=== 平滑过渡配置 ===")
        logger.info(f"初始阈值: {curriculum_config.terminate_when_motion_far_initial_threshold}")
        logger.info(f"阈值范围: [{curriculum_config.terminate_when_motion_far_threshold_min}, {curriculum_config.terminate_when_motion_far_threshold_max}]")
        logger.info(f"调整速率: {curriculum_config.terminate_when_motion_far_curriculum_degree}")
        
    else:
        # 如果没有预训练模型，使用标准配置
        logger.info("未找到预训练模型，使用标准课程学习配置")
        
        # 方法2: 手动设置平滑过渡配置
        if hasattr(config.env.config, 'termination_curriculum'):
            # 从较为宽松的阈值开始
            config.env.config.termination_curriculum.terminate_when_motion_far_initial_threshold = 1.0
            config.env.config.termination_curriculum.terminate_when_motion_far_threshold_max = 2.0
            config.env.config.termination_curriculum.terminate_when_motion_far_threshold_min = 0.3
            config.env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree = 0.00002
    
    # 继续正常的训练流程
    logger.info("开始训练...")
    
    # 这里放置您的正常训练代码
    # from humanoidverse.agents.base_algo.base_algo import BaseAlgo
    # algo = BaseAlgo(config)
    # algo.train()
    
    # 模拟训练过程，展示阈值变化
    simulate_threshold_evolution(config)

def simulate_threshold_evolution(config):
    """
    模拟阈值在训练过程中的演化
    """
    curriculum = config.env.config.termination_curriculum
    
    initial_threshold = curriculum.terminate_when_motion_far_initial_threshold
    min_threshold = curriculum.terminate_when_motion_far_threshold_min  
    max_threshold = curriculum.terminate_when_motion_far_threshold_max
    degree = curriculum.terminate_when_motion_far_curriculum_degree
    level_down = curriculum.terminate_when_motion_far_curriculum_level_down_threshold
    level_up = curriculum.terminate_when_motion_far_curriculum_level_up_threshold
    
    current_threshold = initial_threshold
    
    logger.info("\n=== 模拟训练过程中的阈值演化 ===")
    
    # 模拟不同的average_episode_length场景
    scenarios = [
        (30, "episode太短，放宽阈值"),
        (35, "episode太短，放宽阈值"), 
        (40, "episode长度适中，保持阈值"),
        (45, "episode太长，收紧阈值"),
        (50, "episode太长，收紧阈值"),
        (38, "episode太短，放宽阈值"),
        (43, "episode太长，收紧阈值")
    ]
    
    for episode_length, description in scenarios:
        old_threshold = current_threshold
        
        if episode_length < level_down:
            # 放宽阈值
            current_threshold *= (1 + degree)
        elif episode_length > level_up:
            # 收紧阈值  
            current_threshold *= (1 - degree)
            
        # 限制在范围内
        current_threshold = max(min_threshold, min(current_threshold, max_threshold))
        
        change = current_threshold - old_threshold
        logger.info(f"Episode长度: {episode_length:2d} | {description} | 阈值: {old_threshold:.6f} -> {current_threshold:.6f} (变化: {change:+.6f})")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main() 