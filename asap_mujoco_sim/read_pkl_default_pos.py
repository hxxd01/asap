#!/usr/bin/env python3
"""
从PKL文件读取初始帧关节位置工具

使用项目的Motion Library来正确处理关节映射

功能增强:
1. 支持将读取的关节位置叠加到指定基础位置上
2. 提供多种预定义基础位置类型:
   - 'g1_standard': G1机器人标准站立位置
   - 'zero_pose': 全零姿态
   - 'custom': 用户自定义位置
3. 可选择是否使用基础位置叠加模式

使用方法:
1. 修改main()函数中的base_pos_type变量选择基础位置类型
2. 修改use_base_pos变量控制是否使用叠加模式
3. 如果选择'custom'类型，在相应位置修改base_initial_pos列表
"""

import pickle
import numpy as np
import os
import joblib
import sys
import torch

# 添加项目路径以导入Motion Library
sys.path.append('/home/harry/Desktop/ASAP1')

try:
    from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    from easydict import EasyDict
except ImportError as e:
    print(f"⚠️ 导入Motion Library失败: {e}")
    print("请确保已经安装了所需的依赖包")


def read_pkl_default_pos_with_motion_lib(pkl_file_path, base_pos=None):
    """
    使用项目的Motion Library来正确处理关节映射
    
    Args:
        pkl_file_path: PKL文件路径
        base_pos: 基础初始位置，如果提供，会将读取的位置加到这个基础位置上
    """
    print(f"📁 读取PKL文件: {pkl_file_path}")
    
    if not os.path.exists(pkl_file_path):
        print(f"❌ PKL文件不存在: {pkl_file_path}")
        return None
    
    try:
        # 1. 加载PKL文件
        pkl_data = joblib.load(pkl_file_path)
        print("✅ 使用joblib成功加载PKL文件")
        
        # 2. 处理PKL文件结构
        if isinstance(pkl_data, dict):
            motion_key = list(pkl_data.keys())[0]
            motion_data = pkl_data[motion_key]
            print(f"📊 Motion key: {motion_key}")
        else:
            motion_data = pkl_data
            
        print(f"📊 PKL文件字段: {list(motion_data.keys())}")
        
        # 3. 创建Motion Library的配置
        cfg = EasyDict({
            'asset': {
                'assetRoot': '/home/harry/Desktop/ASAP1/humanoidverse/data/robots/g1',
                'assetFileName': 'g1_23dof_lock_wrist_fitmotionONLY.xml'
            },
            'extend_config': []  # 使用标准配置
        })
        
        # 4. 初始化Motion Library
        print("🔧 初始化Motion Library...")
        motion_lib = Humanoid_Batch(cfg, device=torch.device("cpu"))
        
        # 5. 从PKL文件提取pose_aa数据
        pose_aa = motion_data['pose_aa']
        trans = motion_data['root_trans_offset']
        
        print(f"📈 PKL数据信息:")
        print(f"   pose_aa形状: {pose_aa.shape}")
        print(f"   trans形状: {trans.shape}")
        
        # 6. 转换为torch张量
        pose_aa_torch = torch.from_numpy(pose_aa).float()
        trans_torch = torch.from_numpy(trans).float()
        
        # 7. 使用Motion Library的fk_batch方法处理第一帧
        print("🚀 使用Motion Library处理第一帧...")
        
        # 为了避免gradient计算错误，我们需要提供多帧数据
        # 取前3帧数据，然后只使用第一帧的结果
        num_frames = min(3, pose_aa.shape[0])
        multi_frame_pose = pose_aa_torch[:num_frames]  # 取前几帧
        multi_frame_trans = trans_torch[:num_frames]
        
        # 添加batch维度
        multi_frame_pose = multi_frame_pose.unsqueeze(0)  # (1, num_frames, joints, 3)
        multi_frame_trans = multi_frame_trans.unsqueeze(0)  # (1, num_frames, 3)
        
        print(f"📊 输入数据形状:")
        print(f"   multi_frame_pose: {multi_frame_pose.shape}")
        print(f"   multi_frame_trans: {multi_frame_trans.shape}")
        
        # 8. 执行Forward Kinematics
        result = motion_lib.fk_batch(multi_frame_pose, multi_frame_trans, return_full=True, dt=1.0/30.0)
        
        # 9. 提取第一帧的dof_pos
        dof_pos = result.dof_pos.squeeze()[0].numpy()  # 取第一帧，移除batch维度
        
        print(f"🎯 Motion Library输出的dof_pos:")
        print(f"   形状: {dof_pos.shape}")
        print(f"   数值: {dof_pos}")
        print(f"   范围: [{np.min(dof_pos):.3f}, {np.max(dof_pos):.3f}]")
        
        # 10. 显示关节映射信息
        print(f"📋 关节映射信息:")
        print(f"   总关节数: {len(motion_lib.body_names)}")
        print(f"   actuated_joints_idx: {motion_lib.actuated_joints_idx}")
        print(f"   body_names: {motion_lib.body_names}")
        
        # 11. 处理基础位置叠加
        if base_pos is not None:
            base_pos = np.array(base_pos)
            if len(base_pos) != len(dof_pos):
                print(f"⚠️ 基础位置长度 {len(base_pos)} 与读取位置长度 {len(dof_pos)} 不匹配")
                print("使用读取的原始位置")
                final_pos = dof_pos
            else:
                final_pos = dof_pos + base_pos
                print(f"📊 位置叠加:")
                print(f"   基础位置: {base_pos}")
                print(f"   读取位置: {dof_pos}")
                print(f"   最终位置: {final_pos}")
        else:
            final_pos = dof_pos
            print("📊 未提供基础位置，使用读取的原始位置")
        
        # 12. 返回结果
        if len(final_pos) == 23:
            print("✅ 正确获取23个关节的默认位置!")
            return final_pos
        else:
            print(f"⚠️ 获取的关节数为 {len(final_pos)}，不是预期的23个")
            return final_pos
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_predefined_base_positions():
    """
    获取预定义的基础位置
    """
    positions = {
        'g1_standard': [
            -0.1, 0., 0.,     # 左腿髋关节
            0.3, -0.2, 0.,    # 左腿膝关节和踝关节
            -0.1, 0., 0.,     # 右腿髋关节
            0.3, -0.2, 0.,    # 右腿膝关节和踝关节
            0., 0., 0.,       # 腰部关节
            0., 0., 0.,       # 左臂关节
            0., 0., 0.,       # 右臂关节
            0., 0.           # 其他关节
        ],
        'zero_pose': [0.0] * 23,  # 全零姿态
        'custom': None  # 用户自定义
    }
    return positions


def main():
    """
    主函数
    """
    print("🚀 使用Motion Library读取PKL初始帧关节位置")
    print("=" * 60)
    
    # PKL文件路径
    pkl_file_path = "/home/harry/Desktop/ASAP1/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"
    
    # 选择基础位置类型
    base_positions = get_predefined_base_positions()
    base_pos_type = 'zero_pose'  # 可选: 'g1_standard', 'zero_pose', 'custom'
    
    # 获取基础初始位置
    if base_pos_type == 'custom':
        # 用户自定义位置，在这里修改
        base_initial_pos = [
            -0.1, 0., 0.,     # 左腿髋关节
            0.3, -0.2, 0.,    # 左腿膝关节和踝关节
            -0.1, 0., 0.,     # 右腿髋关节
            0.3, -0.2, 0.,    # 右腿膝关节和踝关节
            0., 0., 0.,       # 腰部关节
            0.2, 0.2, 0.,       # 左臂关节
            0.9, 0.2, -0.2,       # 右臂关节
            0., 0.9           # 其他关节
        ]
    else:
        base_initial_pos = base_positions[base_pos_type]
    
    print(f"📊 基础初始位置类型: {base_pos_type}")
    print(f"   位置数值: {base_initial_pos}")
    print(f"   长度: {len(base_initial_pos)}")
    
    # 使用Motion Library处理 - 可以选择是否使用基础位置
    use_base_pos = True  # 设置为 False 则不使用基础位置叠加
    
    if use_base_pos:
        print(f"📌 使用基础位置叠加模式 (类型: {base_pos_type})")
        dof_pos = read_pkl_default_pos_with_motion_lib(pkl_file_path, base_pos=base_initial_pos)
    else:
        print("📌 使用原始位置模式")
        dof_pos = read_pkl_default_pos_with_motion_lib(pkl_file_path)
    
    if dof_pos is not None:
        print(f"\n📝 最终结果:")
        print(f"default_dof_pos = {dof_pos.tolist()}")
        
        if use_base_pos:
            print(f"\n📊 位置叠加总结:")
            print(f"   基础位置: {base_initial_pos}")
            print(f"   PKL位置:  {(dof_pos - np.array(base_initial_pos)).tolist()}")
            print(f"   最终位置: {dof_pos.tolist()}")
        
        print(f"\n✅ 成功读取{len(dof_pos)}个关节的位置!")
        return dof_pos
    else:
        print("❌ 无法获取关节位置")
        return None


if __name__ == "__main__":
    main() 