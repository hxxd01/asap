#!/usr/bin/env python3
"""
正确的时间对齐处理：
1. 检测动捕数据的真正运动起始点
2. 根据PKL的持续时长裁剪CSV数据
3. 将动捕数据插值到PKL的真实时间点
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.signal import medfilt, find_peaks
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

def parse_dof_axis_from_xml(xml_path):
    """从XML文件解析关节轴向量 - 与mujoco_track.py完全一致"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    joints = root.findall('.//joint')
    dof_axis = []
    for j in joints:
        if 'type' in j.attrib and j.attrib['type'] in ['free', 'float']:
            continue
        axis_str = j.attrib.get('axis', None)
        if axis_str is not None:
            axis = [float(x) for x in axis_str.strip().split()]
            dof_axis.append(axis)
    return np.array(dof_axis, dtype=np.float32)

def fix_pose_aa(pkl_data, xml_path):
    """修复pose_aa计算 - 与mujoco_track.py完全一致"""
    print(f"🔧 修复pose_aa计算...")
    
    # 解析关节轴向量
    dof_axis = parse_dof_axis_from_xml(xml_path)
    print(f"   解析关节轴向量: {dof_axis.shape}")
    
    # 获取数据
    root_rot = pkl_data['root_rot']  # (N, 4) xyzw格式
    dof = pkl_data['dof']  # (N, 23)
    N = len(root_rot)
    
    # 重新计算pose_aa
    pose_aa_list = []
    for i in range(N):
        # base四元数转轴角 - 与mujoco_track.py一致
        root_rot_vec = R.from_quat(root_rot[i]).as_rotvec()  # shape (3,)
        
        # 关节角度与轴向量相乘
        joint_aa = dof[i][:, None] * dof_axis  # shape (23, 3)
        
        # 拼接：base轴角 + 关节轴角 + 3个虚拟关节
        num_augment_joint = 3
        pose_aa_frame = np.concatenate([
            root_rot_vec[None, :],  # (1, 3)
            joint_aa,               # (23, 3)
            np.zeros((num_augment_joint, 3), dtype=np.float32)  # (3, 3)
        ], axis=0)  # shape (27, 3)
        
        pose_aa_list.append(pose_aa_frame)
    
    fixed_pose_aa = np.array(pose_aa_list, dtype=np.float32)  # 确保数据类型为float32
    print(f"✅ pose_aa修复完成: {fixed_pose_aa.shape}")
    print(f"   pose_aa范围: [{fixed_pose_aa.min():.3f}, {fixed_pose_aa.max():.3f}]")
    print(f"   pose_aa数据类型: {fixed_pose_aa.dtype}")
    
    return fixed_pose_aa

def load_csv_data_asap(csv_file):
    """加载ASAP格式的CSV数据，包括位置和速度"""
    print(f"📂 加载CSV文件: {os.path.basename(csv_file)}")
    
    # 查找数据开始的行
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    start_row = None
    for i, line in enumerate(lines):
        if 'Frame#' in line:
            start_row = i + 1  # 跳过表头，从数据行开始
            break
    
    if start_row is None:
        # 尝试pandas直接读取
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ 直接加载CSV成功")
        except Exception as e:
            raise ValueError(f"无法找到数据开始行且直接读取失败: {e}")
    else:
        print(f"   跳过前{start_row}行头部信息")
        df = pd.read_csv(csv_file, skiprows=start_row)
    
    # 检查并清理数据
    print(f"   原始数据形状: {df.shape}")
    
    # 过滤有效数据（但保留所有时间点进行运动检测）
    valid_positions = (df['XToGlobal1'] != 0) | (df['YToGlobal1'] != 0) | (df['ZToGlobal1'] != 0)
    valid_timestamps = df['Timestamp'] != 0
    df_clean = df[valid_positions & valid_timestamps].copy()
    
    print(f"   有效数据形状: {df_clean.shape}")
    
    # 提取位置数据 (mm)
    pos_x = df_clean['XToGlobal1'].values
    pos_y = df_clean['YToGlobal1'].values  
    pos_z = df_clean['ZToGlobal1'].values
    
    # 🎯 提取速度数据 (mm/s)
    vel_x = df_clean['VxToGlobal1'].values
    vel_y = df_clean['VyToGlobal1'].values
    vel_z = df_clean['VzToGlobal1'].values
    
    # 提取时间戳并转换为相对时间
    timestamps = df_clean['Timestamp'].values
    csv_time = (timestamps - timestamps[0]) / 1000.0  # 转换为相对秒数
    
    print(f"✅ 数据加载成功")
    print(f"   总帧数: {len(df_clean)}")
    print(f"   时间范围: {csv_time[0]:.2f}s - {csv_time[-1]:.2f}s ({csv_time[-1]:.1f}s)")
    print(f"   位置范围: X[{pos_x.min():.1f}, {pos_x.max():.1f}] Y[{pos_y.min():.1f}, {pos_y.max():.1f}] Z[{pos_z.min():.1f}, {pos_z.max():.1f}] mm")
    print(f"   速度范围: Vx[{vel_x.min():.1f}, {vel_x.max():.1f}] Vy[{vel_y.min():.1f}, {vel_y.max():.1f}] Vz[{vel_z.min():.1f}, {vel_z.max():.1f}] mm/s")
    
    return df_clean, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, csv_time

def detect_motion_start_by_pattern(pos_x, pos_y, pos_z, csv_time, search_start=8.0, search_end=25.0, offset_seconds=2.0):
    """基于复杂运动模式检测策略开始点"""
    print(f"🔍 基于运动模式检测策略起始点...")
    print(f"   策略搜索范围: {search_start}s - {search_end}s")
    print(f"   起始点前推时长: {offset_seconds}s")
    
    from scipy.signal import find_peaks
    
    # 确定搜索范围（跳过下放阶段）
    search_start_idx = int(search_start * 120)  # 8秒后开始搜索，避开下放阶段
    search_end_idx = int(search_end * 120)
    
    search_start_idx = max(0, search_start_idx)
    search_end_idx = min(len(pos_y), search_end_idx)
    
    if search_end_idx <= search_start_idx:
        print("❌ 搜索范围无效，使用默认起始点")
        return int(8.0 * 120)  # 8秒处
    
    print(f"   搜索帧范围: {search_start_idx} - {search_end_idx}")
    
    # 提取搜索范围内的数据
    y_search = pos_y[search_start_idx:search_end_idx]
    x_search = pos_x[search_start_idx:search_end_idx]
    z_search = pos_z[search_start_idx:search_end_idx]
    time_search = csv_time[search_start_idx:search_end_idx]
    
    print(f"   Y轴搜索范围: [{y_search.min():.1f}, {y_search.max():.1f}] mm")
    print(f"   X轴搜索范围: [{x_search.min():.1f}, {x_search.max():.1f}] mm")
    
    # 策略1: 检测显著的多轴运动复杂度增加
    window_size = 120  # 1秒窗口
    complexity_scores = []
    
    for i in range(len(y_search) - window_size):
        window_y = y_search[i:i+window_size]
        window_x = x_search[i:i+window_size]
        window_z = z_search[i:i+window_size]
        
        # 计算位置变化的标准差（复杂度指标）
        y_std = np.std(np.diff(window_y))
        x_std = np.std(np.diff(window_x))
        z_std = np.std(np.diff(window_z))
        
        # 计算总的运动复杂度
        total_complexity = y_std + x_std + z_std
        complexity_scores.append(total_complexity)
    
    complexity_scores = np.array(complexity_scores)
    
    # 策略2: 检测从稳定到复杂运动的转变
    baseline_complexity = np.mean(complexity_scores[:60])  # 前0.5秒作为基线
    complexity_threshold = baseline_complexity + 2 * np.std(complexity_scores[:60])
    
    print(f"   基线运动复杂度: {baseline_complexity:.2f}")
    print(f"   复杂度阈值: {complexity_threshold:.2f}")
    
    # 找到第一个超过阈值的点
    complex_motion_start = None
    for i, score in enumerate(complexity_scores):
        if score > complexity_threshold:
            complex_motion_start = i
            break
    
    if complex_motion_start is None:
        print("❌ 未检测到复杂运动，使用高度变化检测")
        return detect_motion_by_height_change(pos_x, pos_y, pos_z, csv_time, search_start, search_end, offset_seconds)
    
    # 转换回全局索引
    complex_motion_global_idx = complex_motion_start + search_start_idx
    complex_motion_time = csv_time[complex_motion_global_idx]
    
    print(f"✅ 检测到复杂运动开始:")
    print(f"   复杂运动时间: {complex_motion_time:.3f}s")
    print(f"   运动复杂度: {complexity_scores[complex_motion_start]:.2f}")
    
    # 计算策略起始点（复杂运动前推offset_seconds）
    strategy_start_time = complex_motion_time - offset_seconds
    strategy_start_idx = np.argmin(np.abs(csv_time - strategy_start_time))
    
    # 确保不早于搜索开始时间
    if strategy_start_idx < search_start_idx:
        strategy_start_idx = search_start_idx
        strategy_start_time = csv_time[strategy_start_idx]
        print(f"⚠️ 调整起始点到搜索范围内")
    
    actual_strategy_start_time = csv_time[strategy_start_idx]
    
    print(f"✅ 确定策略起始点:")
    print(f"   复杂运动时间: {complex_motion_time:.3f}s")
    print(f"   前推时长: {offset_seconds}s")
    print(f"   计算起始时间: {strategy_start_time:.3f}s")
    print(f"   实际起始时间: {actual_strategy_start_time:.3f}s (帧{strategy_start_idx})")
    print(f"   起始位置: X={pos_x[strategy_start_idx]:.1f}, Y={pos_y[strategy_start_idx]:.1f}, Z={pos_z[strategy_start_idx]:.1f} mm")
    
    return strategy_start_idx

def detect_motion_by_height_change(pos_x, pos_y, pos_z, csv_time, search_start=8.0, search_end=25.0, offset_seconds=2.0):
    """基于高度变化检测策略起始点（备用方法）"""
    print(f"🔍 使用高度变化检测...")
    
    from scipy.signal import find_peaks
    
    search_start_idx = int(search_start * 120)
    search_end_idx = int(search_end * 120)
    search_start_idx = max(0, search_start_idx)
    search_end_idx = min(len(pos_y), search_end_idx)
    
    y_search = pos_y[search_start_idx:search_end_idx]
    time_search = csv_time[search_start_idx:search_end_idx]
    
    # 寻找显著的高度变化点（上升阶段）
    # 检测从低点开始的显著上升
    valleys, _ = find_peaks(-y_search, prominence=20, distance=60)  # 找谷值
    
    # 🔧 添加高度阈值过滤：只有小于850mm的才算有效谷值
    height_threshold = 850.0  # mm
    print(f"   高度阈值过滤: 只考虑小于{height_threshold}mm的谷值")
    
    if len(valleys) == 0:
        print("❌ 未找到显著谷值，使用默认起始点")
        return search_start_idx
    
    # 过滤出符合高度阈值的谷值
    valid_valleys = []
    for valley_idx in valleys:
        if y_search[valley_idx] < height_threshold:
            valid_valleys.append(valley_idx)
    
    if len(valid_valleys) == 0:
        print("❌ 未找到符合高度阈值的有效谷值，使用默认起始点")
        return search_start_idx
    
    # 找到最低的有效谷值
    lowest_valley_idx = valid_valleys[np.argmin(y_search[valid_valleys])]
    lowest_valley_time = time_search[lowest_valley_idx]
    lowest_valley_global_idx = lowest_valley_idx + search_start_idx
    
    print(f"   找到最低有效点: t={lowest_valley_time:.3f}s, y={y_search[lowest_valley_idx]:.1f}mm")
    
    # 检测谷值后的上升运动
    post_valley_y = y_search[lowest_valley_idx:]
    
    if len(post_valley_y) < 120:  # 至少1秒数据
        print("❌ 谷值后数据不足")
        return lowest_valley_global_idx
    
    # 寻找上升起始点（谷值后显著上升开始）
    y_diff = np.diff(post_valley_y)
    smooth_diff = np.convolve(y_diff, np.ones(30)/30, mode='same')  # 平滑化
    
    # 找到持续上升的起始点
    rise_start_idx = None
    for i in range(len(smooth_diff) - 60):  # 至少0.5秒的上升
        if np.mean(smooth_diff[i:i+60]) > 0.5:  # 平均上升速度 > 0.5mm/frame
            rise_start_idx = i
            break
    
    if rise_start_idx is None:
        print("⚠️ 未找到显著上升，使用谷值点")
        strategy_start_idx = lowest_valley_global_idx
    else:
        rise_global_idx = lowest_valley_idx + rise_start_idx + search_start_idx
        rise_time = csv_time[rise_global_idx]
        
        # 前推offset_seconds
        strategy_start_time = rise_time - offset_seconds
        strategy_start_idx = np.argmin(np.abs(csv_time - strategy_start_time))
        
        # 确保不早于搜索开始
        if strategy_start_idx < search_start_idx:
            strategy_start_idx = search_start_idx
        
        print(f"   找到上升起始: t={rise_time:.3f}s")
        print(f"   前推{offset_seconds}s后: t={csv_time[strategy_start_idx]:.3f}s")
    
    actual_start_time = csv_time[strategy_start_idx]
    print(f"   最终起始点: t={actual_start_time:.3f}s (帧{strategy_start_idx})")
    
    return strategy_start_idx

def detect_motion_start_by_significant_pattern(pos_x, pos_y, pos_z, csv_time, search_start=10.0, search_end=40.0, offset_seconds=None):
    """智能检测策略起始点 - 寻找第一个显著谷值"""
    print(f"🔍 智能检测策略起始点...")
    print(f"   搜索范围: {search_start}s - {search_end}s")
    
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    
    # 步骤1: 检测第一个显著谷值
    print(f"📊 步骤1: 检测第一个显著谷值...")
    
    # 确定搜索范围 - 扩大搜索范围以包含第一个谷值
    search_start_idx = int(search_start * 120)
    search_end_idx = int(search_end * 120)
    
    search_start_idx = max(0, search_start_idx)
    search_end_idx = min(len(pos_y), search_end_idx)
    
    if search_end_idx <= search_start_idx:
        print("❌ 搜索范围无效")
        return int(15.0 * 120)
    
    print(f"   搜索帧范围: {search_start_idx} - {search_end_idx}")
    
    # 提取搜索范围内的数据
    y_search = pos_y[search_start_idx:search_end_idx]
    x_search = pos_x[search_start_idx:search_end_idx]
    z_search = pos_z[search_start_idx:search_end_idx]
    time_search = csv_time[search_start_idx:search_end_idx]
    
    print(f"   Y轴搜索范围: [{y_search.min():.1f}, {y_search.max():.1f}] mm")
    
    # 寻找显著的Y轴谷值 - 使用更敏感的参数
    print(f"   寻找第一个显著谷值...")
    
    # 使用更敏感的参数来检测谷值
    valleys, valley_properties = find_peaks(-y_search, prominence=10, distance=60)  # 0.5秒内的谷值
    
    # 🔧 添加高度阈值过滤：只有小于850mm的才算有效谷值
    height_threshold = 850.0  # mm
    print(f"   高度阈值过滤: 只考虑小于{height_threshold}mm的谷值")
    
    # 打印找到的所有谷值
    if len(valleys) > 0:
        valley_times = [time_search[v] for v in valleys]
        valley_heights = [y_search[v] for v in valleys]
        print(f"   找到{len(valleys)}个候选谷值:")
        for i, (t, h) in enumerate(zip(valley_times, valley_heights)):
            status = "✅" if h < height_threshold else "❌"
            print(f"     谷值{i+1}: t={t:.1f}s, h={h:.0f}mm {status}")
        
        # 过滤出符合高度阈值的谷值
        valid_valleys = []
        valid_valley_times = []
        valid_valley_heights = []
        
        for i, valley_idx in enumerate(valleys):
            height = y_search[valley_idx]
            if height < height_threshold:
                valid_valleys.append(valley_idx)
                valid_valley_times.append(valley_times[i])
                valid_valley_heights.append(height)
        
        if len(valid_valleys) > 0:
            print(f"   ✅ 有效谷值数量: {len(valid_valleys)}个")
            
            # 选择第一个有效谷值作为策略标志
            first_valley_idx = valid_valleys[0]
            strategy_event_time = valid_valley_times[0]
            strategy_height = valid_valley_heights[0]
            strategy_event_idx = first_valley_idx + search_start_idx
            strategy_type = "first_valid_valley"
            
            print(f"✅ 选择第一个有效谷值作为策略标志: t={strategy_event_time:.3f}s, h={strategy_height:.1f}mm")
        else:
            print("❌ 未找到符合高度阈值的有效谷值，使用搜索范围中点")
            strategy_event_time = (search_start + search_end) / 2
            strategy_type = "fallback_no_valid_valley"
            strategy_event_idx = int((strategy_event_time - csv_time[0]) * 120)
    else:
        print("❌ 未找到显著谷值，使用搜索范围中点")
        strategy_event_time = (search_start + search_end) / 2
        strategy_type = "fallback"
        strategy_event_idx = int((strategy_event_time - csv_time[0]) * 120)
    
    # 步骤2: 使用精确分析得出的最佳前推时间
    if offset_seconds is None:
        offset_seconds = 2.35  # 根据之前的analyze_motion_start.py精确分析得出的最优值
        print(f"📊 步骤2: 使用精确分析得出的最佳前推时间")
        print(f"   前推时长: {offset_seconds}s (基于运动复杂度和速度的精确分析)")
    
    print(f"   最终前推时长: {offset_seconds}s")
    
    # 步骤3: 计算最终策略起始点
    strategy_start_time = strategy_event_time - offset_seconds
    strategy_start_idx = np.argmin(np.abs(csv_time - strategy_start_time))
    
    # 确保不早于搜索开始
    if strategy_start_idx < search_start_idx:
        strategy_start_idx = search_start_idx
        strategy_start_time = csv_time[strategy_start_idx]
        print(f"⚠️ 调整起始点到搜索范围内")
    
    actual_start_time = csv_time[strategy_start_idx]
    
    print(f"✅ 确定策略起始点:")
    print(f"   策略标志类型: {strategy_type}")
    print(f"   策略事件时间: {strategy_event_time:.3f}s")
    print(f"   智能前推时长: {offset_seconds:.1f}s")
    print(f"   计算起始时间: {strategy_start_time:.3f}s")
    print(f"   实际起始时间: {actual_start_time:.3f}s (帧{strategy_start_idx})")
    print(f"   起始位置: X={pos_x[strategy_start_idx]:.1f}, Y={pos_y[strategy_start_idx]:.1f}, Z={pos_z[strategy_start_idx]:.1f} mm")
    
    return strategy_start_idx

def detect_motion_start_simple(pos_x, pos_y, pos_z, csv_time, baseline_duration=2.0, threshold_factor=3.0):
    """简单的运动起始点检测（备用方法）"""
    print(f"🔍 使用简单运动检测...")
    
    # 建立基线（前N秒的数据）
    baseline_mask = csv_time <= baseline_duration
    baseline_indices = np.where(baseline_mask)[0]
    
    if len(baseline_indices) < 10:
        print("❌ 基线数据不足，使用默认起始点")
        return int(2.0 * 120)  # 2秒处
    
    # 计算基线变化统计
    baseline_x = pos_x[baseline_mask]
    baseline_y = pos_y[baseline_mask]  
    baseline_z = pos_z[baseline_mask]
    
    x_std = np.std(np.diff(baseline_x))
    y_std = np.std(np.diff(baseline_y))
    z_std = np.std(np.diff(baseline_z))
    
    # 设置运动检测阈值
    x_threshold = threshold_factor * x_std
    y_threshold = threshold_factor * y_std
    z_threshold = threshold_factor * z_std
    
    # 计算位置变化
    pos_diff_x = np.abs(np.diff(pos_x))
    pos_diff_y = np.abs(np.diff(pos_y))
    pos_diff_z = np.abs(np.diff(pos_z))
    
    # 检测运动起始点
    motion_detected = (
        (pos_diff_x > x_threshold) |
        (pos_diff_y > y_threshold) |
        (pos_diff_z > z_threshold)
    )
    
    baseline_end_idx = baseline_indices[-1]
    
    # 从基线结束后开始检查运动
    for i in range(baseline_end_idx, len(motion_detected)):
        if motion_detected[i]:
            motion_start_idx = i + 1
            break
    else:
        motion_start_idx = baseline_end_idx
    
    motion_start_time = csv_time[motion_start_idx]
    
    print(f"   简单检测起始点: 帧{motion_start_idx}, 时间{motion_start_time:.3f}s")
    
    return motion_start_idx

def load_real_pkl_with_times(pkl_file):
    """加载真实世界PKL文件，获取持续时间"""
    print(f"📂 加载真实世界PKL文件: {os.path.basename(pkl_file)}")
    
    data = joblib.load(pkl_file)
    
    # 处理嵌套结构
    if isinstance(data, dict):
        trajectory_key = None
        for key, value in data.items():
            if isinstance(value, dict) and 'motion_times' in value:
                trajectory_key = key
                break
        
        if trajectory_key:
            pkl_data = data[trajectory_key]
            print(f"   使用轨迹键: '{trajectory_key}'")
        else:
            pkl_data = data
            print(f"   使用顶层数据")
    else:
        pkl_data = data
    
    print(f"✅ PKL文件加载成功")
    print(f"   总帧数: {len(pkl_data['motion_times'])}")
    
    # 分析motion_times
    motion_times = pkl_data['motion_times']
    pkl_time_relative = motion_times - motion_times[0]  # 相对时间
    pkl_duration = pkl_time_relative[-1]  # 总持续时间
    
    print(f"   时间范围: {pkl_time_relative[0]:.3f}s - {pkl_time_relative[-1]:.3f}s")
    print(f"   总持续时间: {pkl_duration:.3f}s")
    
    return pkl_data, pkl_time_relative, pkl_duration

def crop_csv_by_duration(df, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, csv_time, motion_start_idx, pkl_duration):
    """根据PKL持续时间裁剪CSV数据"""
    print(f"✂️ 根据PKL持续时间裁剪CSV数据...")
    print(f"   运动起始帧: {motion_start_idx}")
    print(f"   PKL持续时间: {pkl_duration:.3f}s")
    
    # 从运动起始点开始计算
    motion_start_time = csv_time[motion_start_idx]
    target_end_time = motion_start_time + pkl_duration
    
    print(f"   运动起始时间: {motion_start_time:.3f}s")
    print(f"   目标结束时间: {target_end_time:.3f}s")
    
    # 找到结束帧
    end_frame_candidates = np.where(csv_time >= target_end_time)[0]
    if len(end_frame_candidates) == 0:
        # CSV数据不够长，使用所有数据
        end_idx = len(csv_time) - 1
        actual_end_time = csv_time[end_idx]
        print(f"⚠️ CSV数据不够长，使用所有可用数据到 {actual_end_time:.3f}s")
    else:
        end_idx = end_frame_candidates[0]
        actual_end_time = csv_time[end_idx]
        print(f"   实际结束时间: {actual_end_time:.3f}s (帧{end_idx})")
    
    # 裁剪数据
    cropped_indices = slice(motion_start_idx, end_idx + 1)
    
    df_cropped = df.iloc[cropped_indices].copy()
    pos_x_cropped = pos_x[cropped_indices]
    pos_y_cropped = pos_y[cropped_indices]
    pos_z_cropped = pos_z[cropped_indices]
    vel_x_cropped = vel_x[cropped_indices]
    vel_y_cropped = vel_y[cropped_indices]
    vel_z_cropped = vel_z[cropped_indices]
    csv_time_cropped = csv_time[cropped_indices]
    
    # 重新调整时间为从0开始
    csv_time_cropped = csv_time_cropped - csv_time_cropped[0]
    
    cropped_duration = csv_time_cropped[-1]
    duration_match = abs(cropped_duration - pkl_duration)
    
    print(f"✅ 数据裁剪完成:")
    print(f"   原始帧数: {len(csv_time)}")
    print(f"   裁剪后帧数: {len(csv_time_cropped)}")
    print(f"   裁剪后时长: {cropped_duration:.3f}s")
    print(f"   与PKL时长差异: {duration_match:.3f}s")
    
    return df_cropped, pos_x_cropped, pos_y_cropped, pos_z_cropped, vel_x_cropped, vel_y_cropped, vel_z_cropped, csv_time_cropped

def fix_occlusion_outliers(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, threshold_low=1.0, threshold_jump=500.0):
    """修复遮挡异常值"""
    print(f"🔧 修复遮挡异常值...")
    
    def fix_axis(pos_data, axis_name):
        outliers = []
        
        # 检测接近0的值
        near_zero = np.abs(pos_data) < threshold_low
        outliers.extend(np.where(near_zero)[0])
        
        # 检测大跳变
        diff = np.abs(np.diff(pos_data))
        large_jumps = diff > threshold_jump
        jump_indices = np.where(large_jumps)[0] + 1
        outliers.extend(jump_indices)
        
        # 中值滤波检测
        window_size = min(21, len(pos_data) // 10)
        if window_size % 2 == 0:
            window_size += 1
        
        median_filtered = medfilt(pos_data, kernel_size=window_size)
        residuals = np.abs(pos_data - median_filtered)
        threshold_med = np.median(residuals) + 3 * np.std(residuals)
        median_outliers = np.where(residuals > threshold_med)[0]
        outliers.extend(median_outliers)
        
        outliers = np.unique(outliers)
        
        if len(outliers) > 0:
            print(f"   {axis_name}轴: 修复{len(outliers)}个异常点 ({len(outliers)/len(pos_data)*100:.1f}%)")
            
            # 插值修复
            fixed_data = pos_data.copy()
            valid_indices = np.setdiff1d(np.arange(len(pos_data)), outliers)
            
            if len(valid_indices) >= 2:
                try:
                    if len(valid_indices) >= 4:
                        interp_func = interp1d(valid_indices, pos_data[valid_indices], 
                                             kind='cubic', fill_value='extrapolate')
                    else:
                        interp_func = interp1d(valid_indices, pos_data[valid_indices], 
                                             kind='linear', fill_value='extrapolate')
                    fixed_data[outliers] = interp_func(outliers)
                except Exception as e:
                    print(f"      插值失败，保持原数据: {e}")
            
            return fixed_data
        else:
            print(f"   {axis_name}轴: 无需修复")
            return pos_data.copy()
    
    pos_x_fixed = fix_axis(pos_x, 'X')
    pos_y_fixed = fix_axis(pos_y, 'Y')
    pos_z_fixed = fix_axis(pos_z, 'Z')
    
    # 🎯 同样修复速度数据的异常值
    vel_x_fixed = fix_axis(vel_x, 'Vx')
    vel_y_fixed = fix_axis(vel_y, 'Vy')
    vel_z_fixed = fix_axis(vel_z, 'Vz')
    
    return pos_x_fixed, pos_y_fixed, pos_z_fixed, vel_x_fixed, vel_y_fixed, vel_z_fixed

def coordinate_transform_to_robot(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, csv_time, xy_mapping="normal", robot_orientation="forward"):
    """坐标系转换：从CSV坐标系转换到机器人坐标系
    
    Args:
        xy_mapping (str): X和Y的映射方式
            - "normal": CSV X->机器人X, CSV Z->机器人Y (默认)
            - "swapped": CSV Z->机器人X, CSV X->机器人Y (交换X和Y)
        robot_orientation (str): 机器人朝向
            - "forward": 朝向+X方向 (默认)
            - "backward": 朝向-X方向 (后退)
            - "left": 朝向+Y方向 (左转)
            - "right": 朝向-Y方向 (右转)
    """
    print(f"🔄 进行坐标系转换...")
    print(f"   XY映射模式: {xy_mapping}")
    print(f"   机器人朝向: {robot_orientation}")
    
    if xy_mapping == "normal":
        print(f"   CSV -> 机器人坐标系映射:")
        print(f"     CSV X -> 机器人 X")
        print(f"     CSV Y -> 机器人 Z (高度)")
        print(f"     CSV Z -> 机器人 Y")
        
        # 位置转换：CSV(X,Y,Z) -> 机器人(X,Z,Y)，单位mm->m
        robot_pos_x = pos_x / 1000.0    # CSV X -> 机器人 X (mm->m)
        robot_pos_y = pos_z / 1000.0    # CSV Z -> 机器人 Y (mm->m)
        robot_pos_z = pos_y / 1000.0    # CSV Y -> 机器人 Z (mm->m)
        
        # 🎯 速度转换：使用动捕原始速度数据，单位mm/s->m/s
        robot_vel_x = vel_x / 1000.0    # CSV Vx -> 机器人 Vx (mm/s->m/s)
        robot_vel_y = vel_z / 1000.0    # CSV Vz -> 机器人 Vy (mm/s->m/s)
        robot_vel_z = vel_y / 1000.0    # CSV Vy -> 机器人 Vz (mm/s->m/s)
        
    elif xy_mapping == "swapped":
        print(f"   CSV -> 机器人坐标系映射 (交换X和Y):")
        print(f"     CSV X -> 机器人 Y")
        print(f"     CSV Y -> 机器人 Z (高度)")
        print(f"     CSV Z -> 机器人 X")
        
        # 位置转换：CSV(X,Y,Z) -> 机器人(Z,X,Y)，单位mm->m (交换X和Y)
        robot_pos_x = pos_z / 1000.0    # CSV Z -> 机器人 X (mm->m)
        robot_pos_y = pos_x / 1000.0    # CSV X -> 机器人 Y (mm->m)
        robot_pos_z = pos_y / 1000.0    # CSV Y -> 机器人 Z (mm->m)
        
        # 🎯 速度转换：使用动捕原始速度数据，单位mm/s->m/s (交换X和Y)
        robot_vel_x = vel_z / 1000.0    # CSV Vz -> 机器人 Vx (mm/s->m/s)
        robot_vel_y = vel_x / 1000.0    # CSV Vx -> 机器人 Vy (mm/s->m/s)
        robot_vel_z = vel_y / 1000.0    # CSV Vy -> 机器人 Vz (mm/s->m/s)
        
    else:
        raise ValueError(f"不支持的XY映射模式: {xy_mapping}，请使用 'normal' 或 'swapped'")
    
    # 关键修复：将第一帧对齐到 [0, 0, 0.8]
    print(f"🔧 对齐第一帧到 [0, 0, 0.8]...")
    initial_offset_x = 0.0 - robot_pos_x[0]
    initial_offset_y = 0.0 - robot_pos_y[0]
    initial_offset_z = 0.8 - robot_pos_z[0]  # 目标高度0.8m
    
    print(f"   初始偏移: X={initial_offset_x:.3f}, Y={initial_offset_y:.3f}, Z={initial_offset_z:.3f} m")
    
    # 应用偏移到位置（速度不需要偏移，因为是相对量）
    robot_pos_x = robot_pos_x + initial_offset_x
    robot_pos_y = robot_pos_y + initial_offset_y
    robot_pos_z = robot_pos_z + initial_offset_z
    
    # 🎯 使用动捕原始速度数据，速度作为相对量不需要偏移
    print(f"🎯 使用动捕原始速度数据（无需偏移）...")
    
    # 🔧 根据机器人朝向调整轨迹
    print(f"🔧 调整机器人朝向: {robot_orientation}")
    
    if robot_orientation == "forward":
        # 默认朝向+X，无需调整
        print(f"   保持默认朝向: +X方向")
        pass
    elif robot_orientation == "backward":
        # 朝向-X方向，将整个轨迹旋转180度
        print(f"   调整朝向为-X方向（旋转180度）")
        robot_pos_x = -robot_pos_x
        robot_pos_y = -robot_pos_y
        robot_vel_x = -robot_vel_x
        robot_vel_y = -robot_vel_y
    elif robot_orientation == "left":
        # 朝向+Y方向，将X和Y互换并调整符号
        print(f"   调整朝向为+Y方向（逆时针90度）")
        temp_pos_x = robot_pos_x.copy()
        temp_vel_x = robot_vel_x.copy()
        robot_pos_x = -robot_pos_y
        robot_pos_y = temp_pos_x
        robot_vel_x = -robot_vel_y
        robot_vel_y = temp_vel_x
    elif robot_orientation == "right":
        # 朝向-Y方向，将X和Y互换并调整符号
        print(f"   调整朝向为-Y方向（顺时针90度）")
        temp_pos_x = robot_pos_x.copy()
        temp_vel_x = robot_vel_x.copy()
        robot_pos_x = robot_pos_y
        robot_pos_y = -temp_pos_x
        robot_vel_x = robot_vel_y
        robot_vel_y = -temp_vel_x
    else:
        raise ValueError(f"不支持的机器人朝向: {robot_orientation}，请使用 'forward', 'backward', 'left', 'right'")
    
    # 验证对齐结果
    print(f"   验证对齐结果:")
    print(f"     第一帧位置: X={robot_pos_x[0]:.6f}, Y={robot_pos_y[0]:.6f}, Z={robot_pos_z[0]:.6f}")
    print(f"     Z轴(高度)应为0.8m: {robot_pos_z[0]:.6f} ✅" if abs(robot_pos_z[0] - 0.8) < 0.001 else f"     ❌ Z轴对齐错误")
    
    print(f"✅ 坐标系转换完成")
    print(f"   转换后位置范围:")
    print(f"     机器人X: [{robot_pos_x.min():.3f}, {robot_pos_x.max():.3f}] m")
    print(f"     机器人Y: [{robot_pos_y.min():.3f}, {robot_pos_y.max():.3f}] m") 
    print(f"     机器人Z: [{robot_pos_z.min():.3f}, {robot_pos_z.max():.3f}] m (高度)")
    print(f"   转换后速度范围:")
    print(f"     机器人Vx: [{robot_vel_x.min():.3f}, {robot_vel_x.max():.3f}] m/s")
    print(f"     机器人Vy: [{robot_vel_y.min():.3f}, {robot_vel_y.max():.3f}] m/s")
    print(f"     机器人Vz: [{robot_vel_z.min():.3f}, {robot_vel_z.max():.3f}] m/s")
    print(f"   初始位置: [{robot_pos_x[0]:.3f}, {robot_pos_y[0]:.3f}, {robot_pos_z[0]:.3f}] m")
    print(f"   初始速度: [{robot_vel_x[0]:.3f}, {robot_vel_y[0]:.3f}, {robot_vel_z[0]:.3f}] m/s")
    
    return {
        'pos': np.column_stack([robot_pos_x, robot_pos_y, robot_pos_z]),
        'vel': np.column_stack([robot_vel_x, robot_vel_y, robot_vel_z]),
        'time': csv_time
    }

def interpolate_to_pkl_times(mocap_robot_data, pkl_time):
    """将动捕数据插值到PKL的真实时间点"""
    print(f"📊 插值到PKL时间点...")
    
    mocap_time = mocap_robot_data['time']
    mocap_pos = mocap_robot_data['pos']
    mocap_vel = mocap_robot_data['vel']
    
    print(f"   动捕时间范围: {mocap_time[0]:.3f}s - {mocap_time[-1]:.3f}s")
    print(f"   PKL时间范围: {pkl_time[0]:.3f}s - {pkl_time[-1]:.3f}s")
    
    # 插值到PKL时间点
    interpolated_pos = np.zeros((len(pkl_time), 3))
    interpolated_vel = np.zeros((len(pkl_time), 3))
    
    for i in range(3):
        # 位置插值
        pos_interp = interp1d(mocap_time, mocap_pos[:, i], 
                             kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_pos[:, i] = pos_interp(pkl_time)
        
        # 速度插值
        vel_interp = interp1d(mocap_time, mocap_vel[:, i], 
                             kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_vel[:, i] = vel_interp(pkl_time)
    
    print(f"✅ 插值完成")
    print(f"   插值后位置范围:")
    print(f"     X: [{interpolated_pos[:, 0].min():.3f}, {interpolated_pos[:, 0].max():.3f}] m")
    print(f"     Y: [{interpolated_pos[:, 1].min():.3f}, {interpolated_pos[:, 1].max():.3f}] m")
    print(f"     Z: [{interpolated_pos[:, 2].min():.3f}, {interpolated_pos[:, 2].max():.3f}] m")
    
    return {
        'pos': interpolated_pos.astype(np.float32),
        'vel': interpolated_vel.astype(np.float32)
    }

def merge_data_to_pkl(pkl_data, interpolated_data, xml_path=None):
    """将插值后的动捕数据融合到PKL数据中"""
    print(f"🔗 融合数据到PKL...")
    
    # 创建融合后的数据
    merged_data = pkl_data.copy()
    
    # 更新位置和速度信息
    merged_data['root_trans_offset'] = interpolated_data['pos']
    merged_data['root_lin_vel'] = interpolated_data['vel']
    
    # 🔧 修复pose_aa计算 - 与mujoco_track.py一致
    if xml_path is not None:
        fixed_pose_aa = fix_pose_aa(merged_data, xml_path)
        merged_data['pose_aa'] = fixed_pose_aa
        print(f"   ✅ 已修复pose_aa计算")
    else:
        print(f"   ⚠️ 未提供XML路径，跳过pose_aa修复")
    
    print(f"✅ 数据融合完成")
    print(f"   更新后位置范围: [{merged_data['root_trans_offset'].min():.3f}, {merged_data['root_trans_offset'].max():.3f}]")
    print(f"   更新后速度范围: [{merged_data['root_lin_vel'].min():.3f}, {merged_data['root_lin_vel'].max():.3f}]")
    
    # 验证机器人高度
    initial_height = merged_data['root_trans_offset'][0, 2]  # Z轴是高度
    print(f"   验证机器人初始高度: {initial_height:.3f} m")
    
    return merged_data

def visualize_processing_results(original_time, original_pos_y, motion_start_idx, 
                               cropped_time, cropped_pos_z, pkl_time, interpolated_pos, save_path):
    """可视化处理结果"""
    print("📊 生成处理结果可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 运动起始点检测
    axes[0, 0].plot(original_time, original_pos_y, 'b-', linewidth=1, alpha=0.7, label='Original Y-position')
    axes[0, 0].axvline(x=original_time[motion_start_idx], color='red', linestyle='--', 
                      label=f'Motion start: {original_time[motion_start_idx]:.2f}s')
    axes[0, 0].axvline(x=2.0, color='orange', linestyle=':', alpha=0.7, label='Baseline end: 2s')
    axes[0, 0].set_title('Motion Start Detection', fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Y Position (mm)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2: 时长裁剪结果
    axes[0, 1].plot(cropped_time, cropped_pos_z*1000, 'g-', linewidth=1, label='Cropped Z-height')
    axes[0, 1].axhline(y=800, color='orange', linestyle='--', alpha=0.7, label='0.8m target')
    axes[0, 1].set_title('Duration Cropping Result', fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Z Height (mm)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 子图3: PKL时间点插值
    axes[1, 0].plot(cropped_time, cropped_pos_z*1000, 'g-', linewidth=1, alpha=0.7, label='Cropped data')
    axes[1, 0].plot(pkl_time, interpolated_pos[:, 2]*1000, 'ro', markersize=2, label='Interpolated to PKL times')
    axes[1, 0].set_title('Interpolation to PKL Times', fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Z Height (mm)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4: 最终3D轨迹
    axes[1, 1].plot(interpolated_pos[:, 0], interpolated_pos[:, 1], 'purple', linewidth=2, label='3D trajectory')
    axes[1, 1].scatter(interpolated_pos[0, 0], interpolated_pos[0, 1], c='green', s=50, label='Start', zorder=5)
    axes[1, 1].scatter(interpolated_pos[-1, 0], interpolated_pos[-1, 1], c='red', s=50, label='End', zorder=5)
    axes[1, 1].set_title('Final 3D Trajectory (X-Y)', fontweight='bold')
    axes[1, 1].set_xlabel('X Position (m)')
    axes[1, 1].set_ylabel('Y Position (m)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 处理结果可视化已保存: {save_path}")
    plt.show()

def save_merged_pkl(merged_data, original_pkl_file, output_dir):
    """保存融合后的PKL文件（自动转换为motion0格式）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(original_pkl_file).replace('.pkl', '')
    

    
    # 转换为motion0格式
    motion_data = {
        'motion0': merged_data
    }
    
    output_file_motion0 = f"{timestamp}_correct_aligned_{base_name}_motion0.pkl"
    output_path_motion0 = os.path.join(output_dir, output_file_motion0)
    joblib.dump(motion_data, output_path_motion0)
    print(f"💾 motion0格式PKL已保存: {output_file_motion0}")
    
    # 验证转换结果
    verify_data = joblib.load(output_path_motion0)
    print(f"🔍 验证motion0格式:")
    print(f"  顶层键: {list(verify_data.keys())}")
    print(f"  motion0子键: {list(verify_data['motion0'].keys())}")
    print(f"  dof形状: {verify_data['motion0']['dof'].shape}")
    
    return output_path_motion0  # 返回motion0格式的路径

def main():
    # ========== 📝 配置区域 ==========
    
    # 文件路径 - 🔧 请修改为你的实际文件路径
    csv_file = "/home/user/pbhc-main- cqh723/pbhc-main/real-data-motion-process/output/fixed_csv/31-23131-asap.csv"  # 📝 Step1输出的修复CSV文件
    pkl_file = "/home/user/pbhc-main- cqh723/pbhc-main/final-aligine/origine-data/8.1-pkl/20250731_230438_real_motion_trajectory_select_sleep_protected.pkl"  # 📝 机器人PKL文件
    output_dir = "/home/user/pbhc-main- cqh723/pbhc-main/real-data-motion-process/output"  # 📝 输出到统一目录
    
    # XML文件路径 - 用于修复pose_aa计算
    xml_path = "asap_mujoco_sim/g1/g1_23dof_lock_wrist.xml"  # ✅ 修正为正确的相对路径
    
    # 🎯 XY映射模式配置 - 在这里调整X和Y的对齐方式
    # "normal": CSV X->机器人X, CSV Z->机器人Y (默认映射)
    # "swapped": CSV Z->机器人X, CSV X->机器人Y (交换X和Y)
    xy_mapping_mode = "normal"  # 🔧 修改这里来调整X和Y的映射！
    
    # 🎯 机器人朝向配置 - 在这里调整机器人的朝向
    # "forward": 朝向+X方向 (默认)
    # "backward": 朝向-X方向
    # "left": 朝向+Y方向 (左转90度)
    # "right": 朝向-Y方向 (右转90度)
    robot_orientation_mode = "right"  # 🔧 恢复原始设置，保持数据对齐正确性
    
    # ===============================
    
    print("🎯 正确的时间对齐处理")
    print("=" * 60)
    print(f"🎯 当前XY映射模式: {xy_mapping_mode}")
    if xy_mapping_mode == "normal":
        print("   📍 CSV X -> 机器人 X (前后方向)")
        print("   📍 CSV Y -> 机器人 Z (高度方向)")  
        print("   📍 CSV Z -> 机器人 Y (左右方向)")
    elif xy_mapping_mode == "swapped":
        print("   📍 CSV X -> 机器人 Y (左右方向) [交换]")
        print("   📍 CSV Y -> 机器人 Z (高度方向)")
        print("   📍 CSV Z -> 机器人 X (前后方向) [交换]")
    
    print(f"🎯 当前机器人朝向: {robot_orientation_mode}")
    if robot_orientation_mode == "forward":
        print("   🧭 机器人朝向: +X方向 (默认前进)")
    elif robot_orientation_mode == "backward":
        print("   🧭 机器人朝向: -X方向 (向后，解决脚滑)")
    elif robot_orientation_mode == "left":
        print("   🧭 机器人朝向: +Y方向 (左转90度)")
    elif robot_orientation_mode == "right":
        print("   🧭 机器人朝向: -Y方向 (右转90度)")
    print("=" * 60)
    print("📝 处理流程:")
    print("  1️⃣ 加载CSV数据")
    print("  2️⃣ 检测真正的运动起始点")
    print("  3️⃣ 根据PKL持续时长裁剪CSV")
    print("  4️⃣ 修复遮挡异常值")
    print("  5️⃣ 坐标系转换")
    print("  6️⃣ 插值到PKL时间点")
    print("  7️⃣ 融合到PKL文件")
    print("  8️⃣ 修复pose_aa计算")
    
    # 检查文件
    if not os.path.exists(csv_file):
        print(f"❌ CSV文件不存在: {csv_file}")
        return
    
    if not os.path.exists(pkl_file):
        print(f"❌ PKL文件不存在: {pkl_file}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 加载CSV数据
        print("\n📊 第1步: 加载CSV数据")
        df, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, csv_time = load_csv_data_asap(csv_file)
        
        # 2. 检测策略起始点
        print("\n🔍 第2步: 检测策略起始点")
        motion_start_idx = detect_motion_start_by_significant_pattern(pos_x, pos_y, pos_z, csv_time)
        
        # 3. 加载PKL获取持续时间
        print("\n📂 第3步: 加载PKL获取持续时间")
        pkl_data, pkl_time, pkl_duration = load_real_pkl_with_times(pkl_file)
        
        # 4. 根据PKL持续时间裁剪CSV
        print("\n✂️ 第4步: 根据PKL持续时间裁剪CSV")
        df_cropped, pos_x_cropped, pos_y_cropped, pos_z_cropped, vel_x_cropped, vel_y_cropped, vel_z_cropped, csv_time_cropped = crop_csv_by_duration(
            df, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, csv_time, motion_start_idx, pkl_duration)
        
        # 5. 修复遮挡异常值
        print("\n🔧 第5步: 修复遮挡异常值")
        pos_x_fixed, pos_y_fixed, pos_z_fixed, vel_x_fixed, vel_y_fixed, vel_z_fixed = fix_occlusion_outliers(
            pos_x_cropped, pos_y_cropped, pos_z_cropped, vel_x_cropped, vel_y_cropped, vel_z_cropped)
        
        # 6. 坐标系转换
        print("\n🔄 第6步: 坐标系转换")
        print(f"   使用XY映射模式: {xy_mapping_mode}")
        print(f"   使用机器人朝向: {robot_orientation_mode}")
        
        mocap_robot_data = coordinate_transform_to_robot(
            pos_x_fixed, pos_y_fixed, pos_z_fixed, vel_x_fixed, vel_y_fixed, vel_z_fixed, csv_time_cropped, 
            xy_mapping=xy_mapping_mode, robot_orientation=robot_orientation_mode)
        
        # 7. 插值到PKL时间点
        print("\n📊 第7步: 插值到PKL时间点")
        interpolated_data = interpolate_to_pkl_times(mocap_robot_data, pkl_time)
        
        # 8. 融合到PKL
        print("\n🔗 第8步: 融合数据到PKL")
        merged_data = merge_data_to_pkl(pkl_data, interpolated_data, xml_path)
        
        # 9. 可视化
        print("\n📊 第9步: 生成可视化")
        vis_path = os.path.join(output_dir, "correct_alignment_results.png")
        visualize_processing_results(csv_time, pos_y, motion_start_idx, 
                                   csv_time_cropped, mocap_robot_data['pos'][:, 2], 
                                   pkl_time, interpolated_data['pos'], vis_path)
        
        # 10. 保存结果
        print("\n💾 第10步: 保存结果")
        output_path = save_merged_pkl(merged_data, pkl_file, output_dir)
        
        print(f"\n✅ 正确时间对齐完成!")
        print(f"📊 处理总结:")
        print(f"   输入CSV: {os.path.basename(csv_file)} ({len(csv_time)}帧, {csv_time[-1]:.1f}s)")
        print(f"   输入PKL: {os.path.basename(pkl_file)} ({len(pkl_time)}帧, {pkl_duration:.1f}s)")
        print(f"   运动起始: 第{motion_start_idx}帧 ({csv_time[motion_start_idx]:.2f}s)")
        print(f"   裁剪后CSV: {len(csv_time_cropped)}帧 ({csv_time_cropped[-1]:.1f}s)")
        print(f"   时长匹配度: {abs(csv_time_cropped[-1] - pkl_duration):.3f}s 差异")
        print(f"📁 输出文件: {os.path.basename(output_path)}")
        print(f"📊 可视化文件: correct_alignment_results.png")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 