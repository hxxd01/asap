#!/usr/bin/env python3
"""
改进版动捕数据遮挡异常值修复 - Step 1 Improved
更智能的异常值检测和修复策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from scipy.stats import zscore

def load_csv_data_asap(csv_file):
    """加载ASAP格式的CSV数据"""
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
    
    # 提取位置数据 (mm)
    pos_x = df['XToGlobal1'].values
    pos_y = df['YToGlobal1'].values  
    pos_z = df['ZToGlobal1'].values
    
    # 提取速度数据 (mm/s) - 处理空值
    vel_x = pd.to_numeric(df['VxToGlobal1'], errors='coerce').values
    vel_y = pd.to_numeric(df['VyToGlobal1'], errors='coerce').values
    vel_z = pd.to_numeric(df['VzToGlobal1'], errors='coerce').values
    
    # 将空值替换为0
    vel_x = np.nan_to_num(vel_x, nan=0.0)
    vel_y = np.nan_to_num(vel_y, nan=0.0)
    vel_z = np.nan_to_num(vel_z, nan=0.0)
    
    print(f"✅ 数据加载成功")
    print(f"   总帧数: {len(df)}")
    print(f"   采样率: 120Hz")
    print(f"   持续时间: {len(df)/120:.2f}秒")
    print(f"   位置范围: X[{pos_x.min():.1f}, {pos_x.max():.1f}] Y[{pos_y.min():.1f}, {pos_y.max():.1f}] Z[{pos_z.min():.1f}, {pos_z.max():.1f}] mm")
    
    return df, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z

def detect_occlusion_outliers_improved(pos_data, axis_name, adaptive_thresholds=True):
    """改进的异常值检测"""
    print(f"🔍 检测{axis_name}轴遮挡异常值（改进版）...")
    
    outliers = set()
    
    # 计算数据的统计特征
    data_range = pos_data.max() - pos_data.min()
    data_std = np.std(pos_data)
    data_median = np.median(pos_data)
    
    # 🔧 改进1: 自适应阈值
    if adaptive_thresholds:
        # 基于数据分布动态调整阈值
        threshold_low = max(1.0, data_std * 0.1)  # 动态低值阈值
        threshold_jump = max(100.0, data_std * 5)  # 动态跳变阈值
    else:
        threshold_low = 1.0
        threshold_jump = 500.0
    
    print(f"   自适应阈值: 低值={threshold_low:.1f}mm, 跳变={threshold_jump:.1f}mm")
    
    # 方法1: 检测接近0的值 - 改进版
    # 只有当数据明显偏离中心且接近0时才认为是异常
    if abs(data_median) > threshold_low * 2:  # 只有当数据中心不在0附近时才检测
        near_zero = np.abs(pos_data) < threshold_low
        zero_outliers = np.where(near_zero)[0]
        outliers.update(zero_outliers)
        print(f"   方法1(接近0): 检测到 {len(zero_outliers)} 个异常点")
    else:
        print(f"   方法1(接近0): 跳过，数据中心在原点附近")
    
    # 方法2: 检测突然的大跳变 - 改进版
    diff = np.abs(np.diff(pos_data))
    # 使用动态阈值，考虑局部变化
    rolling_std = pd.Series(diff).rolling(window=min(20, len(diff)//5), center=True).std()
    dynamic_threshold = np.maximum(threshold_jump, rolling_std * 3).fillna(threshold_jump)
    # 确保长度匹配
    assert len(dynamic_threshold) == len(diff), f"长度不匹配: dynamic_threshold={len(dynamic_threshold)}, diff={len(diff)}"
    large_jumps = diff > dynamic_threshold
    
    jump_indices = np.where(large_jumps)[0] + 1
    outliers.update(jump_indices)
    print(f"   方法2(大跳变): 检测到 {len(jump_indices)} 个异常点")
    
    # 方法3: 使用中值滤波检测异常值 - 改进版
    window_size = min(21, len(pos_data) // 10)
    if window_size % 2 == 0:
        window_size += 1
    
    median_filtered = medfilt(pos_data, kernel_size=window_size)
    residuals = np.abs(pos_data - median_filtered)
    
    # 🔧 改进2: 使用更保守的阈值
    threshold_med = np.median(residuals) + 4 * np.std(residuals)  # 从3倍改为4倍
    median_outliers = np.where(residuals > threshold_med)[0]
    outliers.update(median_outliers)
    print(f"   方法3(中值滤波): 检测到 {len(median_outliers)} 个异常点")
    
    # 🔧 改进3: 后处理过滤
    outliers = np.array(sorted(outliers))
    
    # 过滤掉孤立的单点异常（可能是正常的快速动作）
    if len(outliers) > 0:
        # 计算每个异常点的邻域密度
        isolated_outliers = []
        for i, outlier_idx in enumerate(outliers):
            # 检查前后5个点的范围内是否有其他异常点
            neighbor_range = 5
            neighbors = outliers[
                (outliers >= outlier_idx - neighbor_range) & 
                (outliers <= outlier_idx + neighbor_range) &
                (outliers != outlier_idx)
            ]
            if len(neighbors) == 0:  # 孤立点
                isolated_outliers.append(outlier_idx)
        
        if isolated_outliers:
            print(f"   过滤孤立异常点: {len(isolated_outliers)} 个")
            outliers = np.setdiff1d(outliers, isolated_outliers)
    
    print(f"   最终检测到 {len(outliers)} 个异常点")
    print(f"   异常点比例: {len(outliers)/len(pos_data)*100:.2f}%")
    
    if len(outliers) > 0:
        print(f"   异常值范围: [{pos_data[outliers].min():.1f}, {pos_data[outliers].max():.1f}] mm")
    
    return outliers

def interpolate_outliers_safe(pos_data, outliers, method='cubic'):
    """安全的插值修复异常值"""
    if len(outliers) == 0:
        return pos_data.copy()
    
    print(f"🔧 使用安全{method}插值修复异常值...")
    
    # 创建修复后的数据副本
    fixed_data = pos_data.copy()
    
    # 获取有效数据点的索引
    valid_indices = np.setdiff1d(np.arange(len(pos_data)), outliers)
    
    if len(valid_indices) < 2:
        print("❌ 有效数据点太少，无法进行插值")
        return pos_data.copy()
    
    # 计算合理的数据范围
    data_min = np.percentile(pos_data[valid_indices], 1)   # 1%分位数
    data_max = np.percentile(pos_data[valid_indices], 99)  # 99%分位数
    data_range = data_max - data_min
    
    # 创建插值函数
    try:
        if method == 'cubic' and len(valid_indices) >= 4:
            # 🔧 改进4: 限制外推范围
            interp_func = interp1d(valid_indices, pos_data[valid_indices], 
                                 kind='cubic', bounds_error=False, 
                                 fill_value=(pos_data[valid_indices[0]], pos_data[valid_indices[-1]]))
        else:
            interp_func = interp1d(valid_indices, pos_data[valid_indices], 
                                 kind='linear', bounds_error=False,
                                 fill_value=(pos_data[valid_indices[0]], pos_data[valid_indices[-1]]))
        
        # 对异常值进行插值
        interpolated_values = interp_func(outliers)
        
        # 🔧 改进5: 验证插值结果
        # 确保插值结果在合理范围内
        valid_interpolated = (interpolated_values >= data_min - data_range * 0.2) & \
                           (interpolated_values <= data_max + data_range * 0.2)
        
        if not np.all(valid_interpolated):
            print(f"   警告: {np.sum(~valid_interpolated)} 个插值结果超出合理范围，使用最近邻替代")
            # 对超出范围的值使用最近有效点的值
            for i, outlier_idx in enumerate(outliers):
                if not valid_interpolated[i]:
                    # 找到最近的有效点
                    distances = np.abs(valid_indices - outlier_idx)
                    nearest_valid_idx = valid_indices[np.argmin(distances)]
                    interpolated_values[i] = pos_data[nearest_valid_idx]
        
        fixed_data[outliers] = interpolated_values
        
        print(f"✅ 安全插值修复完成")
        print(f"   修复前范围: [{pos_data.min():.1f}, {pos_data.max():.1f}] mm")
        print(f"   修复后范围: [{fixed_data.min():.1f}, {fixed_data.max():.1f}] mm")
        
        # 验证修复质量
        if len(outliers) > 0:
            max_change = np.max(np.abs(fixed_data[outliers] - pos_data[outliers]))
            print(f"   最大修复变化: {max_change:.1f} mm")
        
    except Exception as e:
        print(f"❌ 插值失败: {e}")
        return pos_data.copy()
    
    return fixed_data

def analyze_outlier_segments(outliers):
    """分析异常值的连续段"""
    if len(outliers) == 0:
        return []
    
    segments = []
    start = outliers[0]
    
    for i in range(1, len(outliers)):
        # 如果不连续，结束当前段
        if outliers[i] - outliers[i-1] > 1:
            segments.append((start, outliers[i-1]))
            start = outliers[i]
    
    # 添加最后一段
    segments.append((start, outliers[-1]))
    
    print(f"   异常值分为 {len(segments)} 个连续段:")
    for i, (s, e) in enumerate(segments):
        print(f"     段{i+1}: [{s}, {e}] 长度={e-s+1}")
    
    return segments

def visualize_fix_results(original_pos, fixed_pos, outliers, axis_name, save_path=None):
    """可视化修复结果"""
    print(f"📊 生成{axis_name}轴修复可视化...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    time_axis = np.arange(len(original_pos)) / 120.0  # 120Hz
    
    # 子图1: 原始数据和异常值
    ax1.plot(time_axis, original_pos, 'b-', linewidth=1, alpha=0.7, label='Original Data')
    if len(outliers) > 0:
        ax1.scatter(time_axis[outliers], original_pos[outliers], c='red', s=20, label=f'Outliers ({len(outliers)} points)')
    ax1.set_title(f'{axis_name}-axis Position Data - Outlier Detection (Improved)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Position (mm)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: 修复后的数据
    ax2.plot(time_axis, original_pos, 'b-', linewidth=1, alpha=0.5, label='Original Data')
    ax2.plot(time_axis, fixed_pos, 'g-', linewidth=2, label='Fixed Data')
    if len(outliers) > 0:
        ax2.scatter(time_axis[outliers], fixed_pos[outliers], c='orange', s=20, label=f'Interpolated Points ({len(outliers)} points)')
    ax2.set_title(f'{axis_name}-axis Position Data - Fix Results (Improved)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Position (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3: 修复差异
    diff = fixed_pos - original_pos
    ax3.plot(time_axis, diff, 'r-', linewidth=1, label='Fix Difference')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title(f'{axis_name}-axis Fix Difference (Improved)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Difference (mm)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{timestamp}_mocap_fix_improved_{axis_name}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {save_path}")
    plt.close()
    
    return save_path

def fix_velocity_data(vel_data, pos_data_fixed, dt=1/120.0):
    """基于修复后的位置数据重新计算速度"""
    print(f"🔧 重新计算速度数据...")
    
    # 使用中心差分计算速度
    vel_fixed = np.gradient(pos_data_fixed, dt)
    
    print(f"   原始速度范围: [{vel_data.min():.1f}, {vel_data.max():.1f}] mm/s")
    print(f"   修复后速度范围: [{vel_fixed.min():.1f}, {vel_fixed.max():.1f}] mm/s")
    
    return vel_fixed

def process_single_csv(csv_file, csv_output_dir, vis_output_dir):
    """处理单个CSV文件"""
    print(f"\n🎯 处理文件: {os.path.basename(csv_file)}")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n📊 第1步: 加载数据")
    df, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = load_csv_data_asap(csv_file)
    
    # 2. 改进的异常值检测
    print("\n🔍 第2步: 改进的异常值检测")
    outliers_x = detect_occlusion_outliers_improved(pos_x, 'X')
    outliers_y = detect_occlusion_outliers_improved(pos_y, 'Y')
    outliers_z = detect_occlusion_outliers_improved(pos_z, 'Z')
    
    # 分析异常值段
    if len(outliers_x) > 0:
        print(f"\n📈 X轴异常值分析:")
        analyze_outlier_segments(outliers_x)
    
    if len(outliers_y) > 0:
        print(f"\n📈 Y轴异常值分析:")
        analyze_outlier_segments(outliers_y)
    
    if len(outliers_z) > 0:
        print(f"\n📈 Z轴异常值分析:")
        analyze_outlier_segments(outliers_z)
    
    # 3. 安全插值修复
    print("\n🔧 第3步: 安全插值修复")
    pos_x_fixed = interpolate_outliers_safe(pos_x, outliers_x)
    pos_y_fixed = interpolate_outliers_safe(pos_y, outliers_y)
    pos_z_fixed = interpolate_outliers_safe(pos_z, outliers_z)
    
    # 4. 重新计算速度
    print("\n⚡ 第4步: 重新计算速度")
    vel_x_fixed = fix_velocity_data(vel_x, pos_x_fixed)
    vel_y_fixed = fix_velocity_data(vel_y, pos_y_fixed)
    vel_z_fixed = fix_velocity_data(vel_z, pos_z_fixed)
    
    # 5. 生成可视化（只保存Y轴）
    print("\n📊 第5步: 生成Y轴可视化")
    vis_paths = []
    base_name = os.path.basename(csv_file).replace('.csv', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 只为Y轴生成可视化
    vis_path = os.path.join(vis_output_dir, f"{timestamp}_step1_improved_mocap_fix_{base_name}_Y.png")
    vis_paths.append(visualize_fix_results(pos_y, pos_y_fixed, outliers_y, 'Y', vis_path))
    
    # 6. 保存修复后的数据
    print("\n💾 第6步: 保存修复后的数据")
    
    # 创建修复后的DataFrame
    df_fixed = df.copy()
    df_fixed['XToGlobal1'] = pos_x_fixed
    df_fixed['YToGlobal1'] = pos_y_fixed
    df_fixed['ZToGlobal1'] = pos_z_fixed
    df_fixed['VxToGlobal1'] = vel_x_fixed
    df_fixed['VyToGlobal1'] = vel_y_fixed
    df_fixed['VzToGlobal1'] = vel_z_fixed
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(csv_file).replace('.csv', '')
    output_file = f"{timestamp}_step1_improved_fixed_{base_name}.csv"
    output_path = os.path.join(csv_output_dir, output_file)
    
    df_fixed.to_csv(output_path, index=False)
    
    print(f"💾 改进修复后的CSV已保存: {output_file}")
    
    # 7. 输出统计摘要
    print(f"\n✅ {os.path.basename(csv_file)} 修复完成!")
    print("📊 修复统计:")
    print(f"   X轴异常值: {len(outliers_x)} 个")
    print(f"   Y轴异常值: {len(outliers_y)} 个")
    print(f"   Z轴异常值: {len(outliers_z)} 个")
    print(f"   总异常值: {len(set(outliers_x) | set(outliers_y) | set(outliers_z))} 个")
    print(f"   数据完整性: {(1 - len(set(outliers_x) | set(outliers_y) | set(outliers_z)) / len(df)) * 100:.1f}%")
    
    if vis_paths:
        print(f"📊 可视化文件: {len(vis_paths)} 个")
        for path in vis_paths:
            print(f"   - {os.path.basename(path)}")
    
    print(f"📁 修复后CSV文件: {output_file}")
    print(f"📂 CSV保存位置: {csv_output_dir}")
    print(f"📊 可视化保存位置: {vis_output_dir}")
    
    return {
        'csv_file': csv_file,
        'output_file': output_file,
        'outliers_x': len(outliers_x),
        'outliers_y': len(outliers_y),
        'outliers_z': len(outliers_z),
        'total_outliers': len(set(outliers_x) | set(outliers_y) | set(outliers_z)),
        'data_integrity': (1 - len(set(outliers_x) | set(outliers_y) | set(outliers_z)) / len(df)) * 100,
        'vis_paths': vis_paths,
        'success': True
    }

def main():
    # 输入文件夹路径
    input_dir = "/home/user/pbhc-main- cqh723/pbhc-main/final-aligine/8.1-csv"
    
    # 输出目录
    base_output_dir = "/home/user/pbhc-main- cqh723/pbhc-main/real-data-motion-process/output"
    csv_output_dir = os.path.join(base_output_dir, "fixed_csv")  # CSV文件夹
    vis_output_dir = os.path.join(base_output_dir, "visualizations")  # 可视化文件夹
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 确保输出目录存在
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 找到所有CSV文件
    csv_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.csv'):
            csv_files.append(os.path.join(input_dir, file))
    
    if not csv_files:
        print(f"❌ 在目录 {input_dir} 中没有找到CSV文件")
        return
    
    print("🎯 批量动捕数据遮挡异常值修复 - Step 1 Improved")
    print("=" * 80)
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 CSV输出目录: {csv_output_dir}")
    print(f"📊 可视化输出目录: {vis_output_dir}")
    print(f"📄 找到 {len(csv_files)} 个CSV文件:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"   {i}. {os.path.basename(csv_file)}")
    
    # 批量处理
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"\n{'='*20} 处理进度: {i}/{len(csv_files)} {'='*20}")
            result = process_single_csv(csv_file, csv_output_dir, vis_output_dir)
            results.append(result)
            if result['success']:
                successful_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"❌ 处理文件 {os.path.basename(csv_file)} 时出错: {e}")
            failed_count += 1
            results.append({
                'csv_file': csv_file,
                'success': False,
                'error': str(e)
            })
    
    # 输出批量处理总结
    print(f"\n{'='*80}")
    print("🎊 批量处理完成!")
    print("📊 总体统计:")
    print(f"   总文件数: {len(csv_files)}")
    print(f"   ✅ 成功处理: {successful_count}")
    print(f"   ❌ 处理失败: {failed_count}")
    print(f"   📈 成功率: {successful_count/len(csv_files)*100:.1f}%")
    
    if successful_count > 0:
        print(f"\n📊 成功处理文件的详细统计:")
        total_outliers = 0
        total_vis_files = 0
        avg_integrity = 0
        
        for result in results:
            if result['success']:
                total_outliers += result['total_outliers']
                total_vis_files += len(result['vis_paths'])
                avg_integrity += result['data_integrity']
                print(f"   📄 {os.path.basename(result['csv_file'])}: "
                      f"异常值={result['total_outliers']}, "
                      f"完整性={result['data_integrity']:.1f}%, "
                      f"可视化={len(result['vis_paths'])}个")
        
        avg_integrity /= successful_count
        print(f"\n📈 汇总统计:")
        print(f"   总异常值修复: {total_outliers} 个")
        print(f"   总可视化文件: {total_vis_files} 个")
        print(f"   平均数据完整性: {avg_integrity:.1f}%")
    
    if failed_count > 0:
        print(f"\n❌ 失败文件列表:")
        for result in results:
            if not result['success']:
                print(f"   - {os.path.basename(result['csv_file'])}: {result.get('error', '未知错误')}")
    
    print(f"\n📂 所有输出文件保存在:")
    print(f"   📁 CSV文件: {csv_output_dir}")
    print(f"   📊 可视化文件: {vis_output_dir}")
    print("🎉 批量处理任务完成!")

if __name__ == "__main__":
    main() 