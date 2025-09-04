#!/usr/bin/env python3
"""
批量合并文件夹中的所有PKL文件
基于merge.py的逻辑，自动扫描文件夹并合并所有PKL文件
"""

import os
import glob
import joblib
from datetime import datetime

def merge_pkl_folder(input_folder, output_file=None, file_pattern="*.pkl"):
    """
    合并文件夹中的所有PKL文件
    
    Args:
        input_folder (str): 输入文件夹路径
        output_file (str): 输出文件名，如果为None则自动生成
        file_pattern (str): 文件匹配模式，默认"*.pkl"
    """
    print(f"🎯 批量合并PKL文件")
    print(f"📁 输入文件夹: {input_folder}")
    print(f"🔍 文件模式: {file_pattern}")
    print("=" * 60)
    
    # 检查输入文件夹
    if not os.path.exists(input_folder):
        print(f"❌ 输入文件夹不存在: {input_folder}")
        return None
    
    # 查找所有PKL文件
    search_pattern = os.path.join(input_folder, file_pattern)
    pkl_files = glob.glob(search_pattern)
    pkl_files.sort()  # 按文件名排序
    
    if len(pkl_files) == 0:
        print(f"❌ 未找到匹配的PKL文件: {search_pattern}")
        return None
    
    print(f"📊 找到 {len(pkl_files)} 个PKL文件:")
    for i, pkl_file in enumerate(pkl_files):
        print(f"   {i+1}. {os.path.basename(pkl_file)}")
    
    # 开始合并
    print(f"\n🔗 开始合并PKL文件...")
    all_motions = {}
    motion_idx = 0
    total_motions = 0
    
    for i, pkl_file in enumerate(pkl_files):
        print(f"\n📂 处理文件 {i+1}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
        
        try:
            # 加载PKL文件
            data = joblib.load(pkl_file)
            
            # 检查数据结构
            if isinstance(data, dict):
                file_motions = 0
                for key in data:
                    new_key = f"motion{motion_idx}"
                    all_motions[new_key] = data[key]
                    print(f"   添加: {key} -> {new_key}")
                    motion_idx += 1
                    file_motions += 1
                    total_motions += 1
                print(f"   ✅ 从此文件添加了 {file_motions} 个motion")
            else:
                # 如果不是字典格式，直接作为单个motion添加
                new_key = f"motion{motion_idx}"
                all_motions[new_key] = data
                print(f"   添加: 整个文件 -> {new_key}")
                motion_idx += 1
                total_motions += 1
                print(f"   ✅ 从此文件添加了 1 个motion")
                
        except Exception as e:
            print(f"   ❌ 加载文件失败: {e}")
            continue
    
    if total_motions == 0:
        print(f"❌ 没有成功加载任何motion数据")
        return None
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{timestamp}_merged_{total_motions}motions.pkl"
    
    # 保存合并结果
    try:
        joblib.dump(all_motions, output_file)
        print(f"\n✅ 合并完成!")
        print(f"📊 总计: {total_motions} 个motion")
        print(f"📁 输出文件: {output_file}")
        print(f"💾 文件大小: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        # 验证保存的文件
        verify_data = joblib.load(output_file)
        print(f"🔍 验证结果:")
        print(f"   顶层键数量: {len(verify_data)}")
        print(f"   motion键: {list(verify_data.keys())}")
        
        # 检查第一个motion的结构
        if len(verify_data) > 0:
            first_key = list(verify_data.keys())[0]
            first_motion = verify_data[first_key]
            if isinstance(first_motion, dict):
                print(f"   {first_key}包含字段: {list(first_motion.keys())}")
                if 'dof' in first_motion:
                    print(f"   {first_key}.dof形状: {first_motion['dof'].shape}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return None

def main():
    # 配置区域
    input_folder = "/home/user/pbhc-main- cqh723/pbhc-main/real-data-motion-process/output/oriented"  # 🔧 Step3输出的朝向调整后PKL文件
    output_file = None  # None=自动生成文件名，或指定如"merged_motions.pkl"
    file_pattern = "*.pkl"  # 🔧 可以修改匹配模式，如"*motion0.pkl"
    
    print("🎯 批量PKL文件合并工具")
    print("=" * 60)
    
    # 执行合并
    result_file = merge_pkl_folder(input_folder, output_file, file_pattern)
    
    if result_file:
        print(f"\n🎉 成功! 合并文件已保存:")
        print(f"   {os.path.abspath(result_file)}")
    else:
        print(f"\n❌ 合并失败")

if __name__ == "__main__":
    main() 