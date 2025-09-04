#!/usr/bin/env python3
"""
批量机器人朝向调整工具
批量处理目录中的所有PKL文件，调整机器人朝向
"""
import joblib
import numpy as np
import os
import glob
from datetime import datetime
from scipy.spatial.transform import Rotation as R

def rotate_trajectory_complete(positions, velocities, rotations, angular_velocities, smpl_joints, pose_aa, rotation_angle):
    """完整旋转轨迹数据：位置、速度、姿态、角速度、关节位置、pose_aa
    
    Args:
        positions: (N, 3) 位置数据 [x, y, z]
        velocities: (N, 3) 线性速度数据 [vx, vy, vz]
        rotations: (N, 4) 姿态四元数数据 [x, y, z, w]
        angular_velocities: (N, 3) 角速度数据 [wx, wy, wz]
        smpl_joints: (N, J, 3) SMPL关节位置数据 [x, y, z]
        pose_aa: (N, 27, 3) 姿态角轴数据，其中[:, 0, :]是根部姿态
        rotation_angle: 旋转角度 (度数)
    """
    print(f"   🔄 旋转轨迹数据 {rotation_angle}度...")
    
    # 转换为弧度
    angle_rad = np.radians(rotation_angle)
    
    # 创建2D旋转矩阵 (只旋转X-Y平面，Z轴不变)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotation_matrix_3d = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    
    # 1. 旋转位置数据
    rotated_positions = np.dot(positions, rotation_matrix_3d.T)
    
    # 2. 旋转线性速度数据
    rotated_velocities = np.dot(velocities, rotation_matrix_3d.T)
    
    # 3. 旋转角速度数据
    rotated_angular_velocities = np.dot(angular_velocities, rotation_matrix_3d.T)
    
    # 4. 旋转姿态数据（四元数）
    # 创建Z轴旋转四元数
    z_rotation = R.from_euler('z', rotation_angle, degrees=True)
    z_quat = z_rotation.as_quat()  # [x, y, z, w]
    
    # 将原始四元数转换为Rotation对象
    original_rotations = R.from_quat(rotations)  # 输入格式 [x, y, z, w]
    
    # 组合旋转：先应用原始旋转，再应用Z轴旋转
    combined_rotations = z_rotation * original_rotations
    
    # 转换回四元数格式
    rotated_rotations = combined_rotations.as_quat()  # 输出格式 [x, y, z, w]
    
    # 5. 旋转SMPL关节位置数据
    if smpl_joints is not None and smpl_joints.size > 0:
        N, J, _ = smpl_joints.shape
        # 重塑为 (N*J, 3) 进行批量旋转
        joints_reshaped = smpl_joints.reshape(-1, 3)
        # 旋转所有关节位置
        rotated_joints_reshaped = np.dot(joints_reshaped, rotation_matrix_3d.T)
        # 重塑回原始形状
        rotated_smpl_joints = rotated_joints_reshaped.reshape(N, J, 3)
    else:
        rotated_smpl_joints = smpl_joints
    
    # 6. 旋转pose_aa中的根部姿态
    if pose_aa is not None and pose_aa.size > 0:
        rotated_pose_aa = pose_aa.copy()
        # 提取根部姿态 (N, 3) - 角轴表示
        root_pose_aa = pose_aa[:, 0, :]  # shape: (N, 3)
        
        # 将角轴转换为旋转矩阵，应用Z轴旋转，再转回角轴
        for i in range(len(root_pose_aa)):
            if np.linalg.norm(root_pose_aa[i]) > 1e-6:  # 避免零向量
                # 从角轴创建旋转对象
                original_root_rot = R.from_rotvec(root_pose_aa[i])
                # 组合旋转：先应用原始旋转，再应用Z轴旋转
                combined_root_rot = z_rotation * original_root_rot
                # 转换回角轴
                rotated_pose_aa[i, 0, :] = combined_root_rot.as_rotvec()
            else:
                # 如果原始根部姿态是零，直接应用Z轴旋转
                rotated_pose_aa[i, 0, :] = z_rotation.as_rotvec()
    else:
        rotated_pose_aa = pose_aa
    
    return rotated_positions, rotated_velocities, rotated_rotations, rotated_angular_velocities, rotated_smpl_joints, rotated_pose_aa

def load_pkl_data(pkl_file):
    """加载PKL文件数据"""
    print(f"   📂 加载PKL文件: {os.path.basename(pkl_file)}")
    
    try:
        # 加载PKL文件
        data = joblib.load(pkl_file)
        
        # 查找轨迹数据键
        trajectory_key = None
        if isinstance(data, dict):
            # 查找包含轨迹数据的键
            for key in data.keys():
                if isinstance(data[key], dict) and 'root_trans_offset' in data[key]:
                    trajectory_key = key
                    print(f"      使用轨迹键: '{trajectory_key}'")
                    break
        
        if trajectory_key is None:
            raise ValueError("未找到有效的轨迹数据")
        
        pkl_data = data[trajectory_key]
        original_data = data  # 保存完整的原始数据
        
        print(f"   ✅ PKL文件加载成功")
        return original_data, pkl_data, trajectory_key
        
    except Exception as e:
        print(f"   ❌ 加载PKL文件失败: {e}")
        return None, None, None

def save_adjusted_pkl(original_data, adjusted_pkl_data, trajectory_key, pkl_file, rotation_angle, output_dir):
    """保存调整后的PKL文件"""
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(pkl_file))[0]
    output_filename = f"{timestamp}_complete_oriented_{rotation_angle}deg_{base_name}.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    # 更新原始数据中的轨迹
    updated_data = original_data.copy()
    updated_data[trajectory_key] = adjusted_pkl_data
    
    # 保存调整后的数据
    joblib.dump(updated_data, output_path)
    
    print(f"   💾 已保存: {output_filename}")
    return output_path

def process_single_file(pkl_file, rotation_angle, output_dir):
    """处理单个PKL文件"""
    print(f"\n📁 处理文件: {os.path.basename(pkl_file)}")
    
    # 1. 加载PKL数据
    original_data, pkl_data, trajectory_key = load_pkl_data(pkl_file)
    if pkl_data is None:
        print(f"   ❌ 跳过文件: {os.path.basename(pkl_file)}")
        return None
    
    try:
        # 2. 提取所有需要旋转的数据
        original_positions = pkl_data['root_trans_offset'].copy()
        original_velocities = pkl_data['root_lin_vel'].copy() if 'root_lin_vel' in pkl_data else np.zeros_like(original_positions)
        original_rotations = pkl_data['root_rot'].copy()
        original_angular_velocities = pkl_data['root_ang_vel'].copy() if 'root_ang_vel' in pkl_data else np.zeros_like(original_positions)
        original_smpl_joints = pkl_data['smpl_joints'].copy() if 'smpl_joints' in pkl_data else None
        original_pose_aa = pkl_data['pose_aa'].copy() if 'pose_aa' in pkl_data else None
        
        print(f"   📊 数据形状: 位置{original_positions.shape}, 姿态{original_rotations.shape}")
        
        # 3. 完整旋转所有数据
        rotated_positions, rotated_velocities, rotated_rotations, rotated_angular_velocities, rotated_smpl_joints, rotated_pose_aa = rotate_trajectory_complete(
            original_positions, original_velocities, original_rotations, original_angular_velocities, original_smpl_joints, original_pose_aa, rotation_angle)
        
        # 4. 更新PKL数据
        adjusted_pkl_data = pkl_data.copy()
        adjusted_pkl_data['root_trans_offset'] = rotated_positions.astype(np.float32)
        adjusted_pkl_data['root_lin_vel'] = rotated_velocities.astype(np.float32)
        adjusted_pkl_data['root_rot'] = rotated_rotations.astype(np.float32)
        if 'root_ang_vel' in adjusted_pkl_data:
            adjusted_pkl_data['root_ang_vel'] = rotated_angular_velocities.astype(np.float32)
        if 'smpl_joints' in adjusted_pkl_data and rotated_smpl_joints is not None:
            adjusted_pkl_data['smpl_joints'] = rotated_smpl_joints.astype(np.float32)
        if 'pose_aa' in adjusted_pkl_data and rotated_pose_aa is not None:
            adjusted_pkl_data['pose_aa'] = rotated_pose_aa.astype(np.float32)
        
        # 5. 保存调整后的PKL
        output_file = save_adjusted_pkl(original_data, adjusted_pkl_data, trajectory_key, pkl_file, rotation_angle, output_dir)
        
        print(f"   ✅ 处理完成")
        return output_file
        
    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return None

def main():
    """主函数"""
    print("🎯 批量机器人朝向调整工具")
    print("=" * 60)
    
    # 🎯 配置参数
    input_dir = "/home/user/pbhc-main- cqh723/pbhc-main/real-data-motion-process/output"  # Step2输出目录
    output_dir = "/home/user/pbhc-main- cqh723/pbhc-main/real-data-motion-process/output/oriented"  # 朝向调整后输出
    rotation_angle = -90  # 旋转角度：从右手对着您 -> 面对您
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔄 旋转角度: {rotation_angle}度 (右手对着您 -> 面对您)")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有PKL文件
    pkl_pattern = os.path.join(input_dir, "*.pkl")
    pkl_files = sorted(glob.glob(pkl_pattern))
    
    if not pkl_files:
        print(f"❌ 在目录 {input_dir} 中未找到PKL文件")
        return
    
    print(f"📋 找到 {len(pkl_files)} 个PKL文件:")
    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"   {i:2d}. {os.path.basename(pkl_file)}")
    print()
    
    # 批量处理
    successful_files = []
    failed_files = []
    
    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"🔄 [{i}/{len(pkl_files)}] ", end="")
        
        output_file = process_single_file(pkl_file, rotation_angle, output_dir)
        
        if output_file:
            successful_files.append(output_file)
        else:
            failed_files.append(pkl_file)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 批量处理总结:")
    print(f"   ✅ 成功处理: {len(successful_files)} 个文件")
    print(f"   ❌ 处理失败: {len(failed_files)} 个文件")
    
    if successful_files:
        print(f"\n📁 输出文件保存在: {output_dir}")
        print("✅ 成功处理的文件:")
        for file in successful_files:
            print(f"   - {os.path.basename(file)}")
    
    if failed_files:
        print("\n❌ 处理失败的文件:")
        for file in failed_files:
            print(f"   - {os.path.basename(file)}")
    
    print(f"\n🎯 所有文件朝向已从'右手对着您'调整为'面对您'")
    print("🔧 调整内容: ✅位置 ✅速度 ✅姿态 ✅角速度 ✅pose_aa")

if __name__ == "__main__":
    main() 