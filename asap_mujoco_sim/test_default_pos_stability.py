import mujoco, mujoco_viewer
import numpy as np
import onnxruntime
import yaml
import os
import joblib
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import torch
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


def read_conf(config_file):
    cfg = SimpleNamespace()
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    cfg.policy_path = config["policy_path"]
    cfg.cycle_time = config["cycle_time"]
    cfg.xml_path = config["xml_path"]
    cfg.num_single_obs = config["num_single_obs"]
    cfg.simulation_dt = config["simulation_dt"]
    cfg.simulation_duration = config["simulation_duration"]
    cfg.control_decimation = config["control_decimation"]
    cfg.frame_stack = config["frame_stack"]
    cfg.episode_steps = config["episode_steps"]
    cfg.total_steps = config["total_steps"]
    cfg.default_dof_pos = config["default_dof_pos"]
    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]
    cfg.obs_scale_gvec = config["obs_scale_gvec"]
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]
    cfg.obs_scale_hist = config["obs_scale_hist"]
    cfg.num_actions = config["num_actions"]
    cfg.clip_observations = config["clip_observations"]
    cfg.clip_actions = config["clip_actions"]
    cfg.action_scale = config["action_scale"]
    cfg.kps = config["kps"]
    cfg.kds = config["kds"]
    cfg.tau_limit = config["tau_limit"]
    cfg.render = config["render"]
    cfg.use_noise = config["use_noise"]

    # 移动和站立测试相关配置
    cfg.move_to_default_time = config.get("move_to_default_time", 2.0)  # 移动到默认位置的时间(秒)
    cfg.stand_test_time = config.get("stand_test_time", 8.0)  # 站立测试时间(秒)
    cfg.stand_kp_scale = config.get("stand_kp_scale", 1.0)  # 站立测试时的kp缩放
    cfg.stand_kd_scale = config.get("stand_kd_scale", 1.0)  # 站立测试时的kd缩放

    # 平衡控制器相关配置
    cfg.balance_enable = config.get("balance_enable", True)  # 是否启用平衡控制器
    cfg.balance_tilt_threshold = config.get("balance_tilt_threshold", 0.1)  # 倾斜阈值(弧度)
    cfg.balance_correction_strength = config.get("balance_correction_strength", 0.15)  # 平衡修正强度

    return cfg


def get_mujoco_data(data):
    mujoco_data = {}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = np.array([q[4], q[5], q[6], q[3]])
    r = R.from_quat(quat)
    base_angvel = dq[3:6]
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)

    mujoco_data['mujoco_dof_pos'] = q[7:]
    mujoco_data['mujoco_dof_vel'] = dq[6:]
    mujoco_data['mujoco_base_angvel'] = base_angvel
    mujoco_data['mujoco_gvec'] = gvec
    mujoco_data['base_height'] = q[2]  # z坐标作为高度
    mujoco_data['base_pos'] = q[:3]  # 基座位置
    mujoco_data['base_quat'] = quat  # 基座四元数

    # 计算roll和pitch角度（用于平衡控制）
    # 从重力投影计算倾斜角度
    mujoco_data['roll'] = np.arctan2(gvec[1], gvec[2])  # 左右倾斜
    mujoco_data['pitch'] = np.arctan2(-gvec[0], np.sqrt(gvec[1] ** 2 + gvec[2] ** 2))  # 前后倾斜

    return mujoco_data


def pd_control(target_pos, dof_pos, target_vel, dof_vel, kps, kds):
    torque_out = (target_pos - dof_pos) * kps + (target_vel - dof_vel) * kds
    return torque_out


def simple_balance_controller(imu_data, target_angles, cfg):
    """
    简单的平衡控制器 - 使用IMU数据修正目标角度

    Args:
        imu_data: 包含roll和pitch角度的IMU数据
        target_angles: 原始目标角度 (23个关节)
        cfg: 配置参数

    Returns:
        修正后的目标角度
    """
    if not cfg.balance_enable:
        return target_angles

    # 复制目标角度，避免修改原始数据
    corrected_angles = target_angles.copy()

    # G1关节索引 (根据你的配置调整)
    # 假设关节顺序：左腿6个，右腿6个，腰部3个，左臂4个，右臂4个
    left_hip_roll_idx = 2  # 左髋Roll关节
    right_hip_roll_idx = 8  # 右髋Roll关节
    left_hip_pitch_idx = 1  # 左髋Pitch关节
    right_hip_pitch_idx = 7  # 右髋Pitch关节

    # 获取倾斜角度
    roll = imu_data['roll']
    pitch = imu_data['pitch']

    # 左右倾斜修正 (Roll)
    if abs(roll) > cfg.balance_tilt_threshold:
        correction = -cfg.balance_correction_strength * roll
        corrected_angles[left_hip_roll_idx] += correction
        corrected_angles[right_hip_roll_idx] += correction

    # 前后倾斜修正 (Pitch)
    if abs(pitch) > cfg.balance_tilt_threshold:
        correction = -cfg.balance_correction_strength * pitch
        corrected_angles[left_hip_pitch_idx] += correction
        corrected_angles[right_hip_pitch_idx] += correction

    return corrected_angles


def check_stability(mujoco_data, stability_threshold=0.85):
    """
    检查机器人是否稳定
    """
    # 重力投影检查
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    gravity_stable = abs(mujoco_gvec[0]) < stability_threshold and abs(mujoco_gvec[1]) < stability_threshold

    # 基座高度检查
    base_height = mujoco_data["base_height"]
    height_stable = base_height > 0.5  # 基座高度应该大于0.5米

    # 关节速度检查
    dof_vel = mujoco_data["mujoco_dof_vel"]
    vel_stable = np.max(np.abs(dof_vel)) < 10.0  # 关节速度不应该过大

    return gravity_stable and height_stable and vel_stable


def interpolate_to_target(current_pos, target_pos, alpha):
    """
    线性插值到目标位置
    alpha: 0 = current_pos, 1 = target_pos
    """
    return current_pos * (1 - alpha) + target_pos * alpha


def run_move_and_stand_test(cfg):
    """
    运行移动到默认位置并站立测试
    """
    print("🚀 开始移动到默认位置并站立测试...")
    print(f"🚶 移动到默认位置时间: {cfg.move_to_default_time}s")
    print(f"🏠 站立测试时间: {cfg.stand_test_time}s")
    print(f"⚖️  平衡控制器: {'启用' if cfg.balance_enable else '禁用'}")
    if cfg.balance_enable:
        print(f"🎯 倾斜阈值: {cfg.balance_tilt_threshold:.3f} rad ({cfg.balance_tilt_threshold * 57.3:.1f}°)")
        print(f"🔧 修正强度: {cfg.balance_correction_strength:.3f}")

    # 初始化模型
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt

    # 设置初始位置（可以是零位或其他位置）
    initial_pos = np.zeros(cfg.num_actions)  # 从零位开始
    data.qpos[-cfg.num_actions:] = initial_pos
    mujoco.mj_step(model, data)

    print(f"📍 初始位置: 零位")
    print(f"🎯 目标位置: 默认位置 {cfg.default_dof_pos}")

    # 可视化设置
    if cfg.render:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

    # 数据记录
    test_data = {
        'time': [],
        'base_height': [],
        'gravity_x': [],
        'gravity_y': [],
        'gravity_z': [],
        'roll': [],  # 新增：记录roll角度
        'pitch': [],  # 新增：记录pitch角度
        'dof_pos': [],
        'dof_vel': [],
        'target_pos': [],
        'corrected_target_pos': [],  # 新增：记录平衡修正后的目标位置
        'is_stable': [],
        'phase': [],  # 'moving' 或 'standing'
        'position_error': [],  # 位置误差
        'velocity_magnitude': [],  # 关节速度大小
        'balance_correction': []  # 新增：记录平衡修正量
    }

    # 计算步数
    move_steps = int(cfg.move_to_default_time / cfg.simulation_dt)
    stand_steps = int(cfg.stand_test_time / cfg.simulation_dt)
    total_steps = move_steps + stand_steps

    print(f"🚶 移动阶段步数: {move_steps}")
    print(f"🏠 站立阶段步数: {stand_steps}")
    print(f"📊 总步数: {total_steps}")

    # 记录初始关节位置
    initial_joint_pos = data.qpos[-cfg.num_actions:].copy()
    target_joint_pos = np.array(cfg.default_dof_pos)
    stand_kp_scale = cfg.stand_kp_scale
    stand_kd_scale = cfg.stand_kd_scale

    # 计算站立测试时的PD参数
    stand_kps = np.array(cfg.kps) * stand_kp_scale
    stand_kds = np.array(cfg.kds) * stand_kd_scale

    print(f"🔧 站立测试PD参数缩放: kp={stand_kp_scale}, kd={stand_kd_scale}")

    # 主循环
    for step in range(total_steps):
        mujoco_data = get_mujoco_data(data)
        current_time = step * cfg.simulation_dt

        # 判断当前阶段
        if step < move_steps:
            # 移动到默认位置阶段
            phase = 'moving'

            # 计算插值进度 (0 到 1)
            alpha = step / move_steps

            # 插值计算当前目标位置
            current_target_pos = interpolate_to_target(initial_joint_pos, target_joint_pos, alpha)

            # 移动阶段不使用平衡控制器
            corrected_target_pos = current_target_pos.copy()
            balance_correction_magnitude = 0.0

            # 使用PD控制移动到目标位置
            tau = pd_control(corrected_target_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros(cfg.num_actions), mujoco_data["mujoco_dof_vel"],
                             np.array(cfg.kps), np.array(cfg.kds))

        else:
            # 站立测试阶段 - 使用平衡控制器
            phase = 'standing'

            # 目标位置就是默认位置
            current_target_pos = target_joint_pos.copy()

            # 🚀 应用平衡控制器！
            corrected_target_pos = simple_balance_controller(mujoco_data, current_target_pos, cfg)

            # 计算平衡修正量
            balance_correction_magnitude = np.linalg.norm(corrected_target_pos - current_target_pos)

            # 使用修正后的目标位置进行PD控制
            tau = pd_control(corrected_target_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros(cfg.num_actions), mujoco_data["mujoco_dof_vel"],
                             stand_kps, stand_kds)

        # 限制扭矩
        tau_limit = np.array(cfg.tau_limit)
        tau = np.clip(tau, -tau_limit, tau_limit)

        # 执行物理仿真
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        # 记录数据
        if step % 5 == 0:  # 每5步记录一次数据，增加数据密度
            is_stable = check_stability(mujoco_data)

            # 计算位置误差
            position_error = np.linalg.norm(mujoco_data['mujoco_dof_pos'] - current_target_pos)

            # 计算关节速度大小
            velocity_magnitude = np.linalg.norm(mujoco_data['mujoco_dof_vel'])

            test_data['time'].append(current_time)
            test_data['base_height'].append(mujoco_data['base_height'])
            test_data['gravity_x'].append(mujoco_data['mujoco_gvec'][0])
            test_data['gravity_y'].append(mujoco_data['mujoco_gvec'][1])
            test_data['gravity_z'].append(mujoco_data['mujoco_gvec'][2])
            test_data['roll'].append(mujoco_data['roll'])
            test_data['pitch'].append(mujoco_data['pitch'])
            test_data['dof_pos'].append(mujoco_data['mujoco_dof_pos'].copy())
            test_data['dof_vel'].append(mujoco_data['mujoco_dof_vel'].copy())
            test_data['target_pos'].append(current_target_pos.copy())
            test_data['corrected_target_pos'].append(corrected_target_pos.copy())
            test_data['is_stable'].append(is_stable)
            test_data['phase'].append(phase)
            test_data['position_error'].append(position_error)
            test_data['velocity_magnitude'].append(velocity_magnitude)
            test_data['balance_correction'].append(balance_correction_magnitude)

        # 打印进度
        if step % 250 == 0:
            stability_status = "✅ 稳定" if check_stability(mujoco_data) else "❌ 不稳定"
            pos_error = np.linalg.norm(mujoco_data['mujoco_dof_pos'] - current_target_pos)
            tilt_info = f"倾斜: {mujoco_data['roll'] * 57.3:.1f}°/{mujoco_data['pitch'] * 57.3:.1f}°" if phase == 'standing' else ""
            balance_info = f"平衡修正: {balance_correction_magnitude:.4f}" if phase == 'standing' and cfg.balance_enable else ""
            print(f"⏱️  时间: {current_time:.2f}s, 阶段: {phase}, 状态: {stability_status}, 位置误差: {pos_error:.4f}, {tilt_info} {balance_info}")

        # 可视化
        if cfg.render:
            viewer.render()

    if cfg.render:
        viewer.close()

    # 分析结果
    analyze_move_and_stand_results(test_data, cfg)

    return test_data


def analyze_move_and_stand_results(test_data, cfg):
    """
    分析移动和站立测试结果
    """
    print("\n📊 移动和站立测试结果分析:")

    # 转换为numpy数组便于分析
    times = np.array(test_data['time'])
    phases = np.array(test_data['phase'])
    is_stable = np.array(test_data['is_stable'])
    base_heights = np.array(test_data['base_height'])
    gravity_x = np.array(test_data['gravity_x'])
    gravity_y = np.array(test_data['gravity_y'])
    rolls = np.array(test_data['roll'])
    pitches = np.array(test_data['pitch'])
    position_errors = np.array(test_data['position_error'])
    velocity_magnitudes = np.array(test_data['velocity_magnitude'])
    balance_corrections = np.array(test_data['balance_correction'])

    # 分阶段分析
    moving_mask = phases == 'moving'
    standing_mask = phases == 'standing'

    # 移动阶段分析
    if np.any(moving_mask):
        moving_stable_rate = np.mean(is_stable[moving_mask]) * 100
        final_moving_error = position_errors[moving_mask][-1] if np.any(moving_mask) else 0
        print(f"🚶 移动阶段稳定率: {moving_stable_rate:.1f}%")
        print(f"🎯 移动结束时位置误差: {final_moving_error:.4f}")

    # 站立阶段分析
    if np.any(standing_mask):
        standing_stable_rate = np.mean(is_stable[standing_mask]) * 100
        avg_standing_error = np.mean(position_errors[standing_mask])
        max_standing_error = np.max(position_errors[standing_mask])
        avg_velocity = np.mean(velocity_magnitudes[standing_mask])
        max_velocity = np.max(velocity_magnitudes[standing_mask])

        # 倾斜角度分析
        max_roll = np.max(np.abs(rolls[standing_mask])) * 57.3  # 转换为度
        max_pitch = np.max(np.abs(pitches[standing_mask])) * 57.3
        avg_roll = np.mean(np.abs(rolls[standing_mask])) * 57.3
        avg_pitch = np.mean(np.abs(pitches[standing_mask])) * 57.3

        # 平衡修正分析
        if cfg.balance_enable:
            avg_balance_correction = np.mean(balance_corrections[standing_mask])
            max_balance_correction = np.max(balance_corrections[standing_mask])
            active_correction_rate = np.mean(balance_corrections[standing_mask] > 0.001) * 100

        print(f"🏠 站立阶段稳定率: {standing_stable_rate:.1f}%")
        print(f"📏 站立阶段平均位置误差: {avg_standing_error:.4f}")
        print(f"📏 站立阶段最大位置误差: {max_standing_error:.4f}")
        print(f"🏃 站立阶段平均关节速度: {avg_velocity:.4f} rad/s")
        print(f"🏃 站立阶段最大关节速度: {max_velocity:.4f} rad/s")
        print(f"📐 平均倾斜角度: Roll={avg_roll:.2f}°, Pitch={avg_pitch:.2f}°")
        print(f"📐 最大倾斜角度: Roll={max_roll:.2f}°, Pitch={max_pitch:.2f}°")

        if cfg.balance_enable:
            print(f"⚖️  平均平衡修正: {avg_balance_correction:.4f}")
            print(f"⚖️  最大平衡修正: {max_balance_correction:.4f}")
            print(f"⚖️  平衡修正激活率: {active_correction_rate:.1f}%")

    # 高度分析
    initial_height = base_heights[0]
    final_height = base_heights[-1]
    height_change = final_height - initial_height
    print(f"📊 初始高度: {initial_height:.3f}m")
    print(f"📊 最终高度: {final_height:.3f}m")
    print(f"📊 高度变化: {height_change:+.3f}m")

    # 重力投影分析
    if np.any(standing_mask):
        max_gravity_x = np.max(np.abs(gravity_x[standing_mask]))
        max_gravity_y = np.max(np.abs(gravity_y[standing_mask]))
        print(f"🌍 站立阶段最大重力投影: X={max_gravity_x:.3f}, Y={max_gravity_y:.3f}")

    # 整体评估
    if np.any(standing_mask):
        if standing_stable_rate > 95 and avg_standing_error < 0.1:
            print("✅ 整体评估: 机器人能够很好地移动到默认位置并保持稳定")
        elif standing_stable_rate > 80 and avg_standing_error < 0.2:
            print("⚠️  整体评估: 机器人基本能够站稳，但稳定性有待改善")
        else:
            print("❌ 整体评估: 机器人难以在默认位置保持稳定")

    # 绘制结果图表
    plot_move_and_stand_results(test_data, cfg)


def plot_move_and_stand_results(test_data, cfg):
    """
    绘制移动和站立测试结果图表
    """
    times = np.array(test_data['time'])
    phases = np.array(test_data['phase'])
    is_stable = np.array(test_data['is_stable'])
    base_heights = np.array(test_data['base_height'])
    gravity_x = np.array(test_data['gravity_x'])
    gravity_y = np.array(test_data['gravity_y'])
    rolls = np.array(test_data['roll'])
    pitches = np.array(test_data['pitch'])
    position_errors = np.array(test_data['position_error'])
    velocity_magnitudes = np.array(test_data['velocity_magnitude'])
    balance_corrections = np.array(test_data['balance_correction'])

    # 找到阶段切换点
    move_end_time = cfg.move_to_default_time

    plt.figure(figsize=(20, 16))

    # 子图1: 基座高度
    plt.subplot(4, 2, 1)
    plt.plot(times, base_heights, 'b-', linewidth=2)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('基座高度 (m)')
    plt.title('基座高度变化')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2: 重力投影
    plt.subplot(4, 2, 2)
    plt.plot(times, gravity_x, 'r-', label='X轴', linewidth=2)
    plt.plot(times, gravity_y, 'g-', label='Y轴', linewidth=2)
    plt.axhline(y=0.85, color='k', linestyle=':', alpha=0.5, label='稳定性阈值')
    plt.axhline(y=-0.85, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('重力投影')
    plt.title('重力投影 (倾斜度)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 倾斜角度 (新增)
    plt.subplot(4, 2, 3)
    plt.plot(times, rolls * 57.3, 'purple', label='Roll (左右)', linewidth=2)
    plt.plot(times, pitches * 57.3, 'orange', label='Pitch (前后)', linewidth=2)
    plt.axhline(y=cfg.balance_tilt_threshold * 57.3, color='k', linestyle=':', alpha=0.5, label='平衡阈值')
    plt.axhline(y=-cfg.balance_tilt_threshold * 57.3, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('倾斜角度 (度)')
    plt.title('倾斜角度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图4: 平衡修正量 (新增)
    plt.subplot(4, 2, 4)
    plt.plot(times, balance_corrections, 'brown', linewidth=2)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('平衡修正量')
    plt.title('平衡控制器修正量')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图5: 位置误差
    plt.subplot(4, 2, 5)
    plt.plot(times, position_errors, 'purple', linewidth=2)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('位置误差')
    plt.title('关节位置误差')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # 使用对数坐标，方便观察小误差

    # 子图6: 关节速度大小
    plt.subplot(4, 2, 6)
    plt.plot(times, velocity_magnitudes, 'orange', linewidth=2)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('关节速度大小 (rad/s)')
    plt.title('关节速度大小')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图7: 稳定性状态
    plt.subplot(4, 2, 7)
    stable_colors = ['red' if not stable else 'green' for stable in is_stable]
    plt.scatter(times, is_stable, c=stable_colors, alpha=0.6, s=20)
    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('稳定状态')
    plt.title('稳定性状态 (绿色=稳定, 红色=不稳定)')
    plt.yticks([0, 1], ['不稳定', '稳定'])
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图8: 阶段标识和综合评估
    plt.subplot(4, 2, 8)

    # 绘制阶段区域
    moving_mask = phases == 'moving'
    standing_mask = phases == 'standing'

    if np.any(moving_mask):
        moving_times = times[moving_mask]
        moving_stability = is_stable[moving_mask]
        plt.fill_between([moving_times[0], moving_times[-1]], [0, 0], [1, 1],
                         alpha=0.3, color='blue', label='移动阶段')

    if np.any(standing_mask):
        standing_times = times[standing_mask]
        standing_stability = is_stable[standing_mask]
        plt.fill_between([standing_times[0], standing_times[-1]], [0, 0], [1, 1],
                         alpha=0.3, color='green', label='站立阶段')

    # 计算滑动窗口稳定率
    window_size = 20
    if len(is_stable) >= window_size:
        stability_rate = []
        window_times = []
        for i in range(window_size, len(is_stable)):
            rate = np.mean(is_stable[i - window_size:i]) * 100
            stability_rate.append(rate)
            window_times.append(times[i])

        plt.plot(window_times, np.array(stability_rate) / 100, 'k-', linewidth=2, label='滑动稳定率')

    plt.axvline(x=move_end_time, color='r', linestyle='--', label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('稳定率')
    plt.title('测试阶段和稳定率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    balance_status = "with_balance" if cfg.balance_enable else "no_balance"
    plot_filename = f"move_and_stand_test_{balance_status}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 结果图表已保存: {plot_filename}")

    plt.show()


if __name__ == '__main__':
    current_directory = os.getcwd()
    config_file = current_directory + "/g1_config/new_mujoco_config.yaml"

    # 读取配置
    cfg = read_conf(config_file)

    # 运行移动和站立测试
    test_data = run_move_and_stand_test(cfg)

    print("🏁 移动和站立测试完成!") 