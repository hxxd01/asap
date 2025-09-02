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

    # 测试相关配置
    cfg.move_to_default_time = config.get("move_to_default_time", 2.0)  # 移动到默认位置的时间
    cfg.hold_default_time = config.get("hold_default_time", 3.0)  # 🌟 保持默认位置的时间
    cfg.stand_test_time = config.get("stand_test_time", 8.0)  # 站立测试时间
    cfg.stand_kp_scale = config.get("stand_kp_scale", 1.0)
    cfg.stand_kd_scale = config.get("stand_kd_scale", 1.0)

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
    mujoco_data['base_height'] = q[2]
    mujoco_data['base_pos'] = q[:3]
    mujoco_data['base_quat'] = quat

    return mujoco_data


def pd_control(target_pos, dof_pos, target_vel, dof_vel, kps, kds):
    torque_out = (target_pos - dof_pos) * kps + (target_vel - dof_vel) * kds
    return torque_out


def check_stability(mujoco_data, stability_threshold=0.85):
    """检查机器人是否稳定"""
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    gravity_stable = abs(mujoco_gvec[0]) < stability_threshold and abs(mujoco_gvec[1]) < stability_threshold

    base_height = mujoco_data["base_height"]
    height_stable = base_height > 0.5

    dof_vel = mujoco_data["mujoco_dof_vel"]
    vel_stable = np.max(np.abs(dof_vel)) < 10.0

    return gravity_stable and height_stable and vel_stable


def interpolate_to_target(current_pos, target_pos, alpha):
    """线性插值到目标位置"""
    return current_pos * (1 - alpha) + target_pos * alpha


def run_realistic_move_and_stand_test(cfg):
    """
    运行现实的移动和站立测试，包含关键的"保持默认位置"阶段
    """
    print("🚀 开始现实的移动和站立测试...")
    print(f"🚶 阶段1: 移动到默认位置时间: {cfg.move_to_default_time}s")
    print(f"🏠 阶段2: 保持默认位置时间: {cfg.hold_default_time}s (模拟default_pos_state)")
    print(f"🧪 阶段3: 站立测试时间: {cfg.stand_test_time}s")

    # 初始化模型
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt

    # 设置初始位置
    initial_pos = np.zeros(cfg.num_actions)
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
        'dof_pos': [],
        'dof_vel': [],
        'target_pos': [],
        'is_stable': [],
        'phase': [],  # 'moving', 'holding', 'standing'
        'position_error': [],
        'velocity_magnitude': []
    }

    # 计算步数
    move_steps = int(cfg.move_to_default_time / cfg.simulation_dt)
    hold_steps = int(cfg.hold_default_time / cfg.simulation_dt)  # 🌟 新增阶段
    stand_steps = int(cfg.stand_test_time / cfg.simulation_dt)
    total_steps = move_steps + hold_steps + stand_steps

    print(f"🚶 阶段1步数: {move_steps}")
    print(f"🏠 阶段2步数: {hold_steps}")
    print(f"🧪 阶段3步数: {stand_steps}")
    print(f"📊 总步数: {total_steps}")

    # 记录初始关节位置
    initial_joint_pos = data.qpos[-cfg.num_actions:].copy()
    target_joint_pos = np.array(cfg.default_dof_pos)

    # 计算站立测试时的PD参数
    stand_kps = np.array(cfg.kps) * cfg.stand_kp_scale
    stand_kds = np.array(cfg.kds) * cfg.stand_kd_scale

    print(f"🔧 站立测试PD参数缩放: kp={cfg.stand_kp_scale}, kd={cfg.stand_kd_scale}")

    # 主循环
    for step in range(total_steps):
        mujoco_data = get_mujoco_data(data)
        current_time = step * cfg.simulation_dt

        # 判断当前阶段
        if step < move_steps:
            # 🚶 阶段1: 移动到默认位置
            phase = 'moving'
            alpha = step / move_steps
            current_target_pos = interpolate_to_target(initial_joint_pos, target_joint_pos, alpha)

            tau = pd_control(current_target_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros(cfg.num_actions), mujoco_data["mujoco_dof_vel"],
                             np.array(cfg.kps), np.array(cfg.kds))

        elif step < move_steps + hold_steps:
            # 🏠 阶段2: 保持默认位置 (模拟实机的default_pos_state)
            phase = 'holding'
            current_target_pos = target_joint_pos.copy()

            # 🌟 这里模拟实机的default_pos_state函数
            # 持续使用PD控制保持默认位置，给机器人时间稳定
            tau = pd_control(current_target_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros(cfg.num_actions), mujoco_data["mujoco_dof_vel"],
                             np.array(cfg.kps), np.array(cfg.kds))

        else:
            # 🧪 阶段3: 站立测试 - 禁用策略，只用PD控制
            phase = 'standing'
            current_target_pos = target_joint_pos.copy()

            # 使用可能不同的PD参数进行站立测试
            tau = pd_control(current_target_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros(cfg.num_actions), mujoco_data["mujoco_dof_vel"],
                             stand_kps, stand_kds)

        # 限制扭矩
        tau_limit = np.array(cfg.tau_limit)
        tau = np.clip(tau, -tau_limit, tau_limit)

        # 执行物理仿真
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        # 记录数据
        if step % 5 == 0:
            is_stable = check_stability(mujoco_data)
            position_error = np.linalg.norm(mujoco_data['mujoco_dof_pos'] - current_target_pos)
            velocity_magnitude = np.linalg.norm(mujoco_data['mujoco_dof_vel'])

            test_data['time'].append(current_time)
            test_data['base_height'].append(mujoco_data['base_height'])
            test_data['gravity_x'].append(mujoco_data['mujoco_gvec'][0])
            test_data['gravity_y'].append(mujoco_data['mujoco_gvec'][1])
            test_data['gravity_z'].append(mujoco_data['mujoco_gvec'][2])
            test_data['dof_pos'].append(mujoco_data['mujoco_dof_pos'].copy())
            test_data['dof_vel'].append(mujoco_data['mujoco_dof_vel'].copy())
            test_data['target_pos'].append(current_target_pos.copy())
            test_data['is_stable'].append(is_stable)
            test_data['phase'].append(phase)
            test_data['position_error'].append(position_error)
            test_data['velocity_magnitude'].append(velocity_magnitude)

        # 打印进度
        if step % 250 == 0:
            stability_status = "✅ 稳定" if check_stability(mujoco_data) else "❌ 不稳定"
            pos_error = np.linalg.norm(mujoco_data['mujoco_dof_pos'] - current_target_pos)
            phase_emoji = {"moving": "🚶", "holding": "🏠", "standing": "🧪"}
            print(f"⏱️  时间: {current_time:.2f}s, 阶段: {phase_emoji[phase]} {phase}, "
                  f"状态: {stability_status}, 位置误差: {pos_error:.4f}")

        # 可视化
        if cfg.render:
            viewer.render()

    if cfg.render:
        viewer.close()

    # 分析结果
    analyze_realistic_results(test_data, cfg)

    return test_data


def analyze_realistic_results(test_data, cfg):
    """分析现实的测试结果"""
    print("\n📊 现实的移动和站立测试结果分析:")

    # 转换为numpy数组
    times = np.array(test_data['time'])
    phases = np.array(test_data['phase'])
    is_stable = np.array(test_data['is_stable'])
    base_heights = np.array(test_data['base_height'])
    gravity_x = np.array(test_data['gravity_x'])
    gravity_y = np.array(test_data['gravity_y'])
    position_errors = np.array(test_data['position_error'])
    velocity_magnitudes = np.array(test_data['velocity_magnitude'])

    # 分阶段分析
    moving_mask = phases == 'moving'
    holding_mask = phases == 'holding'
    standing_mask = phases == 'standing'

    # 各阶段分析
    if np.any(moving_mask):
        moving_stable_rate = np.mean(is_stable[moving_mask]) * 100
        print(f"🚶 移动阶段稳定率: {moving_stable_rate:.1f}%")

    if np.any(holding_mask):
        holding_stable_rate = np.mean(is_stable[holding_mask]) * 100
        holding_final_error = position_errors[holding_mask][-1] if np.any(holding_mask) else 0
        print(f"🏠 保持阶段稳定率: {holding_stable_rate:.1f}%")
        print(f"🏠 保持阶段结束时位置误差: {holding_final_error:.4f}")

    if np.any(standing_mask):
        standing_stable_rate = np.mean(is_stable[standing_mask]) * 100
        avg_standing_error = np.mean(position_errors[standing_mask])
        max_standing_error = np.max(position_errors[standing_mask])
        avg_velocity = np.mean(velocity_magnitudes[standing_mask])
        max_velocity = np.max(velocity_magnitudes[standing_mask])

        print(f"🧪 站立阶段稳定率: {standing_stable_rate:.1f}%")
        print(f"🧪 站立阶段平均位置误差: {avg_standing_error:.4f}")
        print(f"🧪 站立阶段最大位置误差: {max_standing_error:.4f}")
        print(f"🧪 站立阶段平均关节速度: {avg_velocity:.4f} rad/s")
        print(f"🧪 站立阶段最大关节速度: {max_velocity:.4f} rad/s")

    # 高度分析
    initial_height = base_heights[0]
    final_height = base_heights[-1]
    height_change = final_height - initial_height
    print(f"📊 初始高度: {initial_height:.3f}m")
    print(f"📊 最终高度: {final_height:.3f}m")
    print(f"📊 高度变化: {height_change:+.3f}m")

    # 🌟 关键比较：保持阶段 vs 站立测试阶段
    if np.any(holding_mask) and np.any(standing_mask):
        stability_improvement = standing_stable_rate - holding_stable_rate
        print(f"\n🔍 关键对比分析:")
        print(f"保持阶段稳定率: {holding_stable_rate:.1f}%")
        print(f"站立测试稳定率: {standing_stable_rate:.1f}%")
        print(f"稳定性变化: {stability_improvement:+.1f}%")

        if stability_improvement > -5:
            print("✅ 机器人在保持阶段已经稳定，能够很好地过渡到站立测试")
        else:
            print("⚠️  机器人在保持阶段稳定，但站立测试阶段稳定性下降")
            print("💡 建议：调整站立测试时的PD参数")

    # 绘制结果图表
    plot_realistic_results(test_data, cfg)


def plot_realistic_results(test_data, cfg):
    """绘制现实的测试结果图表"""
    times = np.array(test_data['time'])
    phases = np.array(test_data['phase'])
    is_stable = np.array(test_data['is_stable'])
    base_heights = np.array(test_data['base_height'])
    gravity_x = np.array(test_data['gravity_x'])
    gravity_y = np.array(test_data['gravity_y'])
    position_errors = np.array(test_data['position_error'])
    velocity_magnitudes = np.array(test_data['velocity_magnitude'])

    # 找到阶段切换点
    move_end_time = cfg.move_to_default_time
    hold_end_time = cfg.move_to_default_time + cfg.hold_default_time

    plt.figure(figsize=(20, 12))

    # 子图1: 基座高度
    plt.subplot(3, 3, 1)
    plt.plot(times, base_heights, 'b-', linewidth=2)
    plt.axvline(x=move_end_time, color='orange', linestyle='--', alpha=0.7, label='开始保持阶段')
    plt.axvline(x=hold_end_time, color='r', linestyle='--', alpha=0.7, label='开始站立测试')
    plt.xlabel('时间 (s)')
    plt.ylabel('基座高度 (m)')
    plt.title('基座高度变化')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2: 重力投影
    plt.subplot(3, 3, 2)
    plt.plot(times, gravity_x, 'r-', label='X轴', linewidth=2)
    plt.plot(times, gravity_y, 'g-', label='Y轴', linewidth=2)
    plt.axhline(y=0.85, color='k', linestyle=':', alpha=0.5, label='稳定性阈值')
    plt.axhline(y=-0.85, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=move_end_time, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=hold_end_time, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 (s)')
    plt.ylabel('重力投影')
    plt.title('重力投影 (倾斜度)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 位置误差
    plt.subplot(3, 3, 3)
    plt.plot(times, position_errors, 'purple', linewidth=2)
    plt.axvline(x=move_end_time, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=hold_end_time, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 (s)')
    plt.ylabel('位置误差')
    plt.title('关节位置误差')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 子图4: 关节速度大小
    plt.subplot(3, 3, 4)
    plt.plot(times, velocity_magnitudes, 'orange', linewidth=2)
    plt.axvline(x=move_end_time, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=hold_end_time, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 (s)')
    plt.ylabel('关节速度大小 (rad/s)')
    plt.title('关节速度大小')
    plt.grid(True, alpha=0.3)

    # 子图5: 稳定性状态
    plt.subplot(3, 3, 5)
    stable_colors = ['red' if not stable else 'green' for stable in is_stable]
    plt.scatter(times, is_stable, c=stable_colors, alpha=0.6, s=20)
    plt.axvline(x=move_end_time, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=hold_end_time, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 (s)')
    plt.ylabel('稳定状态')
    plt.title('稳定性状态')
    plt.yticks([0, 1], ['不稳定', '稳定'])
    plt.grid(True, alpha=0.3)

    # 子图6: 阶段标识
    plt.subplot(3, 3, 6)
    moving_mask = phases == 'moving'
    holding_mask = phases == 'holding'
    standing_mask = phases == 'standing'

    if np.any(moving_mask):
        moving_times = times[moving_mask]
        plt.fill_between([moving_times[0], moving_times[-1]], [0, 0], [1, 1],
                         alpha=0.3, color='blue', label='移动阶段')

    if np.any(holding_mask):
        holding_times = times[holding_mask]
        plt.fill_between([holding_times[0], holding_times[-1]], [0, 0], [1, 1],
                         alpha=0.3, color='orange', label='保持阶段')

    if np.any(standing_mask):
        standing_times = times[standing_mask]
        plt.fill_between([standing_times[0], standing_times[-1]], [0, 0], [1, 1],
                         alpha=0.3, color='red', label='站立测试')

    plt.axvline(x=move_end_time, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=hold_end_time, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 (s)')
    plt.ylabel('阶段')
    plt.title('测试阶段')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图7-9: 各阶段稳定率分析
    phase_names = ['moving', 'holding', 'standing']
    phase_colors = ['blue', 'orange', 'red']
    phase_emojis = ['🚶', '🏠', '🧪']

    for i, (phase_name, color, emoji) in enumerate(zip(phase_names, phase_colors, phase_emojis)):
        plt.subplot(3, 3, 7 + i)
        mask = phases == phase_name
        if np.any(mask):
            phase_times = times[mask]
            phase_stability = is_stable[mask]

            # 计算滑动窗口稳定率
            window_size = min(20, len(phase_stability) // 2)
            if len(phase_stability) >= window_size:
                stability_rate = []
                window_times = []
                for j in range(window_size, len(phase_stability)):
                    rate = np.mean(phase_stability[j - window_size:j]) * 100
                    stability_rate.append(rate)
                    window_times.append(phase_times[j])

                plt.plot(window_times, stability_rate, color=color, linewidth=2)
                plt.axhline(y=90, color='g', linestyle=':', alpha=0.5, label='优秀阈值')
                plt.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='良好阈值')
                plt.xlabel('时间 (s)')
                plt.ylabel('稳定率 (%)')
                plt.title(f'{emoji} {phase_name}阶段稳定率')
                plt.legend()
                plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"realistic_move_and_stand_test_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 结果图表已保存: {plot_filename}")

    plt.show()


if __name__ == '__main__':
    current_directory = os.getcwd()
    config_file = current_directory + "/g1_config/mujoco_config.yaml"

    # 读取配置
    cfg = read_conf(config_file)

    # 设置测试参数
    cfg.move_to_default_time = 2.0  # 移动到默认位置
    cfg.hold_default_time = 3.0  # 🌟 保持默认位置 (模拟实机的default_pos_state)
    cfg.stand_test_time = 5.0  # 站立测试
    cfg.stand_kp_scale = 1.0  # 可以调整这个来测试不同的PD参数
    cfg.stand_kd_scale = 1.0

    # 运行现实的测试
    test_data = run_realistic_move_and_stand_test(cfg)

    print("🏁 现实的移动和站立测试完成!")
    print("\n💡 如果保持阶段稳定但站立测试阶段不稳定，建议:")
    print("   1. 增加 stand_kd_scale (提高阻尼)")
    print("   2. 调整 stand_kp_scale (调整刚度)")
    print("   3. 检查扭矩限制是否合适")