import mujoco, mujoco_viewer
import numpy as np
import yaml
import os
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
from datetime import datetime
import matplotlib.pyplot as plt
import time


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


def check_stability(mujoco_data, strict_mode=False):
    """检查机器人是否稳定"""
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    base_height = mujoco_data["base_height"]
    dof_vel = mujoco_data["mujoco_dof_vel"]

    if strict_mode:
        # 严格模式：更严格的稳定性检查
        gravity_stable = abs(mujoco_gvec[0]) < 0.7 and abs(mujoco_gvec[1]) < 0.7
        height_stable = base_height > 0.6
        vel_stable = np.max(np.abs(dof_vel)) < 5.0
    else:
        # 正常模式
        gravity_stable = abs(mujoco_gvec[0]) < 0.85 and abs(mujoco_gvec[1]) < 0.85
        height_stable = base_height > 0.5
        vel_stable = np.max(np.abs(dof_vel)) < 10.0

    return gravity_stable and height_stable and vel_stable


def check_catastrophic_failure(mujoco_data):
    """检查是否发生灾难性失败（需要立即重置）"""
    base_height = mujoco_data["base_height"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    dof_vel = mujoco_data["mujoco_dof_vel"]

    # 灾难性失败的条件
    height_failure = base_height < 0.3  # 基座高度过低
    tilt_failure = abs(mujoco_gvec[0]) > 0.95 or abs(mujoco_gvec[1]) > 0.95  # 严重倾斜
    velocity_failure = np.max(np.abs(dof_vel)) > 20.0  # 关节速度过大

    return height_failure or tilt_failure or velocity_failure


def interpolate_to_target(current_pos, target_pos, alpha):
    """线性插值到目标位置"""
    return current_pos * (1 - alpha) + target_pos * alpha


def test_single_pd_config(cfg, kp_scale, kd_scale, test_duration=8.0, render=False):
    """
    测试单个PD参数配置
    返回测试结果和是否成功
    """
    print(f"🧪 测试 KP:{kp_scale:.1f}, KD:{kd_scale:.1f}")

    # 初始化模型
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.simulation_dt

    # 可视化设置
    viewer = None
    if render:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

    # 设置初始位置
    initial_pos = np.zeros(cfg.num_actions)
    data.qpos[-cfg.num_actions:] = initial_pos
    mujoco.mj_step(model, data)

    # 记录数据
    test_data = {
        'time': [],
        'base_height': [],
        'gravity_x': [],
        'gravity_y': [],
        'is_stable': [],
        'position_error': [],
        'velocity_magnitude': [],
        'phase': []
    }

    # 测试参数
    move_time = 2.0
    hold_time = 3.0
    stand_time = test_duration - move_time - hold_time

    move_steps = int(move_time / cfg.simulation_dt)
    hold_steps = int(hold_time / cfg.simulation_dt)
    stand_steps = int(stand_time / cfg.simulation_dt)
    total_steps = move_steps + hold_steps + stand_steps

    # PD参数
    test_kps = np.array(cfg.kps) * kp_scale
    test_kds = np.array(cfg.kds) * kd_scale

    # 目标位置
    target_joint_pos = np.array(cfg.default_dof_pos)
    initial_joint_pos = data.qpos[-cfg.num_actions:].copy()

    # 测试循环
    success = True
    failure_reason = ""
    stable_count = 0
    total_checks = 0

    for step in range(total_steps):
        mujoco_data = get_mujoco_data(data)
        current_time = step * cfg.simulation_dt

        # 判断当前阶段
        if step < move_steps:
            # 移动阶段
            phase = 'moving'
            alpha = step / move_steps
            current_target_pos = interpolate_to_target(initial_joint_pos, target_joint_pos, alpha)
        elif step < move_steps + hold_steps:
            # 保持阶段
            phase = 'holding'
            current_target_pos = target_joint_pos.copy()
        else:
            # 站立测试阶段
            phase = 'standing'
            current_target_pos = target_joint_pos.copy()

        # PD控制
        tau = pd_control(current_target_pos, mujoco_data["mujoco_dof_pos"],
                         np.zeros(cfg.num_actions), mujoco_data["mujoco_dof_vel"],
                         test_kps, test_kds)

        # 限制扭矩
        tau_limit = np.array(cfg.tau_limit)
        tau = np.clip(tau, -tau_limit, tau_limit)

        # 执行物理仿真
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        # 检查灾难性失败
        if check_catastrophic_failure(mujoco_data):
            success = False
            failure_reason = "灾难性失败"
            break

        # 记录数据
        if step % 10 == 0:
            is_stable = check_stability(mujoco_data)
            position_error = np.linalg.norm(mujoco_data['mujoco_dof_pos'] - current_target_pos)
            velocity_magnitude = np.linalg.norm(mujoco_data['mujoco_dof_vel'])

            test_data['time'].append(current_time)
            test_data['base_height'].append(mujoco_data['base_height'])
            test_data['gravity_x'].append(mujoco_data['mujoco_gvec'][0])
            test_data['gravity_y'].append(mujoco_data['mujoco_gvec'][1])
            test_data['is_stable'].append(is_stable)
            test_data['position_error'].append(position_error)
            test_data['velocity_magnitude'].append(velocity_magnitude)
            test_data['phase'].append(phase)

            if is_stable:
                stable_count += 1
            total_checks += 1

            # 如果保持阶段就不稳定，提前结束
            if phase == 'holding' and step > move_steps + hold_steps // 2:
                if stable_count / total_checks < 0.3:  # 稳定率太低
                    success = False
                    failure_reason = "保持阶段不稳定"
                    break

        # 可视化
        if render and viewer:
            viewer.render()

    if render and viewer:
        viewer.close()

    # 计算结果
    if success and total_checks > 0:
        stability_rate = stable_count / total_checks * 100
        avg_position_error = np.mean(test_data['position_error'])
        avg_velocity = np.mean(test_data['velocity_magnitude'])
        final_height = test_data['base_height'][-1] if test_data['base_height'] else 0

        result = {
            'success': True,
            'kp_scale': kp_scale,
            'kd_scale': kd_scale,
            'stability_rate': stability_rate,
            'avg_position_error': avg_position_error,
            'avg_velocity': avg_velocity,
            'final_height': final_height,
            'test_duration': current_time,
            'failure_reason': ""
        }

        print(f"   ✅ 成功 - 稳定率: {stability_rate:.1f}%, 位置误差: {avg_position_error:.4f}")

    else:
        result = {
            'success': False,
            'kp_scale': kp_scale,
            'kd_scale': kd_scale,
            'stability_rate': 0.0,
            'avg_position_error': 999.0,
            'avg_velocity': 999.0,
            'final_height': 0.0,
            'test_duration': current_time,
            'failure_reason': failure_reason
        }

        print(f"   ❌ 失败 - {failure_reason}")

    return result


def run_auto_pd_optimization(cfg, render_best=True):
    """
    运行自动PD参数优化
    """
    print("🔬 开始自动PD参数优化...")
    print("=" * 60)

    # 测试参数范围
    kp_scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    kd_scales = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

    print(f"📊 将测试 {len(kp_scales)} x {len(kd_scales)} = {len(kp_scales) * len(kd_scales)} 种参数组合")
    print(f"🎯 KP缩放范围: {min(kp_scales)} - {max(kp_scales)}")
    print(f"🎯 KD缩放范围: {min(kd_scales)} - {max(kd_scales)}")

    results = []
    successful_results = []

    start_time = time.time()

    for i, kp_scale in enumerate(kp_scales):
        for j, kd_scale in enumerate(kd_scales):
            progress = (i * len(kd_scales) + j + 1) / (len(kp_scales) * len(kd_scales)) * 100
            print(f"\n📈 进度: {progress:.1f}% ({i * len(kd_scales) + j + 1}/{len(kp_scales) * len(kd_scales)})")

            # 测试当前参数组合
            result = test_single_pd_config(cfg, kp_scale, kd_scale, test_duration=8.0, render=False)
            results.append(result)

            if result['success']:
                successful_results.append(result)

    total_time = time.time() - start_time
    print(f"\n⏱️  总测试时间: {total_time:.1f}秒")
    print(f"✅ 成功的参数组合: {len(successful_results)}/{len(results)}")

    # 分析结果
    if successful_results:
        analyze_optimization_results(successful_results, cfg, render_best)
    else:
        print("❌ 没有找到成功的参数组合！")
        print("💡 建议：")
        print("   1. 增加KD参数范围")
        print("   2. 检查默认关节角度是否合理")
        print("   3. 检查扭矩限制是否过低")

    return results


def analyze_optimization_results(results, cfg, render_best=True):
    """分析优化结果"""
    print("\n📊 优化结果分析:")
    print("=" * 80)

    # 按稳定率排序
    sorted_results = sorted(results, key=lambda x: x['stability_rate'], reverse=True)

    print("🏆 最佳参数组合 (按稳定率排序):")
    print("排名  KP缩放  KD缩放  稳定率   位置误差   速度大小   最终高度")
    print("-" * 70)

    for i, result in enumerate(sorted_results[:10]):  # 显示前10名
        print(f"{i + 1:2d}    {result['kp_scale']:6.1f}  {result['kd_scale']:6.1f}  "
              f"{result['stability_rate']:6.1f}%  {result['avg_position_error']:8.4f}  "
              f"{result['avg_velocity']:8.4f}  {result['final_height']:8.3f}m")

    # 选择最佳参数
    best_result = sorted_results[0]
    print(f"\n🎯 推荐参数:")
    print(f"   KP缩放: {best_result['kp_scale']}")
    print(f"   KD缩放: {best_result['kd_scale']}")
    print(f"   预期稳定率: {best_result['stability_rate']:.1f}%")

    # 生成配置文件建议
    original_kps = np.array(cfg.kps)
    original_kds = np.array(cfg.kds)
    recommended_kps = original_kps * best_result['kp_scale']
    recommended_kds = original_kds * best_result['kd_scale']

    print(f"\n🔧 配置文件修改建议:")
    print("原始KP:", [f"{x:.1f}" for x in original_kps])
    print("推荐KP:", [f"{x:.1f}" for x in recommended_kps])
    print("原始KD:", [f"{x:.2f}" for x in original_kds])
    print("推荐KD:", [f"{x:.2f}" for x in recommended_kds])

    # 可视化结果
    plot_optimization_results(results)

    # 用最佳参数重新测试并可视化
    if render_best:
        print(f"\n🎬 使用最佳参数重新测试并可视化...")
        test_single_pd_config(cfg, best_result['kp_scale'], best_result['kd_scale'],
                              test_duration=10.0, render=True)

    return best_result


def plot_optimization_results(results):
    """绘制优化结果"""
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    if not successful_results:
        print("❌ 没有成功的结果可以绘制")
        return

    # 准备数据
    kp_scales = [r['kp_scale'] for r in successful_results]
    kd_scales = [r['kd_scale'] for r in successful_results]
    stability_rates = [r['stability_rate'] for r in successful_results]
    position_errors = [r['avg_position_error'] for r in successful_results]

    # 准备失败的数据
    failed_kp_scales = [r['kp_scale'] for r in failed_results]
    failed_kd_scales = [r['kd_scale'] for r in failed_results]

    plt.figure(figsize=(15, 10))

    # 子图1: 稳定率散点图
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(kp_scales, kd_scales, c=stability_rates, cmap='RdYlGn',
                          s=100, alpha=0.7, edgecolors='black')
    plt.scatter(failed_kp_scales, failed_kd_scales, c='red', s=50, alpha=0.5,
                marker='x', label='失败')
    plt.colorbar(scatter, label='稳定率 (%)')
    plt.xlabel('KP缩放')
    plt.ylabel('KD缩放')
    plt.title('稳定率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 位置误差散点图
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(kp_scales, kd_scales, c=position_errors, cmap='RdYlGn_r',
                          s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, label='位置误差')
    plt.xlabel('KP缩放')
    plt.ylabel('KD缩放')
    plt.title('位置误差分布')
    plt.grid(True, alpha=0.3)

    # 子图3: KP vs 稳定率
    plt.subplot(2, 3, 3)
    plt.scatter(kp_scales, stability_rates, alpha=0.7, s=60)
    plt.xlabel('KP缩放')
    plt.ylabel('稳定率 (%)')
    plt.title('KP vs 稳定率')
    plt.grid(True, alpha=0.3)

    # 子图4: KD vs 稳定率
    plt.subplot(2, 3, 4)
    plt.scatter(kd_scales, stability_rates, alpha=0.7, s=60)
    plt.xlabel('KD缩放')
    plt.ylabel('稳定率 (%)')
    plt.title('KD vs 稳定率')
    plt.grid(True, alpha=0.3)

    # 子图5: 参数组合排名
    plt.subplot(2, 3, 5)
    sorted_results = sorted(successful_results, key=lambda x: x['stability_rate'], reverse=True)
    top_10 = sorted_results[:10]

    x_labels = [f"KP{r['kp_scale']:.1f}\nKD{r['kd_scale']:.1f}" for r in top_10]
    y_values = [r['stability_rate'] for r in top_10]

    bars = plt.bar(range(len(top_10)), y_values, alpha=0.7)
    plt.xlabel('参数组合')
    plt.ylabel('稳定率 (%)')
    plt.title('前10名参数组合')
    plt.xticks(range(len(top_10)), x_labels, rotation=45)
    plt.grid(True, alpha=0.3)

    # 为最佳结果添加标记
    if bars:
        bars[0].set_color('gold')
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(2)

    # 子图6: 成功率统计
    plt.subplot(2, 3, 6)
    success_count = len(successful_results)
    total_count = len(results)
    failure_count = total_count - success_count

    plt.pie([success_count, failure_count], labels=['成功', '失败'],
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title(f'测试成功率\n({success_count}/{total_count})')

    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"pd_optimization_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"📈 优化结果图表已保存")

    plt.show()


if __name__ == '__main__':
    current_directory = os.getcwd()
    config_file = current_directory + "/g1_config/mujoco_config.yaml"

    # 读取配置
    cfg = read_conf(config_file)

    # 运行自动优化
    results = run_auto_pd_optimization(cfg, render_best=True)

    print("\n🏁 自动PD参数优化完成!")
    print("📝 请根据推荐参数修改配置文件中的kps和kds值")