import time
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--duration", type=float, default=10.0, help="simulation duration in seconds")
    args = parser.parse_args()
    
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = args.duration
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    print(f"🤖 开始无头Mujoco仿真测试...")
    print(f"📁 模型文件: {xml_path}")
    print(f"🧠 策略文件: {policy_path}")
    print(f"⏱️  仿真时长: {simulation_duration}秒")
    print(f"🎮 控制指令: {cmd}")

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    total_steps = int(simulation_duration / simulation_dt)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    print("📥 加载策略模型...")
    policy = torch.jit.load(policy_path)
    print("✅ 策略模型加载成功！")

    # Reset robot to initial position
    d.qpos[7:] = default_angles
    mujoco.mj_forward(m, d)

    print("🚀 开始仿真...")
    start_time = time.time()
    
    # 记录一些统计信息
    step_times = []
    actions_history = []
    
    for step in range(total_steps):
        step_start = time.time()
        
        # Control
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        
        # Physics step
        mujoco.mj_step(m, d)
        
        counter += 1
        if counter % control_decimation == 0:
            # Create observation
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj = (qj - default_angles) * dof_pos_scale
            dqj = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega = omega * ang_vel_scale

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9 : 9 + num_actions] = qj
            obs[9 + num_actions : 9 + 2 * num_actions] = dqj
            obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
            
            # Policy inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).detach().numpy().squeeze()
            
            # Transform action to target_dof_pos
            target_dof_pos = action * action_scale + default_angles
            
            # 记录动作
            actions_history.append(action.copy())

        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # 每1000步打印一次进度
        if step % 1000 == 0:
            progress = (step / total_steps) * 100
            avg_step_time = np.mean(step_times[-1000:]) if len(step_times) >= 1000 else np.mean(step_times)
            print(f"📊 进度: {progress:.1f}% | 平均步长时间: {avg_step_time*1000:.2f}ms | 关节位置: {d.qpos[7:12]}")
        
        # 时间控制
        time_until_next_step = m.opt.timestep - step_time
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    total_time = time.time() - start_time
    
    print("\n🎉 仿真完成！")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    print(f"🔢 总步数: {total_steps}")
    print(f"⚡ 平均帧率: {total_steps/total_time:.1f} FPS")
    print(f"🎯 最终关节位置: {d.qpos[7:]}")
    print(f"🎮 动作范围: [{np.min(actions_history):.3f}, {np.max(actions_history):.3f}]")
    print("✅ 策略部署测试成功！无需GPU即可运行。") 