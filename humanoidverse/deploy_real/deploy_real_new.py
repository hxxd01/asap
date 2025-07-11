#!/usr/bin/env python3
"""
ASAP G1 23DOF全身控制真实机器人部署脚本
基于ASAP训练的运动模仿模型，支持23DOF全身控制
"""

import os
from typing import Union
import numpy as np
import time
import torch
import yaml
import argparse

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class ASAP23DOFConfig:
    """ASAP 23DOF全身控制配置类"""

    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            # 基础控制参数
            self.control_dt = config["control_dt"]
            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            # DDS通信配置
            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            # 模型路径
            self.policy_path = config["policy_path"]

            # 23DOF关节到电机的映射
            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]  # 12个腿部关节
            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]  # 11个腰部+手臂关节

            # PD控制参数
            self.kps = config["kps"]  # 腿部PD参数
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)  # 腿部默认角度

            self.arm_waist_kps = config["arm_waist_kps"]  # 腰部+手臂PD参数
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)  # 腰部+手臂默认角度

            # 手腕关节锁定参数
            self.wrist_kps = config["wrist_kps"]  # 手腕PD参数
            self.wrist_kds = config["wrist_kds"]
            self.wrist_target = np.array(config["wrist_target"], dtype=np.float32)  # 手腕默认角度
            self.wrist_joint_idx = [19, 20, 21, 26, 27, 28]  # 手腕关节索引

            # 观测缩放参数
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            # ASAP特定参数
            self.frame_stack = config["frame_stack"]
            self.num_single_obs = config["num_single_obs"]
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.cycle_time = config["cycle_time"]

            # ASAP观测缩放参数
            self.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
            self.obs_scale_dof_pos = config["obs_scale_dof_pos"]
            self.obs_scale_dof_vel = config["obs_scale_dof_vel"]
            self.obs_scale_gvec = config["obs_scale_gvec"]
            self.obs_scale_refmotion = config["obs_scale_refmotion"]
            self.obs_scale_hist = config["obs_scale_hist"]

            # 限制参数
            self.clip_observations = config["clip_observations"]
            self.clip_actions = config["clip_actions"]
            self.use_noise = config["use_noise"]

            # 23DOF默认位置组合
            self.default_dof_pos_23 = np.concatenate([
                self.default_angles,  # 腿部12DOF
                self.arm_waist_target  # 腰部+手臂11DOF
            ])


class ASAP23DOFController:
    """ASAP 23DOF全身控制器"""

    def __init__(self, config: ASAP23DOFConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # 加载策略网络
        self.policy = torch.jit.load(config.policy_path)
        print(f"✅ 成功加载ASAP策略网络: {config.policy_path}")

        # 初始化23DOF全身控制变量
        self.qj = np.zeros(config.num_actions, dtype=np.float32)  # 23DOF关节位置
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)  # 23DOF关节速度
        self.action = np.zeros(config.num_actions, dtype=np.float32)

        self.target_dof_pos = config.default_dof_pos_23.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # 初始化历史观测字典
        self.hist_dict = {
            'actions': np.zeros((config.frame_stack, config.num_actions), dtype=np.float32),
            'base_ang_vel': np.zeros((config.frame_stack, 3), dtype=np.float32),
            'dof_pos': np.zeros((config.frame_stack, config.num_actions), dtype=np.float32),
            'dof_vel': np.zeros((config.frame_stack, config.num_actions), dtype=np.float32),
            'projected_gravity': np.zeros((config.frame_stack, 3), dtype=np.float32),
            'ref_motion_phase': np.zeros((config.frame_stack, 1), dtype=np.float32),
        }

        # 初始化历史观测缓存
        history_keys = ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']
        hist_obs = []
        for key in history_keys:
            hist_obs.append(self.hist_dict[key].reshape(1, -1))
        self.hist_obs_c = np.concatenate(hist_obs, axis=1)

        # 初始化DDS通信
        if config.msg_type == "hg":
            # G1使用hg消息类型
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # H1使用go消息类型
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # 等待机器人状态数据
        self.wait_for_low_state()

        # 初始化命令消息
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("✅ 成功连接到G1机器人")

    def zero_torque_state(self):
        print("🔄 进入零扭矩状态")
        print("⏳ 等待Start按钮启动...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("🚶 移动到默认位置 (2秒)...")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        # 组合所有控制的关节索引和参数
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        print(default_pos)
        dof_size = len(dof_idx)

        # 记录当前位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # 平滑移动到默认位置
        for i in range(num_step):
            alpha = i / num_step
            # 策略控制的23DOF关节
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # 手腕关节锁定到默认角度
            for j, motor_idx in enumerate(self.config.wrist_joint_idx):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[j]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("✅ 已到达默认位置")

    def default_pos_state(self):
        print("🏠 保持默认位置状态")
        print("⏳ 等待A按钮开始ASAP 23DOF策略...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # 腿部关节
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # 手臂和腰部关节
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # 手腕关节锁定到默认角度
            for i, motor_idx in enumerate(self.config.wrist_joint_idx):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def update_hist_obs(self, obs_single):
        """更新历史观测"""
        slices = {
            'actions': slice(0, 23),
            'base_ang_vel': slice(23, 26),
            'dof_pos': slice(26, 49),
            'dof_vel': slice(49, 72),
            'projected_gravity': slice(72, 75),
            'ref_motion_phase': slice(75, 76)
        }

        for key, slc in slices.items():
            # 移除最旧的条目，添加新观测
            arr = np.delete(self.hist_dict[key], -1, axis=0)
            arr = np.vstack((obs_single[slc], arr))
            self.hist_dict[key] = arr

        # 连接历史观测
        hist_obs = np.concatenate([
            self.hist_dict[key].reshape(1, -1)
            for key in self.hist_dict.keys()
        ], axis=1).astype(np.float32)

        return hist_obs

    def get_obs(self, robot_data, action):
        """构建380维ASAP观测向量"""
        config = self.config

        # 添加噪声（真实部署时关闭）
        if config.use_noise:
            noise_base_ang_vel = (np.random.rand(3) * 2. - 1.) * 0.3
            noise_projected_gravity = (np.random.rand(3) * 2. - 1.) * 0.2
            noise_dof_pos = (np.random.rand(23) * 2. - 1.) * 0.01
            noise_dof_vel = (np.random.rand(23) * 2. - 1.) * 1.0
        else:
            noise_base_ang_vel = np.zeros(3)
            noise_projected_gravity = np.zeros(3)
            noise_dof_pos = np.zeros(23)
            noise_dof_vel = np.zeros(23)

        # 计算运动相位
        ref_motion_phase = (self.counter + 1) * config.control_dt / config.cycle_time
        ref_motion_phase = np.clip(ref_motion_phase % 1.0, 0, 1)

        # 构建单帧观测 (76维)
        obs_single = np.zeros(config.num_single_obs, dtype=np.float32)

        # 0:23 - 动作
        obs_single[0:23] = action

        # 23:26 - 基座角速度
        obs_single[23:26] = (robot_data['base_angvel'] + noise_base_ang_vel) * config.obs_scale_base_ang_vel

        # 26:49 - 关节位置偏移
        obs_single[26:49] = (robot_data['dof_pos_offset'] + noise_dof_pos) * config.obs_scale_dof_pos

        # 49:72 - 关节速度
        obs_single[49:72] = (robot_data['dof_vel'] + noise_dof_vel) * config.obs_scale_dof_vel

        # 72:75 - 重力向量
        obs_single[72:75] = (robot_data['gvec'] + noise_projected_gravity) * config.obs_scale_gvec

        # 75:76 - 运动相位
        obs_single[75] = ref_motion_phase * config.obs_scale_refmotion

        # 更新历史观测
        hist_obs_cat = self.update_hist_obs(obs_single)

        # 构建完整观测 (380维)
        num_obs_input = (config.frame_stack + 1) * config.num_single_obs
        obs_all = np.zeros(num_obs_input, dtype=np.float32)

        # 当前观测
        obs_all[0:23] = obs_single[0:23]  # 动作
        obs_all[23:26] = obs_single[23:26]  # 基座角速度
        obs_all[26:49] = obs_single[26:49]  # 关节位置
        obs_all[49:72] = obs_single[49:72]  # 关节速度

        # 历史观测 (304维)
        obs_all[72:376] = hist_obs_cat[0] * config.obs_scale_hist

        # 当前重力和相位
        obs_all[376:379] = obs_single[72:75]  # 重力向量
        obs_all[379] = obs_single[75]  # 运动相位

        # 限制观测范围
        obs_all = np.clip(obs_all, -config.clip_observations, config.clip_observations)

        return obs_all.reshape(1, -1)

    def run(self):
        self.counter += 1

        # 读取23DOF关节状态
        # 腿部12DOF
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        # 腰部和手臂11DOF
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.qj[12 + i] = self.low_state.motor_state[motor_idx].q
            self.dqj[12 + i] = self.low_state.motor_state[motor_idx].dq

        # 处理IMU数据
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32).flatten()

        if self.config.imu_type == "torso":
            # H1和H1_2的IMU在躯干上，需要转换到骨盆坐标系
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega,
                                               imu_quat=quat, imu_omega=ang_vel)

        # 构建传感器数据字典
        gravity_orientation = get_gravity_orientation(quat)
        robot_data = {
            'dof_pos': self.qj,
            'dof_vel': self.dqj,
            'base_angvel': ang_vel,
            'gvec': gravity_orientation,
            'dof_pos_offset': self.qj - self.config.default_dof_pos_23,
        }

        # 获取遥控器命令 (ASAP不使用遥控器，但保留接口)
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        # 构建380维观测向量
        obs_buff = self.get_obs(robot_data, self.action)

        # 策略网络推理
        obs_tensor = torch.from_numpy(obs_buff).float()
        with torch.no_grad():
            self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        # 限制动作范围
        self.action = np.clip(self.action, -self.config.clip_actions, self.config.clip_actions)

        # 🌟 23DOF全身控制：计算所有23DOF的目标位置
        target_all_pos = self.config.default_dof_pos_23 + self.action * self.config.action_scale

        # 发送电机命令
        # 腿部12DOF控制
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_all_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 🌟 腰部和手臂11DOF控制（23DOF全身控制的关键）
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_all_pos[12 + i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 🌟 手腕关节锁定控制（解决0力矩问题）
        for i, motor_idx in enumerate(self.config.wrist_joint_idx):
            self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 发送命令
        self.send_cmd(self.low_cmd)

        # 调试信息输出
        if self.counter % 100 == 0:  # 每2秒打印一次
            phase = (self.counter * self.config.control_dt / self.config.cycle_time) % 1.0
            print(f"⏱️  时间: {self.counter * self.config.control_dt:.2f}s, "
                  f"相位: {phase:.3f}, "
                  f"cmd: [{self.cmd[0]:.2f}, {self.cmd[1]:.2f}, {self.cmd[2]:.2f}]")

        time.sleep(self.config.control_dt)


def main():
    parser = argparse.ArgumentParser(description='ASAP G1 23DOF全身控制真实机器人部署')
    parser.add_argument("net", type=str, help="网络接口 (例如: eth0)")
    parser.add_argument("config", type=str, help="配置文件名 (在configs文件夹中)", default="asap_g1_motion_tracking_23dof.yaml")
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(CURRENT_DIR, "configs", args.config)
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        exit(1)

    config = ASAP23DOFConfig(config_path)

    # 初始化DDS通信
    ChannelFactoryInitialize(0, args.net)

    controller = ASAP23DOFController(config)

    print("🚀 开始ASAP G1 23DOF全身控制部署...")
    print(f"📊 观测维度: {config.num_obs}, 动作维度: {config.num_actions}")
    print(f"🔄 运动周期: {config.cycle_time}秒")
    print("🦾 全身控制: 腿部12DOF + 腰部3DOF + 手臂8DOF")

    try:
        # 进入零扭矩状态，按Start键继续
        controller.zero_torque_state()

        # 移动到默认位置
        controller.move_to_default_pos()

        # 进入默认位置状态，按A键继续
        controller.default_pos_state()

        print("🎬 ASAP 23DOF策略开始运行...")
        print("🛑 按Select键退出")

        while True:
            controller.run()
            # 按Select键退出
            if controller.remote_controller.button[KeyMap.select] == 1:
                break

    except KeyboardInterrupt:
        print("\n⚠️  检测到Ctrl+C")

    # 进入阻尼状态
    print("🛑 进入阻尼状态...")
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("🏁 部署完成")


if __name__ == "__main__":
    main()