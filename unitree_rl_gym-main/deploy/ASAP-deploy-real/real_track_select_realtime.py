#!/usr/bin/env python3
"""
真实世界G1机器人轨迹采集脚本 - 真实时间版本
基于ASAP 23DOF PBHC部署，采集格式与mujoco_track.py完全一致
增加Select键停止策略功能
✅ 使用真实时间戳记录motion_time，而非理论控制时间
"""

import os
from typing import Union
import numpy as np
import time
import torch
import yaml
import argparse
import signal
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
from scipy.spatial.transform import Rotation as R

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


def parse_joint_limits_from_xml(xml_path):
    """从URDF文件解析关节力矩限制（仅解析23个控制关节）"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # G1机器人的23个控制关节（按控制顺序）
    control_joint_names = [
        # 腿部关节 (12个)
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        # 腰部关节 (3个)
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        # 手臂关节 (8个)
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
    ]
    
    torque_limits = []
    joint_names = []
    
    # 按顺序查找每个控制关节的力矩限制
    for joint_name in control_joint_names:
        joint_elem = root.find(f".//joint[@name='{joint_name}']")
        
        if joint_elem is not None:
            joint_names.append(joint_name)
            
            # 查找关节限制
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                # 获取力矩限制（URDF中的effort属性）
                effort_str = limit_elem.attrib.get('effort', None)
                if effort_str is not None:
                    torque_limit = float(effort_str)
                    print(f"✅ 关节 {joint_name}: {torque_limit} Nm")
                else:
                    # 如果没有effort属性，使用默认值
                    torque_limit = 100.0
                    print(f"⚠️  关节 {joint_name} 没有effort属性，使用默认值 {torque_limit} Nm")
            else:
                # 如果没有limit标签，使用默认值
                torque_limit = 100.0
                print(f"⚠️  关节 {joint_name} 没有limit标签，使用默认值 {torque_limit} Nm")
        else:
            # 如果没有找到关节，使用默认值
            torque_limit = 100.0
            joint_names.append(joint_name)
            print(f"❌ 未找到关节 {joint_name}，使用默认值 {torque_limit} Nm")
        
        torque_limits.append(torque_limit)
    
    print(f"📊 总共解析了 {len(torque_limits)} 个控制关节的力矩限制")
    return np.array(torque_limits, dtype=np.float32), joint_names


class RealTrackConfig:
    """真实世界轨迹采集配置类"""
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
            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            config_arm_waist = config["arm_waist_joint2motor_idx"]
            
            # PBHC语义重映射
            self.arm_waist_joint2motor_idx = []
            for i, idx in enumerate(config_arm_waist):
                if i >= 7:  # 右臂部分
                    real_idx = 22 + (i - 7)  # 22,23,24,25
                    self.arm_waist_joint2motor_idx.append(real_idx)
                else:  # 腰部和左臂保持不变
                    self.arm_waist_joint2motor_idx.append(idx)
            
            # PD控制参数
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            # 手腕关节锁定参数
            self.wrist_kps = config["wrist_kps"]
            self.wrist_kds = config["wrist_kds"]
            self.wrist_target = np.array(config["wrist_target"], dtype=np.float32)
            self.wrist_joint_idx = [19, 20, 21, 26, 27, 28]

            # ASAP参数
            self.frame_stack = config["frame_stack"]
            self.num_single_obs = config["num_single_obs"]
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.cycle_time = config["cycle_time"]

            # 观测缩放参数
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
            self.action_scale = config["action_scale"]
            
            # 力矩限制参数
            self.use_torque_limit = config.get("use_torque_limit", False)
            self.torque_limit_scale = config.get("torque_limit_scale", 0.8)  # 默认使用URDF限制的80%
            
            # 23DOF默认位置组合
            self.default_dof_pos_23 = np.concatenate([
                self.default_angles,
                self.arm_waist_target
            ])

            # 轨迹采集参数
            self.episode_steps = config.get("episode_steps", 300)  # 每个episode步数
            self.total_steps = config.get("total_steps", 3000)     # 总采集步数
            
            # Termination参数
            self.use_termination = config.get("use_termination", True)
            self.termination_gravity_x = config.get("termination_gravity_x", 0.8)
            self.termination_gravity_y = config.get("termination_gravity_y", 0.8)
            
            # XML文件路径 - 用于解析关节轴向量
            self.xml_path = config["xml_path"]


class RealTrackController:
    """真实世界轨迹采集控制器 - 真实时间版本"""
    def __init__(self, config: RealTrackConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # 加载策略网络
        self.policy = torch.jit.load(config.policy_path)
        print(f"✅ 成功加载ASAP策略网络: {config.policy_path}")

        # 解析关节轴向量 - 修复pose_aa计算
        self.dof_axis = parse_dof_axis_from_xml(config.xml_path)
        print(f"✅ 成功解析关节轴向量: {self.dof_axis.shape} 从 {config.xml_path}")
        
        # 解析关节力矩限制（从URDF文件读取）
        if config.use_torque_limit:
            urdf_path = os.path.join(os.path.dirname(__file__), 'g1_urdf', 'g1_29dof_anneal_23dof.urdf')
            self.joint_torque_limits, self.joint_names = parse_joint_limits_from_xml(urdf_path)
            self.scaled_torque_limits = self.joint_torque_limits * config.torque_limit_scale
            print(f"✅ 成功解析关节力矩限制: {len(self.joint_torque_limits)}个关节（从URDF）")
            print(f"🔧 力矩缩放比例: {config.torque_limit_scale:.2f}")
            print(f"📊 力矩限制范围: {self.scaled_torque_limits.min():.1f} - {self.scaled_torque_limits.max():.1f} Nm")
        else:
            self.joint_torque_limits = None
            self.scaled_torque_limits = None
            print("⚠️  未启用关节力矩限制")

        # 初始化23DOF控制变量
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        
        self.target_dof_pos = config.default_dof_pos_23.copy()
        self.counter = 0
        self.current_step = 0

        # ⏰ 真实时间追踪
        self.start_time = None  # 轨迹开始时间
        self.last_real_time = None  # 上一帧的真实时间

        # 🎬 轨迹数据采集初始化 - 与mujoco_track.py格式完全一致
        self.motions_for_saving = {
            'root_trans_offset': [],    # base位置偏移 (暂时设为零，等待动补)
            'pose_aa': [],              # 姿态轴角表示 (base + joints + 虚拟关节)
            'dof': [],                  # 关节角度 (23DOF)
            'root_rot': [],             # base四元数 (从IMU获取)
            'action': [],               # 策略动作
            'terminate': [],            # 终止标志
            'root_lin_vel': [],         # base线速度 (暂时设为零，等待动补)
            'root_ang_vel': [],         # base角速度 (从IMU获取)
            'dof_vel': [],              # 关节速度 (23DOF)
            'motion_times': [],         # ✅ 从0开始的真实时间（基于实际时间间隔）
            'real_dt': [],              # ✅ 真实时间间隔
            'theoretical_times': []     # 🔍 理论时间（用于对比分析）
        }

        # 采集统计
        self.termination_stats = {
            'gravity_terminations': 0,
            'normal_completions': 0,
            'total_episodes': 0
        }

        # 时间统计
        self.time_stats = {
            'total_real_time': 0.0,
            'total_theoretical_time': 0.0,
            'max_dt': 0.0,
            'min_dt': float('inf'),
            'dt_violations': 0,  # 超过1.5倍控制周期的次数
        }

        # 力矩限制统计
        self.torque_limit_count = 0

        # 🛡️ 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 🕐 动态sleep时间控制
        self.loop_start_time = None
        self.adaptive_sleep_stats = {
            'total_adaptive_sleeps': 0,
            'total_overruns': 0,
            'avg_processing_time': 0.0,
            'avg_sleep_time': 0.0,
            'max_processing_time': 0.0,
            'min_sleep_time': float('inf')
        }

        # 初始化历史观测
        self.hist_dict = {
            'actions': np.zeros((config.frame_stack, config.num_actions), dtype=np.float32),
            'base_ang_vel': np.zeros((config.frame_stack, 3), dtype=np.float32),
            'dof_pos': np.zeros((config.frame_stack, config.num_actions), dtype=np.float32),
            'dof_vel': np.zeros((config.frame_stack, config.num_actions), dtype=np.float32),
            'projected_gravity': np.zeros((config.frame_stack, 3), dtype=np.float32),
            'ref_motion_phase': np.zeros((config.frame_stack, 1), dtype=np.float32),
        }

        history_keys = ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'projected_gravity', 'ref_motion_phase']
        hist_obs = []
        for key in history_keys:
            hist_obs.append(self.hist_dict[key].reshape(1, -1))
        self.hist_obs_c = np.concatenate(hist_obs, axis=1)

        # 初始化DDS通信
        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # 等待机器人状态数据
        self.wait_for_low_state()

        # 初始化命令消息
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def _signal_handler(self, signum, frame):
        """信号处理函数，保存轨迹数据"""
        print(f"\n⚠️  检测到信号 {signum} (Ctrl+C)，正在保存轨迹数据...")
        self.save_trajectory_data()
        print("🏁 程序安全退出。")
        exit(0)

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
            self.start_loop_timing()
            self.adaptive_sleep("wait_connection")
        print("✅ 成功连接到G1机器人")

    def zero_torque_state(self):
        print("🔄 进入零扭矩状态")
        print("⏳ 等待Start按钮启动...")
        while self.remote_controller.button[KeyMap.start] != 1:
            self.start_loop_timing()
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self.adaptive_sleep("zero_torque")

    def move_to_default_pos(self):
        print("🚶 移动到默认位置 (2秒)...")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            self.start_loop_timing()
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = self.low_cmd.motor_cmd[motor_idx].q - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, kps[j], kds[j], current_vel)
                if should_exit:
                    print("🚨 移动到默认位置时检测到力矩超限，程序退出")
                    exit(1)
            
            for j, motor_idx in enumerate(self.config.wrist_joint_idx):
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[j]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = self.config.wrist_target[j] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.wrist_kps[j], self.config.wrist_kds[j], current_vel)
                if should_exit:
                    print("🚨 移动到默认位置时检测到力矩超限，程序退出")
                    exit(1)
            
            self.send_cmd(self.low_cmd)
            self.adaptive_sleep("move_to_default")
        print("✅ 已到达默认位置")

    def default_pos_state(self):
        print("🏠 保持默认位置状态")
        print("⏳ 等待A按钮开始轨迹采集...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self.start_loop_timing()
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = self.config.default_angles[i] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.kps[i], self.config.kds[i], current_vel)
                if should_exit:
                    print("🚨 保持默认位置时检测到力矩超限，程序退出")
                    exit(1)

            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = self.config.arm_waist_target[i] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.arm_waist_kps[i], self.config.arm_waist_kds[i], current_vel)
                if should_exit:
                    print("🚨 保持默认位置时检测到力矩超限，程序退出")
                    exit(1)
            
            for i, motor_idx in enumerate(self.config.wrist_joint_idx):
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = self.config.wrist_target[i] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.wrist_kps[i], self.config.wrist_kds[i], current_vel)
                if should_exit:
                    print("🚨 保持默认位置时检测到力矩超限，程序退出")
                    exit(1)
                
            self.send_cmd(self.low_cmd)
            self.adaptive_sleep("default_position")

    def start_loop_timing(self):
        """开始循环时间记录（用于动态sleep控制）"""
        self.loop_start_time = time.time()

    def adaptive_sleep(self, phase_name="unknown"):
        """动态sleep控制 - 确保稳定的控制频率
        
        Args:
            phase_name: 当前执行阶段名称（用于调试）
        """
        if self.loop_start_time is None:
            # 如果没有记录开始时间，使用固定sleep
            time.sleep(self.config.control_dt)
            return
        
        # 计算实际处理时间
        current_time = time.time()
        processing_time = current_time - self.loop_start_time
        
        # 计算动态sleep时间
        target_sleep_time = self.config.control_dt - processing_time
        actual_sleep_time = max(0.001, target_sleep_time)  # 最小sleep 1ms
        
        # 更新统计
        self.adaptive_sleep_stats['total_adaptive_sleeps'] += 1
        if target_sleep_time <= 0:
            self.adaptive_sleep_stats['total_overruns'] += 1
        
        # 更新累计统计
        total_sleeps = self.adaptive_sleep_stats['total_adaptive_sleeps']
        self.adaptive_sleep_stats['avg_processing_time'] = (
            (self.adaptive_sleep_stats['avg_processing_time'] * (total_sleeps - 1) + processing_time) / total_sleeps
        )
        self.adaptive_sleep_stats['avg_sleep_time'] = (
            (self.adaptive_sleep_stats['avg_sleep_time'] * (total_sleeps - 1) + actual_sleep_time) / total_sleeps
        )
        self.adaptive_sleep_stats['max_processing_time'] = max(
            self.adaptive_sleep_stats['max_processing_time'], processing_time
        )
        self.adaptive_sleep_stats['min_sleep_time'] = min(
            self.adaptive_sleep_stats['min_sleep_time'], actual_sleep_time
        )
        
        # 执行动态sleep
        time.sleep(actual_sleep_time)
        
        # 重置循环开始时间
        self.loop_start_time = None

    def start_trajectory_timing(self):
        """开始轨迹时间记录"""
        self.start_time = time.time()
        self.last_real_time = self.start_time
        print(f"⏰ 轨迹时间记录开始: {datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S.%f')[:-3]}")

    def apply_torque_limit(self, motor_cmd, motor_idx, position_error, kp, kd, velocity):
        """应用关节力矩限制
        
        策略：
        1. 计算期望力矩 = kp * 位置误差 + kd * 速度
        2. 如果超过限制：将tau置0并返回True（表示需要退出策略）
        3. 如果在限制内：正常控制，tau=0，返回False
        
        Returns:
            bool: True表示力矩超限需要退出策略，False表示正常
        """
        motor_cmd.tau = 0  # 位置控制模式tau始终设为0
        
        if not self.config.use_torque_limit or self.scaled_torque_limits is None:
            return False  # 不使用力矩限制，正常继续
        
        # 计算期望力矩
        expected_torque = kp * position_error + kd * velocity
        
        # 获取该关节的力矩限制
        if motor_idx < len(self.scaled_torque_limits):
            torque_limit = self.scaled_torque_limits[motor_idx]
        else:
            # 如果超出范围，使用默认限制
            torque_limit = 50.0  # 默认50 Nm
        
        # 检查力矩是否超限
        if abs(expected_torque) > torque_limit:
            # 力矩超限：记录并要求退出策略
            self.torque_limit_count += 1
            
            print(f"🚨 关节 {motor_idx} 力矩超限: {expected_torque:.1f} > {torque_limit:.1f} Nm")
            print(f"🛑 为保护硬件安全，退出当前策略")
            
            return True  # 要求退出策略
        
        return False  # 力矩在限制范围内，继续正常执行

    def check_termination(self, robot_data):
        """检查termination条件 - 与mujoco_track.py一致"""
        if not self.config.use_termination:
            return False

        gvec = robot_data['gvec']
        gravity_x_violation = abs(gvec[0]) > self.config.termination_gravity_x
        gravity_y_violation = abs(gvec[1]) > self.config.termination_gravity_y
        should_terminate = gravity_x_violation or gravity_y_violation

        if should_terminate:
            print(f"[Termination] Step {self.counter}: "
                  f"gravity_x={gvec[0]:.3f}(>{self.config.termination_gravity_x}), "
                  f"gravity_y={gvec[1]:.3f}(>{self.config.termination_gravity_y})")

        return should_terminate

    def collect_trajectory_data(self):
        """采集轨迹数据 - 使用真实时间戳"""
        current_time = time.time()
        
        # ✅ 计算真实时间和时间间隔
        motion_time = current_time - self.start_time  # ✅ 从0开始的真实时间
        real_dt = current_time - self.last_real_time if self.last_real_time is not None else self.config.control_dt
        theoretical_time = self.counter * self.config.control_dt
        
        # 更新时间统计
        self.time_stats['max_dt'] = max(self.time_stats['max_dt'], real_dt)
        self.time_stats['min_dt'] = min(self.time_stats['min_dt'], real_dt)
        if real_dt > 1.5 * self.config.control_dt:
            self.time_stats['dt_violations'] += 1
        
        self.last_real_time = current_time
        
        # ✅ 关节角度 (23DOF) - 与mujoco_track.py中的dof对应
        dof = self.qj.copy()
        
        # ✅ base四元数 (从IMU获取) - 与mujoco_track.py中的root_rot对应
        quat = self.low_state.imu_state.quaternion
        # 转换为xyzw顺序与mujoco_track.py一致
        root_rot = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        
        # 🔄 base位置偏移 (暂时设为零，等待动补)
        root_trans_offset = np.zeros(3, dtype=np.float32)
        
        # ✅ base角速度 (从IMU获取) - 与mujoco_track.py中的root_ang_vel对应
        root_ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        
        # 🔄 base线速度 (暂时设为零，等待动补)
        root_lin_vel = np.zeros(3, dtype=np.float32)
        
        # ✅ 关节速度 (23DOF) - 与mujoco_track.py中的dof_vel对应
        dof_vel = self.dqj.copy()
        
        # ✅ 策略动作 - 与mujoco_track.py中的action对应
        action = self.action.copy()
        
        # 🔧 构建pose_aa - 与mujoco_track.py格式一致
        # base四元数转轴角
        root_rot_quat = [root_rot[3], root_rot[0], root_rot[1], root_rot[2]]  # wxyz
        root_rot_vec = R.from_quat(root_rot_quat).as_rotvec()  # shape (3,)
        
        # 关节角度与真实轴向量相乘 - 已修复pose_aa计算
        joint_aa = dof[:, None] * self.dof_axis  # shape (23, 3)
        
        # 拼接：base轴角 + 关节轴角 + 3个虚拟关节
        num_augment_joint = 3
        pose_aa = np.concatenate([
            root_rot_vec[None, :],  # (1, 3)
            joint_aa,               # (23, 3)
            np.zeros((num_augment_joint, 3), dtype=np.float32)  # (3, 3)
        ], axis=0)  # shape (27, 3)
        
        # 保存数据 - 增强的时间记录
        self.motions_for_saving['root_trans_offset'].append(root_trans_offset)
        self.motions_for_saving['root_rot'].append(root_rot)
        self.motions_for_saving['dof'].append(dof)
        self.motions_for_saving['pose_aa'].append(pose_aa)
        self.motions_for_saving['action'].append(action)
        self.motions_for_saving['root_lin_vel'].append(root_lin_vel)
        self.motions_for_saving['root_ang_vel'].append(root_ang_vel)
        self.motions_for_saving['dof_vel'].append(dof_vel)
        self.motions_for_saving['motion_times'].append(motion_time)  # ✅ 从0开始的真实时间
        self.motions_for_saving['real_dt'].append(real_dt)  # ✅ 真实时间间隔
        self.motions_for_saving['theoretical_times'].append(theoretical_time)  # 🔍 理论时间
        
        return motion_time, real_dt

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
            arr = np.delete(self.hist_dict[key], -1, axis=0)
            arr = np.vstack((obs_single[slc], arr))
            self.hist_dict[key] = arr
        
        hist_obs = np.concatenate([
            self.hist_dict[key].reshape(1, -1)
            for key in self.hist_dict.keys()
        ], axis=1).astype(np.float32)
        
        return hist_obs

    def get_obs(self, robot_data, action):
        """构建380维ASAP观测向量"""
        config = self.config
        
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
        
        ref_motion_phase = (self.counter + 1) * config.control_dt / config.cycle_time
        ref_motion_phase = np.clip(ref_motion_phase % 1.0, 0, 1)
        
        obs_single = np.zeros(config.num_single_obs, dtype=np.float32)
        obs_single[0:23] = action
        obs_single[23:26] = (robot_data['base_angvel'] + noise_base_ang_vel) * config.obs_scale_base_ang_vel
        obs_single[26:49] = (robot_data['dof_pos_offset'] + noise_dof_pos) * config.obs_scale_dof_pos
        obs_single[49:72] = (robot_data['dof_vel'] + noise_dof_vel) * config.obs_scale_dof_vel
        obs_single[72:75] = (robot_data['gvec'] + noise_projected_gravity) * config.obs_scale_gvec
        obs_single[75] = ref_motion_phase * config.obs_scale_refmotion
        
        hist_obs_cat = self.update_hist_obs(obs_single)
        
        num_obs_input = (config.frame_stack + 1) * config.num_single_obs
        obs_all = np.zeros(num_obs_input, dtype=np.float32)
        
        obs_all[0:23] = obs_single[0:23]
        obs_all[23:26] = obs_single[23:26]
        obs_all[26:49] = obs_single[26:49]
        obs_all[49:72] = obs_single[49:72]
        obs_all[72:376] = hist_obs_cat[0] * config.obs_scale_hist
        obs_all[376:379] = obs_single[72:75]
        obs_all[379] = obs_single[75]
        
        obs_all = np.clip(obs_all, -config.clip_observations, config.clip_observations)
        return obs_all.reshape(1, -1)

    def run_episode(self):
        """运行一个episode"""
        episode_terminated_early = False
        episode_steps = 0
        
        for step in range(self.config.episode_steps):
            # ⏰ 开始循环时间记录（用于动态sleep控制）
            self.start_loop_timing()
            
            self.counter += 1
            self.current_step += 1
            episode_steps += 1
            
            # 检查Select键退出
            if self.remote_controller.button[KeyMap.select] == 1:
                print("🛑 检测到Select按钮，退出轨迹采集")
                return "user_exit"
            
            # 读取23DOF关节状态
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.qj[i] = self.low_state.motor_state[motor_idx].q
                self.dqj[i] = self.low_state.motor_state[motor_idx].dq
            
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.qj[12 + i] = self.low_state.motor_state[motor_idx].q
                self.dqj[12 + i] = self.low_state.motor_state[motor_idx].dq

            # 处理IMU数据
            quat = self.low_state.imu_state.quaternion
            ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32).flatten()

            if self.config.imu_type == "torso":
                waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
                waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
                quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, 
                                                 imu_quat=quat, imu_omega=ang_vel)

            # 构建机器人数据
            gravity_orientation = get_gravity_orientation(quat)
            robot_data = {
                'dof_pos': self.qj,
                'dof_vel': self.dqj,
                'base_angvel': ang_vel,
                'gvec': gravity_orientation,
                'dof_pos_offset': self.qj - self.config.default_dof_pos_23,
            }

            # 检查termination
            should_terminate = self.check_termination(robot_data)

            # 策略推理
            obs_buff = self.get_obs(robot_data, self.action)
            obs_tensor = torch.from_numpy(obs_buff).float()
            with torch.no_grad():
                self.action = self.policy(obs_tensor).detach().numpy().squeeze()
            
            self.action = np.clip(self.action, -self.config.clip_actions, self.config.clip_actions)
            target_all_pos = self.config.default_dof_pos_23 + self.action * self.config.action_scale

            # 发送电机命令
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = target_all_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = target_all_pos[i] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.kps[i], self.config.kds[i], current_vel)
                if should_exit:
                    print("🚨 策略执行中检测到力矩超限，退出当前episode")
                    return "torque_limit_exit"
            
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = target_all_pos[12 + i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = target_all_pos[12 + i] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.arm_waist_kps[i], self.config.arm_waist_kds[i], current_vel)
                if should_exit:
                    print("🚨 策略执行中检测到力矩超限，退出当前episode")
                    return "torque_limit_exit"

            for i, motor_idx in enumerate(self.config.wrist_joint_idx):
                current_pos = self.low_state.motor_state[motor_idx].q
                current_vel = self.low_state.motor_state[motor_idx].dq
                
                self.low_cmd.motor_cmd[motor_idx].q = self.config.wrist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.wrist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.wrist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
                # 应用力矩限制
                position_error = self.config.wrist_target[i] - current_pos
                should_exit = self.apply_torque_limit(self.low_cmd.motor_cmd[motor_idx], motor_idx, 
                                                    position_error, self.config.wrist_kps[i], self.config.wrist_kds[i], current_vel)
                if should_exit:
                    print("🚨 策略执行中检测到力矩超限，退出当前episode")
                    return "torque_limit_exit"

            self.send_cmd(self.low_cmd)

            # ✅ 采集轨迹数据 - 使用真实时间
            motion_time, real_dt = self.collect_trajectory_data()
            
            # 设置terminate标志
            if should_terminate:
                self.motions_for_saving['terminate'].append(True)
                print(f"[Early Termination] Episode terminated at step {episode_steps}/{self.config.episode_steps}")
                episode_terminated_early = True
            elif episode_steps == self.config.episode_steps:
                self.motions_for_saving['terminate'].append(True)
            else:
                self.motions_for_saving['terminate'].append(False)

            # 调试输出 - 增强的时间信息与动态sleep统计
            if self.current_step % 50 == 0:
                phase = (self.counter * self.config.control_dt / self.config.cycle_time) % 1.0
                dt_error = abs(real_dt - self.config.control_dt) / self.config.control_dt * 100
                
                # 动态sleep统计
                if self.adaptive_sleep_stats['total_adaptive_sleeps'] > 0:
                    avg_proc_ms = self.adaptive_sleep_stats['avg_processing_time'] * 1000
                    avg_sleep_ms = self.adaptive_sleep_stats['avg_sleep_time'] * 1000
                    overrun_rate = self.adaptive_sleep_stats['total_overruns'] / self.adaptive_sleep_stats['total_adaptive_sleeps'] * 100
                    print(f"⏱️  步数: {self.current_step}/{self.config.total_steps}, "
                          f"运行时间: {motion_time:.2f}s, "
                          f"实际dt: {real_dt*1000:.1f}ms, "
                          f"dt误差: {dt_error:.1f}%, "
                          f"处理: {avg_proc_ms:.1f}ms, "
                          f"sleep: {avg_sleep_ms:.1f}ms, "
                          f"超时率: {overrun_rate:.1f}%, "
                          f"相位: {phase:.3f}")
                else:
                    print(f"⏱️  步数: {self.current_step}/{self.config.total_steps}, "
                          f"运行时间: {motion_time:.2f}s, "
                          f"实际dt: {real_dt*1000:.1f}ms, "
                          f"dt误差: {dt_error:.1f}%, "
                          f"相位: {phase:.3f}")

            # 🕐 动态sleep控制 - 确保一致的策略执行频率
            self.adaptive_sleep("policy_execution")

            # 提前termination
            if should_terminate:
                break

        # 更新统计
        self.termination_stats['total_episodes'] += 1
        if episode_terminated_early:
            self.termination_stats['gravity_terminations'] += 1
        else:
            self.termination_stats['normal_completions'] += 1

        return "episode_complete"

    def save_trajectory_data(self):
        """保存轨迹数据 - 增强的时间分析"""
        if not self.motions_for_saving['motion_times']:
            print("📊 无轨迹数据需要保存")
            return

        # 转换为numpy数组
        result = {}
        for k in self.motions_for_saving:
            result[k] = np.array(self.motions_for_saving[k])
        
        # 计算时间统计
        real_times = result['motion_times']  # 从0开始的真实时间
        theoretical_times = result['theoretical_times']
        real_dts = result['real_dt']
        
        # 总运行时长（motion_times已经是从0开始）
        total_real_duration = real_times[-1] if len(real_times) > 0 else 0
        self.time_stats['total_real_time'] = total_real_duration
        self.time_stats['total_theoretical_time'] = theoretical_times[-1] if len(theoretical_times) > 0 else 0
        
        # 添加时间分析信息
        # FPS计算说明：
        # - fps (主要): 真实帧率 = 总帧数 / 真实总时长，反映数据的实际采集频率
        # - theoretical_fps: 理论帧率 = 1 / control_dt，基于配置的目标频率
        result['fps'] = len(real_times) / total_real_duration if total_real_duration > 0 else 0  # 主要FPS：真实帧率
        result['theoretical_fps'] = 1.0 / self.config.control_dt  # 理论帧率（参考用）
        result['time_stats'] = self.time_stats
        
        # 添加力矩限制统计
        result['torque_limit_stats'] = {
            'enabled': self.config.use_torque_limit,
            'scale': self.config.torque_limit_scale if self.config.use_torque_limit else None,
            'trigger_count': self.torque_limit_count if self.config.use_torque_limit else 0,
            'joint_limits': self.joint_torque_limits.tolist() if self.config.use_torque_limit and self.joint_torque_limits is not None else None,
            'scaled_limits': self.scaled_torque_limits.tolist() if self.config.use_torque_limit and self.scaled_torque_limits is not None else None
        }
        
        # 添加动态Sleep控制统计
        result['adaptive_sleep_stats'] = self.adaptive_sleep_stats.copy()

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_real_motion_trajectory_realtime.pkl"
        
        # 保存pickle文件
        with open(filename, 'wb') as f:
            pickle.dump(result, f)

        # 打印详细统计信息
        print(f"\n📊 轨迹数据保存完成: {filename}")
        print(f"📈 总帧数: {len(result['motion_times'])}")
        print(f"⏱️  真实总时长: {self.time_stats['total_real_time']:.2f}s")
        print(f"🕐 理论总时长: {self.time_stats['total_theoretical_time']:.2f}s")
        print(f"📊 时间差异: {abs(self.time_stats['total_real_time'] - self.time_stats['total_theoretical_time']):.3f}s")
        print(f"🎬 主要帧率(真实): {result['fps']:.1f}Hz")
        print(f"🕐 理论帧率: {result['theoretical_fps']:.1f}Hz")
        fps_diff = abs(result['fps'] - result['theoretical_fps'])
        fps_error_pct = fps_diff / result['theoretical_fps'] * 100 if result['theoretical_fps'] > 0 else 0
        print(f"📊 FPS误差: {fps_diff:.1f}Hz ({fps_error_pct:.1f}%)")
        print(f"⚡ 时间间隔统计:")
        print(f"  目标dt: {self.config.control_dt*1000:.1f}ms")
        print(f"  最大dt: {self.time_stats['max_dt']*1000:.1f}ms")
        print(f"  最小dt: {self.time_stats['min_dt']*1000:.1f}ms")
        print(f"  超时次数: {self.time_stats['dt_violations']}")
        print(f"🎯 Episode统计:")
        print(f"  总Episodes: {self.termination_stats['total_episodes']}")
        print(f"  正常完成: {self.termination_stats['normal_completions']}")
        print(f"  提前终止: {self.termination_stats['gravity_terminations']}")
        if self.termination_stats['total_episodes'] > 0:
            rate = self.termination_stats['gravity_terminations'] / self.termination_stats['total_episodes'] * 100
            print(f"  提前终止率: {rate:.1f}%")
        
        # 力矩限制统计
        if self.config.use_torque_limit:
            print(f"🔧 力矩限制统计:")
            print(f"  力矩限制启用: ✅")
            print(f"  缩放比例: {self.config.torque_limit_scale:.2f}")
            print(f"  超限次数: {self.torque_limit_count}")
            if self.torque_limit_count > 0:
                print(f"  策略行为: 检测到超限后立即退出")
            if len(result['motion_times']) > 0:
                trigger_rate = self.torque_limit_count / len(result['motion_times']) * 100
                print(f"  超限率: {trigger_rate:.2f}%")
        else:
            print(f"🔧 力矩限制统计:")
            print(f"  力矩限制启用: ❌")
        
        # 动态Sleep控制统计
        print(f"🔄 动态Sleep控制统计:")
        if self.adaptive_sleep_stats['total_adaptive_sleeps'] > 0:
            avg_proc_ms = self.adaptive_sleep_stats['avg_processing_time'] * 1000
            avg_sleep_ms = self.adaptive_sleep_stats['avg_sleep_time'] * 1000
            max_proc_ms = self.adaptive_sleep_stats['max_processing_time'] * 1000
            min_sleep_ms = self.adaptive_sleep_stats['min_sleep_time'] * 1000
            overrun_rate = self.adaptive_sleep_stats['total_overruns'] / self.adaptive_sleep_stats['total_adaptive_sleeps'] * 100
            cpu_usage = self.adaptive_sleep_stats['avg_processing_time'] / self.config.control_dt * 100
            print(f"  ✅ 动态控制启用: {self.adaptive_sleep_stats['total_adaptive_sleeps']}个周期")
            print(f"  ⏱️  平均处理时间: {avg_proc_ms:.1f}ms")
            print(f"  💤 平均sleep时间: {avg_sleep_ms:.1f}ms")
            print(f"  📈 最大处理时间: {max_proc_ms:.1f}ms")
            print(f"  📉 最小sleep时间: {min_sleep_ms:.1f}ms")
            print(f"  ⚠️  处理超时次数: {self.adaptive_sleep_stats['total_overruns']}")
            print(f"  📊 处理超时率: {overrun_rate:.1f}%")
            print(f"  🔧 平均CPU使用率: {cpu_usage:.1f}%")
            target_cycle_ms = self.config.control_dt * 1000
            actual_cycle_ms = avg_proc_ms + avg_sleep_ms
            cycle_accuracy = (1 - abs(actual_cycle_ms - target_cycle_ms) / target_cycle_ms) * 100
            print(f"  🎯 目标周期: {target_cycle_ms:.1f}ms")
            print(f"  ⏰ 实际周期: {actual_cycle_ms:.1f}ms")
            print(f"  ✅ 周期精度: {cycle_accuracy:.1f}%")
        else:
            print(f"  ❌ 无动态sleep统计数据")

    def run_trajectory_collection(self):
        """运行轨迹采集主循环"""
        print("🎬 开始轨迹采集...")
        print(f"📊 目标总步数: {self.config.total_steps}")
        print(f"📈 每个episode步数: {self.config.episode_steps}")
        print("🛑 按Select键随时退出")
        
        # ⏰ 开始时间记录
        self.start_trajectory_timing()

        while self.current_step < self.config.total_steps:
            result = self.run_episode()
            
            if result == "user_exit":
                break
            elif result == "torque_limit_exit":
                print("🚨 因力矩超限退出轨迹采集")
                break

        # 保存轨迹数据
        self.save_trajectory_data()


def main():
    parser = argparse.ArgumentParser(description='G1机器人真实世界轨迹采集 (真实时间版本)')
    parser.add_argument("net", type=str, help="网络接口 (例如: eth0)")
    parser.add_argument("config", type=str, help="配置文件名 (在configs文件夹中)", default="pbhc_real.yaml")
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(CURRENT_DIR, "configs", args.config)
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        exit(1)
        
    config = RealTrackConfig(config_path)

    # 初始化DDS通信
    ChannelFactoryInitialize(0, args.net)

    controller = RealTrackController(config)

    print("🚀 真实世界G1轨迹采集系统启动 (真实时间版本)")
    print(f"📊 观测维度: {config.num_obs}, 动作维度: {config.num_actions}")
    print(f"🔄 运动周期: {config.cycle_time}秒")
    print("🦾 PBHC版本: 23DOF全身控制，无手腕")
    print("📋 数据格式: 与mujoco_track.py完全一致")
    print("💾 轨迹数据将保存为pickle文件")
    print("⏰ ✅ motion_time从0开始，基于真实时间间隔累计")
    print("📊 ✅ 提供时间间隔分析和性能统计")
    print(f"🕐 ✅ 动态Sleep控制已启用 (目标频率: {1/config.control_dt:.0f}Hz)")
    print("   确保控制频率稳定和数据采集间隔一致")
    if config.use_torque_limit:
        print(f"🔧 ✅ 关节力矩限制已启用 (缩放比例: {config.torque_limit_scale:.2f})")
        print("   策略: 超限时将tau置0并立即退出策略")
    else:
        print("🔧 ❌ 关节力矩限制未启用")
    print("⚠️  使用Ctrl+C或Select键安全退出并保存数据")
    print("✅ 已修复pose_aa计算中的轴向量错误")
    print("🎮 控制说明：Start开始 → A键采集 → Select键随时退出")

    try:
        # 准备阶段
        controller.zero_torque_state()
        controller.move_to_default_pos()
        
        # 等待A按钮开始轨迹采集
        controller.default_pos_state()
        
        # 开始轨迹采集
        controller.run_trajectory_collection()
                
    except KeyboardInterrupt:
        print("\n⚠️  检测到Ctrl+C")
        controller.save_trajectory_data()
    
    # 进入阻尼状态
    print("🛑 进入阻尼状态...")
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("🏁 轨迹采集完成")


if __name__ == "__main__":
    main() 