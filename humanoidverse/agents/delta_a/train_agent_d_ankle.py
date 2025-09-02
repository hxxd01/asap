import torch
import torch.nn as nn
import torch.optim as optimim
import numpy as np
from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
import hydra
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from hydra.utils import instantiate
from loguru import logger
from rich.progress import track
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from humanoidverse.agents.mh_ppo.mh_ppo import MHPPO

# 注释掉OnlineMeanCovBatch导入 - 与action_stats.npz相关，当前训练中不使用
# from humanoidverse.envs.legged_base_task.cov import OnlineMeanCovBatch
console = Console()

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask
from pathlib import Path
from omegaconf import OmegaConf
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from hydra.utils import instantiate


# 固定4维脚踝delta action模型，训练策略
class PPODeltaDAnkle(MHPPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)

        # 脚踝关节索引
        self.ankle_indices = [4, 5, 10, 11]  # left_ankle_pitch, left_ankle_roll, right_ankle_pitch, right_ankle_roll

        self.policy_checkpoint = config.policy_checkpoint if hasattr(config, 'policy_checkpoint') else None

        # 注释掉动作统计相关代码 - 当前训练中不需要使用action_stats.npz文件
        # stats = OnlineMeanCovBatch(dim=23)
        # stats.load("action_stats.npz")
        # self.mean = stats.get_mean()
        # self.cov = stats.get_cov()
        # self.inv_cov = np.linalg.inv(self.cov + 1e-6 * np.eye(self.cov.shape[0]))

        if self.policy_checkpoint is not None:
            has_config = True
            checkpoint = Path(self.policy_checkpoint)
            config_path = checkpoint.parent / "config.yaml"
            if not config_path.exists():
                config_path = checkpoint.parent.parent / "config.yaml"
                if not config_path.exists():
                    has_config = False
                    logger.error(f"Could not find config path: {config_path}")

            if has_config:
                logger.info(f"Loading training config file from {config_path}")
                with open(config_path) as file:
                    policy_config = OmegaConf.load(file)

                if policy_config.eval_overrides is not None:
                    policy_config = OmegaConf.merge(
                        policy_config, policy_config.eval_overrides
                    )

                pre_process_config(policy_config)

                self.loaded_policy: BaseAlgo = instantiate(policy_config.algo, env=self.env, device=self.device, log_dir=None)
                self.loaded_policy.algo_obs_dim_dict = policy_config.env.config.robot.algo_obs_dim_dict
                self.loaded_policy.setup()
                self.loaded_policy.load(self.policy_checkpoint, load_critic=False)
                self.loaded_policy._eval_mode()
                self.loaded_policy.eval_policy = self.loaded_policy._get_inference_policy()

                # 设置环境的loaded_policy
                if isinstance(self.env, LeggedRobotMotionTracking):
                    self.env.loaded_policy = self.loaded_policy

                for name, param in self.loaded_policy.actor.actor_module.module.named_parameters():
                    param.requires_grad = False

                logger.info(f"Ankle Delta Fine-tuning: Using 4D ankle delta action model")
                logger.info(f"Ankle joints: {self.ankle_indices}")

        self.delta_action_dim_history = []  # 记录每步每个维度的均值

        # 残差动作统计（经过scale后的）- 存储每步的正负平均值
        self.ankle_delta_stats = {
            'left_pitch_positive_avgs': [],  # 存储每步正值的平均值
            'left_pitch_negative_avgs': [],  # 存储每步负值的平均值
            'left_roll_positive_avgs': [],
            'left_roll_negative_avgs': [],
            'right_pitch_positive_avgs': [],
            'right_pitch_negative_avgs': [],
            'right_roll_positive_avgs': [],
            'right_roll_negative_avgs': []
        }

        # clip后实际作用的残差动作统计
        self.ankle_delta_clipped_stats = {
            'left_pitch_positive_avgs': [],  # clip后存储每步正值的平均值
            'left_pitch_negative_avgs': [],  # clip后存储每步负值的平均值
            'left_roll_positive_avgs': [],
            'left_roll_negative_avgs': [],
            'right_pitch_positive_avgs': [],
            'right_pitch_negative_avgs': [],
            'right_roll_positive_avgs': [],
            'right_roll_negative_avgs': []
        }

    def _expand_ankle_to_full(self, ankle_actions_4d):
        """将4维脚踝动作扩展为23维全身动作"""
        batch_size = ankle_actions_4d.shape[0]
        full_actions = torch.zeros(batch_size, 23, device=self.device)
        full_actions[:, self.ankle_indices] = ankle_actions_4d
        return full_actions

    # 注释掉马氏距离函数 - 与action_stats.npz相关，当前训练中不使用
    # def mahalanobis(self,x, mean, inv_cov):
    #     # x: (batch, dim)
    #     diff = x - mean  # (batch, dim)
    #     # einsum更高效
    #     return np.sqrt(np.einsum('bi,ij,bj->b', diff, inv_cov, diff))

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # 当前策略推理
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values
                actions = policy_state_dict["actions"]  # 当前策略的actions

                # 用当前策略的actions构建delta模型的输入
                obs_dict['closed_loop_actor_obs'] = torch.cat([
                    obs_dict['closed_loop_actor_obs'][:, :-self.env.dim_actions],
                    actions  # 用当前策略的actions构建残差模型的输入
                ], dim=1)

                # 关键修改：delta模型输出4维，扩展为23维
                ankle_delta_4d = self.loaded_policy.eval_policy(obs_dict['closed_loop_actor_obs']).detach()  # 4维输出
                delta_actions = self._expand_ankle_to_full(ankle_delta_4d)  # 扩展为23维

                # 收集原始残差动作统计（clip之前的原始值）
                action_scale = self.env.config.robot.control.action_scale
                scaled_ankle_delta_raw = ankle_delta_4d * action_scale  # 4维脚踝残差的真实调整量（原始值）

                # 计算每步的正负平均值 (num_envs个环境的数据) - 原始值统计
                scaled_ankle_numpy_raw = scaled_ankle_delta_raw.cpu().numpy()  # (num_envs, 4)

                for joint_idx, joint_name in enumerate(['left_pitch', 'left_roll', 'right_pitch', 'right_roll']):
                    joint_values = scaled_ankle_numpy_raw[:, joint_idx]  # 当前关节的num_envs个值

                    # 分离正值和负值
                    positive_values = joint_values[joint_values > 0]
                    negative_values = joint_values[joint_values < 0]

                    # 计算正值平均值（如果有正值的话）
                    if len(positive_values) > 0:
                        pos_avg = positive_values.mean()
                        self.ankle_delta_stats[f'{joint_name}_positive_avgs'].append(pos_avg)

                    # 计算负值平均值（如果有负值的话）
                    if len(negative_values) > 0:
                        neg_avg = negative_values.mean()
                        self.ankle_delta_stats[f'{joint_name}_negative_avgs'].append(neg_avg)

                # 微调阶段约束逻辑：使用更小的范围确保稳定性
                min_val = -0.3
                max_val = 0.3
                mask = (delta_actions >= min_val) & (delta_actions <= max_val)
                delta_actions = delta_actions * mask.float()

                # 收集clip后实际作用的残差动作统计
                clipped_ankle_delta_4d = delta_actions[:, self.ankle_indices]  # 提取脚踝部分 (num_envs, 4)
                scaled_ankle_delta_clipped = clipped_ankle_delta_4d * action_scale  # 4维脚踝残差的真实调整量（clip后）

                # 计算每步的正负平均值 (num_envs个环境的数据) - clip后统计
                scaled_ankle_numpy_clipped = scaled_ankle_delta_clipped.cpu().numpy()  # (num_envs, 4)

                for joint_idx, joint_name in enumerate(['left_pitch', 'left_roll', 'right_pitch', 'right_roll']):
                    joint_values = scaled_ankle_numpy_clipped[:, joint_idx]  # 当前关节的num_envs个值

                    # 分离正值和负值
                    positive_values = joint_values[joint_values > 0]
                    negative_values = joint_values[joint_values < 0]

                    # 计算正值平均值（如果有正值的话）
                    if len(positive_values) > 0:
                        pos_avg = positive_values.mean()
                        self.ankle_delta_clipped_stats[f'{joint_name}_positive_avgs'].append(pos_avg)

                    # 计算负值平均值（如果有负值的话）
                    if len(negative_values) > 0:
                        neg_avg = negative_values.mean()
                        self.ankle_delta_clipped_stats[f'{joint_name}_negative_avgs'].append(neg_avg)

                # 记录每步每个维度的均值
                delta_action_dim_mean = delta_actions.mean(dim=0).cpu().numpy()

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                # 最终动作 = 策略动作 + 脚踝delta
                final_actions = actions + delta_actions

                actor_state = {
                    "actions": final_actions,  # 使用相加后的动作
                    "delta_actions": delta_actions,
                }

                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().reshape(self.env.num_envs, self.env.num_rew_fn)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards.view(self.env.num_envs, self.env.num_rew_fn).sum(dim=-1)
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            # prepare data for training
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'),
                                       dones=self.storage.query_key('dones'),
                                       rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def generate_ankle_delta_log(self, pad=40, iteration=None):
        """生成脚踝残差动作统计日志并记录到TensorBoard - 基于正负平均值统计"""
        import numpy as np

        if not any(self.ankle_delta_stats.values()):
            return ""

        ankle_log_string = ""
        joint_names = ['Left Pitch', 'Left Roll', 'Right Pitch', 'Right Roll']
        stat_keys = ['left_pitch', 'left_roll', 'right_pitch', 'right_roll']

        for name, key in zip(joint_names, stat_keys):
            # 获取正值平均值数组和负值平均值数组
            pos_avgs = self.ankle_delta_stats[f'{key}_positive_avgs']
            neg_avgs = self.ankle_delta_stats[f'{key}_negative_avgs']

            # 处理正值平均值统计
            if pos_avgs:
                pos_data = np.array(pos_avgs)
                pos_max = pos_data.max()
                pos_min = pos_data.min()
                pos_mean = pos_data.mean()

                ankle_log_string += f"{f'{name} - Positive Averages Max:':>{pad}} {pos_max:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Positive Averages Min:':>{pad}} {pos_min:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Positive Averages Mean:':>{pad}} {pos_mean:>10.4f}\n"

                # 记录到TensorBoard
                if iteration is not None and hasattr(self, 'writer'):
                    self.writer.add_scalar(f'AnkleDelta/{name}_Positive_Max', pos_max, iteration)
                    self.writer.add_scalar(f'AnkleDelta/{name}_Positive_Min', pos_min, iteration)
                    self.writer.add_scalar(f'AnkleDelta/{name}_Positive_Mean', pos_mean, iteration)
                    # 记录正值平均值分布直方图
                    self.writer.add_histogram(f'AnkleDelta/{name}_Positive_Distribution', pos_data, iteration)

            # 处理负值平均值统计
            if neg_avgs:
                neg_data = np.array(neg_avgs)
                neg_max = neg_data.max()  # 负值中的最大值（最接近0的）
                neg_min = neg_data.min()  # 负值中的最小值（最负的）
                neg_mean = neg_data.mean()

                ankle_log_string += f"{f'{name} - Negative Averages Max:':>{pad}} {neg_max:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Negative Averages Min:':>{pad}} {neg_min:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Negative Averages Mean:':>{pad}} {neg_mean:>10.4f}\n"

                # 记录到TensorBoard
                if iteration is not None and hasattr(self, 'writer'):
                    self.writer.add_scalar(f'AnkleDelta/{name}_Negative_Max', neg_max, iteration)
                    self.writer.add_scalar(f'AnkleDelta/{name}_Negative_Min', neg_min, iteration)
                    self.writer.add_scalar(f'AnkleDelta/{name}_Negative_Mean', neg_mean, iteration)
                    # 记录负值平均值分布直方图
                    self.writer.add_histogram(f'AnkleDelta/{name}_Negative_Distribution', neg_data, iteration)

            # 添加分隔线
            ankle_log_string += "\n"

        # 清空统计数据，为下次收集做准备
        for key in self.ankle_delta_stats:
            self.ankle_delta_stats[key].clear()

        return ankle_log_string

    def generate_ankle_delta_clipped_log(self, pad=40, iteration=None):
        """生成clip后脚踝残差动作统计日志并记录到TensorBoard - 基于正负平均值统计"""
        import numpy as np

        if not any(self.ankle_delta_clipped_stats.values()):
            return ""

        ankle_log_string = ""
        joint_names = ['Left Pitch', 'Left Roll', 'Right Pitch', 'Right Roll']
        stat_keys = ['left_pitch', 'left_roll', 'right_pitch', 'right_roll']

        for name, key in zip(joint_names, stat_keys):
            # 获取正值平均值数组和负值平均值数组
            pos_avgs = self.ankle_delta_clipped_stats[f'{key}_positive_avgs']
            neg_avgs = self.ankle_delta_clipped_stats[f'{key}_negative_avgs']

            # 处理正值平均值统计
            if pos_avgs:
                pos_data = np.array(pos_avgs)
                pos_max = pos_data.max()
                pos_min = pos_data.min()
                pos_mean = pos_data.mean()

                ankle_log_string += f"{f'{name} - Positive Averages Max:':>{pad}} {pos_max:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Positive Averages Min:':>{pad}} {pos_min:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Positive Averages Mean:':>{pad}} {pos_mean:>10.4f}\n"

                # 记录到TensorBoard
                if iteration is not None and hasattr(self, 'writer'):
                    self.writer.add_scalar(f'AnkleDeltaClipped/{name}_Positive_Max', pos_max, iteration)
                    self.writer.add_scalar(f'AnkleDeltaClipped/{name}_Positive_Min', pos_min, iteration)
                    self.writer.add_scalar(f'AnkleDeltaClipped/{name}_Positive_Mean', pos_mean, iteration)
                    # 记录正值平均值分布直方图
                    self.writer.add_histogram(f'AnkleDeltaClipped/{name}_Positive_Distribution', pos_data, iteration)

            # 处理负值平均值统计
            if neg_avgs:
                neg_data = np.array(neg_avgs)
                neg_max = neg_data.max()  # 负值中的最大值（最接近0的）
                neg_min = neg_data.min()  # 负值中的最小值（最负的）
                neg_mean = neg_data.mean()

                ankle_log_string += f"{f'{name} - Negative Averages Max:':>{pad}} {neg_max:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Negative Averages Min:':>{pad}} {neg_min:>10.4f}\n"
                ankle_log_string += f"{f'{name} - Negative Averages Mean:':>{pad}} {neg_mean:>10.4f}\n"

                # 记录到TensorBoard
                if iteration is not None and hasattr(self, 'writer'):
                    self.writer.add_scalar(f'AnkleDeltaClipped/{name}_Negative_Max', neg_max, iteration)
                    self.writer.add_scalar(f'AnkleDeltaClipped/{name}_Negative_Min', neg_min, iteration)
                    self.writer.add_scalar(f'AnkleDeltaClipped/{name}_Negative_Mean', neg_mean, iteration)
                    # 记录负值平均值分布直方图
                    self.writer.add_histogram(f'AnkleDeltaClipped/{name}_Negative_Distribution', neg_data, iteration)

            # 添加分隔线
            ankle_log_string += "\n"

        # 清空统计数据，为下次收集做准备
        for key in self.ankle_delta_clipped_stats:
            self.ankle_delta_clipped_stats[key].clear()

        return ankle_log_string

    def _post_epoch_logging(self, log_dict, width=80, pad=40):
        """重写日志方法，添加残差动作统计"""
        # 调用父类的日志方法
        super()._post_epoch_logging(log_dict, width, pad)

        # 只在指定间隔打印
        if log_dict['it'] % self.logging_interval != 0:
            return

        # 生成原始残差动作日志（clip之前）
        ankle_delta_log = self.generate_ankle_delta_log(pad, iteration=log_dict['it'])

        if ankle_delta_log:
            print("\n" + "=" * width)
            print(" Ankle Delta Action Statistics - Raw Output (rad) ".center(width, '='))
            print("=" * width)
            print(ankle_delta_log, end="")
            print("=" * width + "\n")

        # 生成clip后残差动作日志（实际作用的）
        ankle_delta_clipped_log = self.generate_ankle_delta_clipped_log(pad, iteration=log_dict['it'])

        if ankle_delta_clipped_log:
            print("\n" + "=" * width)
            print(" Ankle Delta Action Statistics - Clipped Output (rad) ".center(width, '='))
            print("=" * width)
            print(ankle_delta_clipped_log, end="")
            print("=" * width + "\n")

    def _pre_eval_env_step(self, actor_state: dict):
        # 当前策略推理
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])

        # 用当前策略动作构建delta模型输入
        obs_with_actions = torch.cat([
            actor_state["obs"]['closed_loop_actor_obs'][:, :-self.env.dim_actions],
            actions
        ], dim=1)

        # delta模型推理：4维输出扩展为23维
        ankle_delta_4d = self.loaded_policy.eval_policy(obs_with_actions).detach()
        delta_actions = self._expand_ankle_to_full(ankle_delta_4d)

        # 最终动作
        final_actions = actions + delta_actions

        actor_state.update({"actions": final_actions, "actions_closed_loop": actions})

        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state