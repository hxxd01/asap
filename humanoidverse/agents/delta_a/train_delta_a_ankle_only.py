import torch
import torch.nn as nn
import torch.optim as optim

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

console = Console()
from humanoidverse.agents.mh_ppo.mh_ppo import MHPPO
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.envs.base_task.base_task import BaseTask
from pathlib import Path
from omegaconf import OmegaConf
from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.utils.helpers import pre_process_config
from humanoidverse.agents.base_algo.base_algo import BaseAlgo
from hydra.utils import instantiate

# 训练delta action模型，只调脚踝关节版本 - 基于原始版本简单修改
class PPODeltaAAnkleOnly(MHPPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        
        # 脚踝关节索引
        self.ankle_indices = [4, 5, 10, 11]  # left_ankle_pitch, left_ankle_roll, right_ankle_pitch, right_ankle_roll
        
        # 修改网络输出维度为4维
        original_actions_dim = env.config.robot.actions_dim
        env.config.robot.actions_dim = 4  # 临时设为4维
        
        super().__init__(env, config, log_dir, device)
        
        # 恢复原始维度
        env.config.robot.actions_dim = original_actions_dim
        
        # 统计数据缓冲区
        self._pos_stats_buffer = []
        self._pos_min_buffer = []
        self._pos_max_buffer = []
        self._lin_vel_stats_buffer = []
        self._lin_vel_min_buffer = []
        self._lin_vel_max_buffer = []
        self.stats_collection_interval = 10000
        self.stats_collection_counter = 0
      
        '''if config.policy_checkpoint is not None:
            has_config = True
            checkpoint = Path(config.policy_checkpoint)
            
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

                policy_config.algo.config.policy_checkpoint = str(checkpoint)
                logger.info(f"Using checkpoint: {checkpoint}")

                pre_process_config(policy_config)

                # 直接使用PPO类创建loaded_policy
                self.loaded_policy = PPO(env=env, config=policy_config.algo.config, device=device, log_dir=None)
                self.loaded_policy.algo_obs_dim_dict = policy_config.env.config.robot.algo_obs_dim_dict
                self.loaded_policy.setup()
                self.loaded_policy.load(str(checkpoint))  
                self.loaded_policy._eval_mode()
                self.loaded_policy.eval_policy = self.loaded_policy._get_inference_policy()

                # 设置环境的loaded_policy
                if isinstance(self.env, LeggedRobotMotionTracking):
                    self.env.loaded_policy = self.loaded_policy

                for name, param in self.loaded_policy.actor.actor_module.module.named_parameters():
                    param.requires_grad = False
                    
                logger.info(f"Ankle-only Delta Action: Network outputs 4D ankle delta actions")
                logger.info(f"Ankle joints: {self.ankle_indices}")'''

    def _expand_ankle_to_full(self, ankle_actions_4d):
        """将4维脚踝动作扩展为23维全身动作"""
        batch_size = ankle_actions_4d.shape[0]
        full_actions = torch.zeros(batch_size, 23, device=self.device)
        full_actions[:, self.ankle_indices] = ankle_actions_4d
        return full_actions

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # 使用loaded policy生成动作作为ref_actions
                pkl_actions = self.env._motion_lib.get_motion_actions(self.env.motion_ids, self.env._motion_times)

                ref_actions = self.loaded_policy.eval_policy(obs_dict['closed_loop_actor_obs']).detach()
                last_actor_slice = obs_dict['actor_obs'][:, -self.env.dim_actions:]
                if not torch.allclose(last_actor_slice, torch.zeros_like(last_actor_slice)):
                    print("Warning: 观测错误")

                # 检查critic_obs最后一段
                last_critic_slice = obs_dict['critic_obs'][:, -self.env.dim_actions:]
                if not torch.allclose(last_critic_slice, torch.zeros_like(last_critic_slice)):
                    print("Warning: 观测错误")
                    
                obs_dict['actor_obs'] = torch.cat([
                    obs_dict['actor_obs'][:, :-self.env.dim_actions],  # 除了ref_actions之外的所有观测
                    pkl_actions  # 这里使用pkl_actions
                ], dim=1)
                
                obs_dict['critic_obs'] = torch.cat([
                    obs_dict['critic_obs'][:, :-self.env.dim_actions],  # 除了ref_actions之外的所有观测
                    pkl_actions
                ], dim=1)
                
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                
                # 关键修改：网络输出4维，扩展为23维
                ankle_delta_4d = policy_state_dict["actions"]  # 4维输出
                delta_actions = self._expand_ankle_to_full(ankle_delta_4d)  # 扩展为23维
                
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values
                
                # 原始逻辑：delta + pkl
                final_actions = delta_actions + pkl_actions

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                
                actor_state = {
                    "actions": final_actions,
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

    def _pre_eval_env_step(self, actor_state: dict):
        # 评估时的处理
        ankle_actions_4d = self.eval_policy(actor_state["obs"]['actor_obs'])  # 4维输出
        actions_closed_loop = self.loaded_policy.eval_policy(actor_state['obs']['closed_loop_actor_obs']).detach()

        # 扩展为23维并与pkl相加
        delta_actions = self._expand_ankle_to_full(ankle_actions_4d)
        pkl_actions = self.env._motion_lib.get_motion_actions(self.env.motion_ids, self.env._motion_times)
        final_actions = delta_actions + pkl_actions

        actor_state.update({"actions": final_actions, "actions_closed_loop": actions_closed_loop})
        
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state 