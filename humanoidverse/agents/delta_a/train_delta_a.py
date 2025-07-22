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


# 训练delta action模型，policy_checkpoint是预训练模型。
class PPODeltaA(MHPPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        super().__init__(env, config, log_dir, device)
        # 统计数据缓冲区
        self._pos_stats_buffer = []
        self._pos_min_buffer = []  # 记录最小值
        self._pos_max_buffer = []  # 记录最大值
        self._lin_vel_stats_buffer = []  # 新增：线性速度统计
        self._lin_vel_min_buffer = []  # 新增：线性速度最小值
        self._lin_vel_max_buffer = []  # 新增：线性速度最大值
        self.stats_collection_interval = 10000  # 默认每10000步打印一次统计数据
        self.stats_collection_counter = 0



                # ----------------- UNCOMMENT THIS FOR ANALYTIC SEARCH FOR OPTIMAL ACTION BASED ON DELTA_A -----------------
                # if not hasattr(env, 'loaded_extra_policy'):
                #     setattr(env, 'loaded_extra_policy', self.loaded_policy)
                # if not hasattr(env.loaded_extra_policy, 'eval_policy'):
                #     setattr(env.loaded_extra_policy, 'eval_policy', self.loaded_policy._get_inference_policy())

                # ----------------- UNCOMMENT THIS FOR ANALYTIC SEARCH FOR OPTIMAL ACTION BASED ON DELTA_A -----------------

    # def _actor_act_step(self, obs_dict):
    #     actions = self.actor.act(obs_dict["actor_obs"])
    #     return self.actor.act_inference(obs_dict["actor_obs"])

    '''def _actor_rollout_step(self, obs_dict, policy_state_dict):
        # 获取pkl_actions
        pkl_actions=self.env._motion_lib.get_motion_actions(self.env.motion_ids, self.env._motion_times)
        obs_dict['actor_obs'] = torch.cat([
            obs_dict['actor_obs'][:, :-self.env.dim_actions],  # 除了ref_actions之外的所有观测
            pkl_actions  # 这里到底是pkl_actions还是ref_actions?
        ], dim=1)
        # 策略网络输出delta_actions
        delta_actions = self._actor_act_step(obs_dict)

        actions = pkl_actions + delta_actions
        policy_state_dict["actions"] = actions
        action_mean = self.actor.action_mean.detach()
        action_sigma = self.actor.action_std.detach()
        actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
        policy_state_dict["action_mean"] = action_mean
        policy_state_dict["action_sigma"] = action_sigma
        policy_state_dict["actions_log_prob"] = actions_log_prob

        assert len(actions.shape) == 2
        assert len(actions_log_prob.shape) == 2
        assert len(action_mean.shape) == 2
        assert len(action_sigma.shape) == 2

        return policy_state_dict'''

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1
        self._post_evaluate_policy()

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # 使用loaded policy生成动作作为ref_actions
                pkl_actions = self.env._motion_lib.get_motion_actions(self.env.motion_ids, self.env._motion_times)


                last_actor_slice = obs_dict['actor_obs'][:, -self.env.dim_actions:]
                if not torch.allclose(last_actor_slice, torch.zeros_like(last_actor_slice)):
                    print("Warning: 观测错误")

                # 检查critic_obs最后一段
                last_critic_slice = obs_dict['critic_obs'][:, -self.env.dim_actions:]
                if not torch.allclose(last_critic_slice, torch.zeros_like(last_critic_slice)):
                    print("Warning: 观测错误")
                obs_dict['actor_obs'] = torch.cat([
                    obs_dict['actor_obs'][:, :-self.env.dim_actions],  # 除了ref_actions之外的所有观测
                    pkl_actions  # 这里到底是pkl_actions还是ref_actions?
                ], dim=1)

                obs_dict['critic_obs'] = torch.cat([
                    obs_dict['critic_obs'][:, :-self.env.dim_actions],  # 除了ref_actions之外的所有观测
                    pkl_actions
                ], dim=1)

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                delta_actions = policy_state_dict["actions"]
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values
                # final_actions = delta_actions + pkl_actions  # 直接计算最终动作
                final_actions = delta_actions + pkl_actions

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])

                    # delta action

                '''if i % 10 == 0:
                    base_ang_vel = obs_dict['actor_obs'][:, 0:3]
                    projected_gravity = obs_dict['actor_obs'][:, 3:6]
                    dof_pos = obs_dict['actor_obs'][:, 6:29]
                    dof_vel = obs_dict['actor_obs'][:, 29:52]
                    actions = obs_dict['actor_obs'][:, 52:75]
                    ref_motion_phase = obs_dict['actor_obs'][:, 75:76]
                    history_actor =obs_dict['actor_obs'][:, 76:]

                    print(
                        f"[Step {i}] base_ang_vel mean: {base_ang_vel.mean().item():.4f}, std: {base_ang_vel.std().item():.4f}, min: {base_ang_vel.min().item():.4f}, max: {base_ang_vel.max().item():.4f}")
                    print(
                        f"[Step {i}] projected_gravity mean: {projected_gravity.mean().item():.4f}, std: {projected_gravity.std().item():.4f}, min: {projected_gravity.min().item():.4f}, max: {projected_gravity.max().item():.4f}")
                    print(
                        f"[Step {i}] pkl_action mean: {pkl_actions.mean().item():.4f}, std: {pkl_actions.std().item():.4f}, min: {pkl_actions.min().item():.4f}, max: {pkl_actions.max().item():.4f}")
                    print(
                        f"[Step {i}] final_action mean: {final_actions.mean().item():.4f}, std: {final_actions.std().item():.4f}, min: {final_actions.min().item():.4f}, max: {final_actions.max().item():.4f}")
                    print(
                        f"actions mean: {actions.mean().item():.4f}, std: {actions.std().item():.4f}, min: {actions.min().item():.4f}, max: {actions.max().item():.4f}")'''

                actor_state = {
                    "actions": final_actions,  # 使用最终动作
                    "delta_actions": delta_actions,
                }

                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                # rewards_stored = rewards.clone().unsqueeze(1)
                rewards_stored = rewards.clone().reshape(self.env.num_envs, self.env.num_rew_fn)
                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                '''if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0'''
                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    # self.cur_reward_sum += rewards.sum(dim=1)  # Sum rewards across reward functions
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
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        actions_closed_loop = self.loaded_policy.eval_policy(actor_state['obs']['closed_loop_actor_obs']).detach()

        actor_state.update({"actions": actions, "actions_closed_loop": actions_closed_loop})
        # actor_state.update({"actions": actions, "actions_closed_loop": actions_closed_loop, "current_closed_loop_actor_obs": actor_state['obs']['closed_loop_actor_obs']})
        # print("updated closed loop actor obs: ", actor_state['current_closed_loop_actor_obs'])
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state



