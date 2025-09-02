# MotionTracking 训练与微调指南

以下命令均在仓库根目录 `/home/harry/Desktop/asap-pbhc` 执行。

## 1. 预训练（23维整体动作策略）

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
project_name=MotionTracking num_envs=4096 \
+obs=motion_tracking/main \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main \
+rewards=motion_tracking/main \
experiment_name=pbhc_pretrain \
robot.motion.motion_file="example/motion_data/Roundhouse_kick.pkl" \
seed=1 \
+device=cuda:0
```

## 2. 评估并导出 ONNX

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/eval_agent.py \
+device=cuda:0 \
+headness=True \
+env.config.enforce_randomize_motion_start_eval=False \
+checkpoint="<你的模型路径>.pt"
```

## 3. 采轨迹（MuJoCo）

```bash
cd asap_mujoco_sim
python mujoco_track_with_processing.py
```

## 4. 残差模型训练（4维脚踝增量）

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=NO_domain_rand \
+rewards=motion_tracking/delta_action_rewards \
+robot=g1/g1_23dof_lock_wrist \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/delta_action_obs \
++rewards.reward_scales.termination=-100 \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=MotionTracking_CR7 \
robot.motion.motion_file="<采轨迹生成的pkl路径>" \
robot.motion.motion_lib_type=origin \
rewards.reward_penalty_curriculum=False \
rewards.reward_penalty_degree=0.00001 \
algo._target_=humanoidverse.agents.delta_a.train_delta_a_ankle_only.PPODeltaAAnkleOnly \
algo.config.num_steps_per_env=50 \
algo=ppo_delta_ankle \
env.config.resample_motion_when_training=True \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
env.config.termination.terminate_by_gravity=True \
robot.asset.self_collisions=0
```

## 5. 微调（残差微调：23维主策略 + 冻结的4维脚踝残差）

- `algo.config.policy_checkpoint`：4维脚踝残差模型（作为冻结基线）
- `checkpoint`：待微调的23维整体策略（加载到当前训练的 actor/critic）

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=main \
+rewards=motion_tracking/final_training \
+robot=g1/g1_23dof_lock_wrist \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/final_training \
+algo.config.policy_checkpoint="/home/harry/Desktop/asap-pbhc/logs/MotionTracking/20250901_164732-delta_action_training-motion_tracking-g1_23dof_lock_wrist/model_4000.pt" \
+checkpoint="/home/harry/Desktop/asap-pbhc/logs/MotionTracking/round_sick/model_30000.pt" \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=final_training \
robot.motion.motion_file=example/motion_data/Roundhouse_kick.pkl \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
algo=mh_ppo \
algo._target_=humanoidverse.agents.delta_a.train_agent_d_ankle.PPODeltaDAnkle \
algo.config.save_interval=100 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
env.config.termination.terminate_by_gravity=True \
robot.asset.self_collisions=0
```

### 重要说明
- 残差微调场景下：训练中的 actor 为 23 维整体策略；`policy_checkpoint` 的 4 维模型仅作为冻结“脚踝残差基线”。
- 若只想微调 4 维残差，则使用“残差模型训练”命令（不加载 23 维 checkpoint）。
- Hydra 覆盖：新增键用 `+key=...`，覆盖已存在键用 `key=...` 或 `++key=...`。
