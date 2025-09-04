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

## 5. 从 TensorBoard 读取 sigma 并回写 YAML（微调前执行）

- 作用：从预训练产生的 TensorBoard 日志中读取自适应 sigma（`adp_sigma_*`），按倍率乘法后写回到 `rewards.reward_tracking_sigma`，用于在微调阶段对齐噪声尺度。
- 何时运行：完成“预训练（第1步）”后、开始“微调”前运行一次。

示例（将 `--log_dir` 替换为你的预训练日志目录）：
```bash
python read_sigma.py \
  --log_dir /home/harry/Desktop/asap-pbhc/logs/MotionTracking/round_sick \
  --yaml humanoidverse/config/rewards/motion_tracking/final_training.yaml \
  --multiplier 2 --no-backup
```
参数说明：
- `--log_dir`：预训练运行生成的 TensorBoard 日志目录（一般位于 `logs/MotionTracking/...`）。
- `--yaml`：目标 YAML 文件路径；脚本仅会更新其中的 `reward_tracking_sigma` 键。
- `--multiplier`：对读取到的 sigma 进行倍率缩放（默认 2.0）。
- `--no-backup`：不生成 `.bak` 备份；若不加该参数，会自动在同目录生成备份。

注意：脚本不会改动其它配置或注释；可重复运行以更新数值。

## 6. 微调（残差微调：23维主策略 + 冻结的4维脚踝残差）

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


