import yaml

# 1. 关节顺序（dof_names），建议直接从mujoco_config.yaml或robot配置文件复制
dof_names = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
]

# 2. stiffness/damping参数（从config.yaml复制）
stiffness = {
    "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150,
    "ankle_pitch": 40, "ankle_roll": 40, "waist_yaw": 400, "waist_roll": 400, "waist_pitch": 400,
    "shoulder_pitch": 100, "shoulder_roll": 100, "shoulder_yaw": 50, "elbow": 50
}
damping = {
    "hip_yaw": 2.0, "hip_roll": 2.0, "hip_pitch": 2.0, "knee": 4.0,
    "ankle_pitch": 2.0, "ankle_roll": 2.0, "waist_yaw": 5.0, "waist_roll": 5.0, "waist_pitch": 5.0,
    "shoulder_pitch": 2.0, "shoulder_roll": 2.0, "shoulder_yaw": 2.0, "elbow": 2.0
}

# 3. 关节名到参数名的映射
joint2param = {
    "left_hip_pitch_joint": "hip_pitch",
    "left_hip_roll_joint": "hip_roll",
    "left_hip_yaw_joint": "hip_yaw",
    "left_knee_joint": "knee",
    "left_ankle_pitch_joint": "ankle_pitch",
    "left_ankle_roll_joint": "ankle_roll",
    "right_hip_pitch_joint": "hip_pitch",
    "right_hip_roll_joint": "hip_roll",
    "right_hip_yaw_joint": "hip_yaw",
    "right_knee_joint": "knee",
    "right_ankle_pitch_joint": "ankle_pitch",
    "right_ankle_roll_joint": "ankle_roll",
    "waist_yaw_joint": "waist_yaw",
    "waist_roll_joint": "waist_roll",
    "waist_pitch_joint": "waist_pitch",
    "left_shoulder_pitch_joint": "shoulder_pitch",
    "left_shoulder_roll_joint": "shoulder_roll",
    "left_shoulder_yaw_joint": "shoulder_yaw",
    "left_elbow_joint": "elbow",
    "right_shoulder_pitch_joint": "shoulder_pitch",
    "right_shoulder_roll_joint": "shoulder_roll",
    "right_shoulder_yaw_joint": "shoulder_yaw",
    "right_elbow_joint": "elbow"
}

# 4. 自动生成kps/kds
kps = [stiffness[joint2param[name]] for name in dof_names]
kds = [damping[joint2param[name]] for name in dof_names]

print("kps:", kps)
print("kds:", kds)







