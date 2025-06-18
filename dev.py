import joblib
import numpy as np

# 1. 读取原始pkl
data = joblib.load('/home/harry/Desktop/ASAP/logs/MotionTracking/20250317_215927-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/g2g800 nog19900pt1_.pkl')
motion = data['motion0']

# 2. 取出terminate信号
terminate = np.array(motion['terminate'])  # shape: (T,)
T = len(terminate)

# 3. 找到所有终止点
episode_ends = np.where(terminate)[0]
if len(episode_ends) == 0 or episode_ends[-1] != T-1:
    # 如果最后一帧不是终止帧，补上
    episode_ends = np.append(episode_ends, T-1)
episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

# 4. 切分每个episode
episodes = []
for start, end in zip(episode_starts, episode_ends):
    ep = {}
    for k in motion.keys():
        # 注意：有的key可能是一维（如fps），只保留原值
        if isinstance(motion[k], np.ndarray) and motion[k].shape[0] == T:
            ep[k] = motion[k][start:end+1]
        else:
            ep[k] = motion[k]
    episodes.append(ep)

print(f"共切分出{len(episodes)}条完整episode")

# 5. 保存为新的pkl文件（每个episode一个motion{i}，与原始格式兼容）
split_data = {}
for i, ep in enumerate(episodes[1:-1]):#跳过第0段和最后一段
    split_data[f'motion{i}'] = ep

joblib.dump(split_data, 'g800.pkl')
print("已保存为 your_motion_file_split.pkl")