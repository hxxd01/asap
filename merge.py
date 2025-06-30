import joblib

file1 = '/home/harry/Desktop/ASAP1/mujocomerge8600.pkl'
file2 = '/home/harry/Desktop/ASAP1/mujoco_motion4300_3.pkl'
output = '/home/harry/Desktop/ASAP1/mujocomerge12900.pkl'

all_motions = {}
motion_idx = 0

for pkl_file in [file1, file2]:
    data = joblib.load(pkl_file)
    for key in data:
        new_key = f"motion{motion_idx}"
        all_motions[new_key] = data[key]
        motion_idx += 1

joblib.dump(all_motions, output)
print(f"合并完成，总共 {len(all_motions)} 个 motion，已保存到 {output}")