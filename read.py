import joblib
import numpy as np
import torch

def print_array_stats(arr, indent=0):
    if isinstance(arr, np.ndarray):
        print("    " * indent + f"  Shape: {arr.shape}")
        if arr.size > 0:
            print("    " * indent + f"  Dtype: {arr.dtype}")
            print("    " * indent + f"  Mean: {np.mean(arr):.4f}")
            print("    " * indent + f"  Std: {np.std(arr):.4f}")
            print("    " * indent + f"  Min: {np.min(arr):.4f}")
            print("    " * indent + f"  Max: {np.max(arr):.4f}")
    elif isinstance(arr, torch.Tensor):
        print("    " * indent + f"  Shape: {list(arr.shape)}")
        if arr.numel() > 0:
            print("    " * indent + f"  Dtype: {arr.dtype}")
            # For integer types like torch.int64, skip mean/std/min/max as it causes errors
            if arr.is_floating_point() or arr.is_complex():
                print("    " * indent + f"  Mean: {torch.mean(arr).item():.4f}")
                print("    " * indent + f"  Std: {torch.std(arr).item():.4f}")
                print("    " * indent + f"  Min: {torch.min(arr).item():.4f}")
                print("    " * indent + f"  Max: {torch.max(arr).item():.4f}")
            else:
                print("    " * indent + "  (Skipping numerical stats for non-floating point tensor)")

def analyze_pkl_structure(data, current_depth=0, max_depth=3, indent=0):
    prefix = "    " * indent
    if current_depth > max_depth:
        print(prefix + f"... (max depth {max_depth} reached)")
        return

    if isinstance(data, dict):
        print(prefix + f"Type: Dict (Keys: {len(data)})")
        for key, value in data.items():
            print(prefix + f"  Key: '{key}'")
            if key == "pose_aa" and isinstance(value, (np.ndarray, torch.Tensor)):
                print(prefix + f"  Type: {type(value).__name__}")
                print(prefix + f"  Shape: {value.shape}")
                print(prefix + f"  Dtype: {value.dtype}")
                print(prefix + f"  Mean: {np.mean(value) if isinstance(value, np.ndarray) else torch.mean(value).item():.4f}")
                print(prefix + f"  Std: {np.std(value) if isinstance(value, np.ndarray) else torch.std(value).item():.4f}")
                print(prefix + f"  Min: {np.min(value) if isinstance(value, np.ndarray) else torch.min(value).item():.4f}")
                print(prefix + f"  Max: {np.max(value) if isinstance(value, np.ndarray) else torch.max(value).item():.4f}")
                print(prefix + f"  Number of joints: {value.shape[1]}")
                print(prefix + f"  Expected joints: 27 (1 root + 23 dof + 3 augment)")
            else:
                analyze_pkl_structure(value, current_depth + 1, max_depth, indent + 1)
    elif isinstance(data, list):
        print(prefix + f"Type: List (Length: {len(data)})")
        if len(data) > 0:
            print(prefix + f"  First element type: {type(data[0])}")
            for i, item in enumerate(data):
                if i >= 1: # Only show first element for brevity in lists
                    break
                print(prefix + f"  Element [{i}]:")
                analyze_pkl_structure(item, current_depth + 1, max_depth, indent + 1)
        else:
            print(prefix + "  List is empty.")
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        print(prefix + f"Type: {type(data).__name__}")
        print_array_stats(data, indent)
    else:
        value_repr = str(data)
        if len(value_repr) > 100:
            value_repr = value_repr[:97] + "..."
        print(prefix + f"Type: {type(data).__name__}, Value: {value_repr}")

def main_analyze(pkl_path):
    print(f"分析文件: {pkl_path}")
    try:
        data = joblib.load(pkl_path)
        print(f"成功加载: {pkl_path}")
        analyze_pkl_structure(data)

    except Exception as e:
        print(f"加载文件失败: {e}")

if __name__ == "__main__":
    pkl_file_to_analyze = "/home/harry/Desktop/ASAP1/mujoco_motion480.pkl"
    main_analyze(pkl_file_to_analyze)
#/home/harry/Desktop/ASAP1/logs/MotionTracking/20250317_215927-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/motions/G19900 gene430_.pkl

#/home/harry/Desktop/ASAP1/asap_mujoco_sim/mujoco_motion.pkl