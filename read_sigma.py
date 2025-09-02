#!/usr/bin/env python3

from tbparse import SummaryReader
import numpy as np
from pathlib import Path
import argparse
import shutil
from typing import Dict, Any


def read_sigma_from_tensorboard(log_dir_path):
    """从tensorboard事件文件中读取sigma参数"""

    print(f"读取tensorboard目录: {log_dir_path}")

    # 使用tbparse读取事件文件
    reader = SummaryReader(log_dir_path)
    df = reader.scalars

    # 存储sigma参数
    sigma_values = {}

    # 查找包含sigma的参数
    sigma_tags = df[df['tag'].str.contains('sigma', case=False, na=False)]

    if len(sigma_tags) > 0:
        # 按step分组，取最新的值
        latest_sigma = sigma_tags.groupby('tag').last()

        for tag, row in latest_sigma.iterrows():
            if 'teleop' in tag:
                sigma_name = tag.split('/')[-1]  # 获取参数名
                sigma_values[sigma_name] = {
                    'step': row['step'],
                    'value': row['value']
                }

    return sigma_values


def print_sigma_values(sigma_values):
    """打印sigma参数值"""
    print("\n找到的sigma参数:")
    print("-" * 50)

    for param_name, param_info in sigma_values.items():
        print(f"{param_name}: {param_info['value']:.6f} (step: {param_info['step']})")

    return sigma_values


def generate_updated_config(sigma_values, multiplier=1.5):
    """生成更新后的配置，将sigma值乘以multiplier"""
    print(f"\n将sigma参数乘以 {multiplier}:")
    print("-" * 50)

    # sigma参数映射 - 从tensorboard名称到配置文件名称的映射
    sigma_mapping = {
        'adp_sigma_teleop_upper_body_pos': 'teleop_upper_body_pos',
        'adp_sigma_teleop_lower_body_pos': 'teleop_lower_body_pos',
        'adp_sigma_teleop_vr_3point_pos': 'teleop_vr_3point_pos',
        'adp_sigma_teleop_feet_pos': 'teleop_feet_pos',
        'adp_sigma_teleop_body_rot': 'teleop_body_rot',
        'adp_sigma_teleop_body_vel': 'teleop_body_vel',
        'adp_sigma_teleop_body_ang_vel': 'teleop_body_ang_vel',
        'adp_sigma_teleop_joint_pos': 'teleop_joint_pos',
        'adp_sigma_teleop_joint_vel': 'teleop_joint_vel',
        'adp_sigma_teleop_max_joint_pos': 'teleop_max_joint_pos'
    }

    updated_config = {}

    for tb_name, config_name in sigma_mapping.items():
        if tb_name in sigma_values:
            original_value = sigma_values[tb_name]['value']
            new_value = original_value * multiplier
            updated_config[config_name] = new_value
            print(f"{config_name}: {original_value:.6f} -> {new_value:.6f}")
        else:
            print(f"未找到参数: {tb_name}")

    return updated_config


def _format_float(value: float) -> str:
    """格式化浮点数为 6 位小数的字符串。"""
    return f"{float(value):.6f}"


def update_yaml_reward_tracking_sigma(yaml_relative_path: str, updated_config: Dict[str, Any], make_backup: bool = True) -> bool:
    """仅替换 YAML 中 rewards.reward_tracking_sigma 下对应键的数值，保留其它内容与注释。

    基于行级替换实现：
    - 定位到 '  reward_tracking_sigma:' 行
    - 在该块内部（四空格缩进）逐行匹配 'key: value' 并替换
    """
    yaml_path = Path(yaml_relative_path)
    if not yaml_path.exists():
        print(f"YAML 文件不存在: {yaml_path}")
        return False

    text = yaml_path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)

    # 备份
    if make_backup:
        backup_path = yaml_path.with_suffix(yaml_path.suffix + ".bak")
        shutil.copyfile(yaml_path, backup_path)
        print(f"已备份到: {backup_path}")

    # 定位块起始
    start_idx = -1
    for i, line in enumerate(lines):
        # 精确匹配两空格缩进的块头
        if line.startswith('  ') and line.strip() == 'reward_tracking_sigma:':
            start_idx = i
            break

    if start_idx == -1:
        print("未找到 'reward_tracking_sigma:' 块，未做修改。")
        return False

    # 在块内进行替换（块内键使用四空格缩进）
    i = start_idx + 1
    changed = False
    while i < len(lines):
        line = lines[i]
        # 到达下一个两空格缩进的顶层键，结束该块
        if line.startswith('  ') and not line.startswith('    '):
            break
        # 仅处理四空格缩进且包含冒号的行
        if line.startswith('    ') and ':' in line:
            # 提取 key
            # 去除行内注释的影响（该块通常无行内注释）
            content = line.strip()
            key_part = content.split(':', 1)[0].strip()
            if key_part in updated_config:
                new_val = _format_float(updated_config[key_part])
                # 重建行，保留原有换行符
                lines[i] = f"    {key_part}: {new_val}" + ("\n" if line.endswith("\n") else "")
                changed = True
        i += 1

    if not changed:
        print("未发现可替换的键，文件保持不变。")
        return False

    yaml_path.write_text(''.join(lines), encoding='utf-8')
    print(f"已更新文件: {yaml_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 TensorBoard sigma 并回写到 YAML 的 reward_tracking_sigma")
    parser.add_argument("--log_dir", type=str, default="/home/harry/Desktop/asap-pbhc/logs/MotionTracking/round_sick", help="TensorBoard 日志目录")
    parser.add_argument("--yaml", type=str, default="humanoidverse/config/rewards/motion_tracking/final_training.yaml", help="目标 YAML 相对路径")
    parser.add_argument("--multiplier", type=float, default=2.0, help="sigma 放大倍率")
    parser.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份")
    args = parser.parse_args()

    try:
        # 读取sigma参数
        sigma_values = read_sigma_from_tensorboard(args.log_dir)

        if sigma_values:
            # 打印原始值
            print_sigma_values(sigma_values)

            # 生成更新后的配置
            updated_config = generate_updated_config(sigma_values, multiplier=args.multiplier)

            print(f"\n生成的配置更新:")
            print("-" * 50)
            print("reward_tracking_sigma:")
            for param_name, value in updated_config.items():
                print(f"  {param_name}: {_format_float(value)}")

            # 写回 YAML（相对路径）
            print("\n写回到 YAML...")
            ok = update_yaml_reward_tracking_sigma(args.yaml, updated_config, make_backup=(not args.no_backup))
            if ok:
                print("写回完成。")
            else:
                print("未进行写回或写回失败。")
        else:
            print("未找到任何sigma参数")
            print("请检查日志目录是否正确，或者训练是否启用了自适应sigma")

    except Exception as e:
        print(f"读取或写回时出错: {e}")
        print(f"请确保目录路径正确: {args.log_dir}，以及 YAML 路径: {args.yaml}")