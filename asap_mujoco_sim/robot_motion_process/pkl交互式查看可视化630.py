#!/usr/bin/env python3
"""
PBHC动作可视化工具 - 交互式版本
自动列出所有可用的pkl文件并让用户选择

使用方法：
    python run_motion_vis_interactive.py
"""

import os
import subprocess
import sys

def get_available_motions():
    """获取所有可用的动作文件"""
    motion_files = []
    motion_dirs = [
        "/home/harry/Desktop/asap/asap_mujoco_sim/example/motion_data",
        # 如果有其他目录包含pkl文件，可以在这里添加
    ]
    
    for motion_dir in motion_dirs:
        if os.path.exists(motion_dir):
            for file in os.listdir(motion_dir):
                if file.endswith('.pkl'):
                    motion_files.append(os.path.join(motion_dir, file))
    
    return sorted(motion_files)

def show_motion_menu(motion_files):
    """显示动作选择菜单"""
    print("🎭 PBHC动作可视化工具")
    print("=" * 50)
    print("📁 可用的动作文件:")
    print()
    
    for i, file in enumerate(motion_files, 1):
        filename = os.path.basename(file)
        # 移除.pkl扩展名并美化显示
        display_name = filename.replace('.pkl', '').replace('_', ' ').title()
        print(f"  {i:2d}. {display_name}")
    
    print(f"  {len(motion_files)+1:2d}. 🚪 退出")
    print()

def run_visualization(pkl_file, speed=1.0):
    """运行可视化工具"""
    print(f"\n🚀 启动可视化: {os.path.basename(pkl_file)}")
    print(f"⚡ 播放速度: {speed}x")
    print()
    
    print("🎮 控制说明:")
    print("  空格键    - 暂停/播放")
    print("  R键       - 重置到开始")
    print("  L键       - 加速播放")
    print("  K键       - 减速播放")
    print("  J键       - 切换倒放")
    print("  左右箭头  - 逐帧控制")
    print("  Q键       - 退出可视化")
    print("=" * 50)
    
    cmd = [
        "python", 
        "robot_motion_process/vis_q_mj.py",
        f"+motion_file={pkl_file}",
        f"+speed={speed}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行出错: {e}")
    except KeyboardInterrupt:
        print("\n👋 用户中断，返回主菜单")
    except Exception as e:
        print(f"❌ 未预期的错误: {e}")

def get_speed_setting():
    """获取播放速度设置"""
    while True:
        try:
            speed_input = input("⚡ 请输入播放速度 (默认1.0): ").strip()
            if not speed_input:
                return 1.0
            speed = float(speed_input)
            if speed > 0:
                return speed
            else:
                print("❌ 速度必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")

def main():
    """主函数"""
    motion_files = get_available_motions()
    
    if not motion_files:
        print("❌ 未找到pkl动作文件")
        print("请确保 example/motion_data 目录中包含.pkl文件")
        return
    
    while True:
        show_motion_menu(motion_files)
        
        try:
            choice = input("🔢 请选择要查看的动作 (输入数字): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            
            if choice_num == len(motion_files) + 1:
                print("👋 再见！")
                break
            elif 1 <= choice_num <= len(motion_files):
                selected_file = motion_files[choice_num - 1]
                
                # 询问播放速度
                speed = get_speed_setting()
                
                # 运行可视化
                run_visualization(selected_file, speed)
                
                # 询问是否继续
                while True:
                    continue_choice = input("\n🔄 是否继续选择其他动作? (y/n): ").strip().lower()
                    if continue_choice in ['y', 'yes', '是']:
                        break
                    elif continue_choice in ['n', 'no', '否']:
                        print("👋 再见！")
                        return
                    else:
                        print("❌ 请输入 y 或 n")
            else:
                print(f"❌ 请输入 1 到 {len(motion_files)+1} 之间的数字")
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main() 