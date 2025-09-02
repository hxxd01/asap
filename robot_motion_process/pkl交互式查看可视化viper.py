#!/usr/bin/env python3
"""
🤖 PBHC动作可视化工具 - Viper机器人专用交互式版本
自动列出所有可用的pkl文件并让用户选择，专门适配27DOF的Viper机器人

使用方法：
    python pkl交互式查看可视化viper.py
"""

import os
import subprocess
import sys

def get_available_motions():
    """获取所有可用的动作文件"""
    motion_files = []
    motion_dirs = [
        "example/motion_data",
        "asap_mujoco_sim-viper-2/motion_data",  # 添加Viper专用目录
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
    print("🤖 Viper机器人动作可视化工具")
    print("=" * 50)
    print("📁 可用的动作文件 (27DOF Viper Robot):")
    print()
    
    for i, file in enumerate(motion_files, 1):
        filename = os.path.basename(file)
        # 移除.pkl扩展名并美化显示
        display_name = filename.replace('.pkl', '').replace('_', ' ').title()
        # 如果包含viper，添加🐍emoji
        if 'viper' in filename.lower():
            display_name = f"🐍 {display_name}"
        print(f"  {i:2d}. {display_name}")
    
    print(f"  {len(motion_files)+1:2d}. 🚪 退出")
    print()

def run_visualization(pkl_file, speed=1.0):
    """运行Viper专用可视化工具"""
    print(f"\n🚀 启动Viper机器人可视化: {os.path.basename(pkl_file)}")
    print(f"⚡ 播放速度: {speed}x")
    print(f"🤖 机器人型号: Viper (27DOF)")
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
    
    # 🔥 使用Viper专用的可视化工具
    cmd = [
        "python", 
        "robot_motion_process/vis_q_mj_viper.py",  # 使用Viper版本
        f"+motion_file={pkl_file}",
        f"+speed={speed}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行出错: {e}")
        print("\n💡 提示:")
        print("   - 确保轨迹数据是27DOF的Viper格式")
        print("   - 检查依赖环境是否正确安装")
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

def check_viper_dependencies():
    """检查Viper可视化工具的依赖"""
    viper_vis_path = "robot_motion_process/vis_q_mj_viper.py"
    if not os.path.exists(viper_vis_path):
        print(f"❌ 未找到Viper可视化工具: {viper_vis_path}")
        print("请确保文件存在后再运行")
        return False
    
    viper_xml_path = "/home/harry/Desktop/PBHC_old/asap_mujoco_sim-viper-2/mujuco_deploy (1)/robot_description/mjcf/scene_deploy.xml"
    if not os.path.exists(viper_xml_path):
        print(f"❌ 未找到Viper机器人模型文件: {viper_xml_path}")
        print("请检查asap_mujoco_sim-viper-2目录是否完整")
        return False
        
    return True

def main():
    """主函数"""
    print("🔍 检查Viper环境...")
    
    if not check_viper_dependencies():
        return
        
    motion_files = get_available_motions()
    
    if not motion_files:
        print("❌ 未找到pkl动作文件")
        print("请确保以下目录中包含.pkl文件:")
        print("  - example/motion_data")
        print("  - asap_mujoco_sim-viper-2/motion_data") 
        return
    
    print("✅ Viper环境检查通过")
    print()
    
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
                
                # 检查文件是否为Viper格式
                print(f"\n📋 选中文件: {os.path.basename(selected_file)}")
                
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