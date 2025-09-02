import mujoco, mujoco_viewer
import numpy as np
import onnxruntime
import yaml
import os
import joblib
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import torch
import pickle
from datetime import datetime


def read_conf(config_file):
    cfg = SimpleNamespace()
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    cfg.policy_path = config["policy_path"]
    cfg.cycle_time = config["cycle_time"]
    cfg.xml_path = config["xml_path"]
    cfg.num_single_obs = config["num_single_obs"]
    cfg.simulation_dt = config["simulation_dt"]
    cfg.simulation_duration = config["simulation_duration"]
    cfg.control_decimation = config["control_decimation"]
    cfg.frame_stack = config["frame_stack"]
    cfg.episode_steps = config["episode_steps"]
    cfg.total_steps = config["total_steps"]
    cfg.default_dof_pos = config["default_dof_pos"]
    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]
    cfg.obs_scale_gvec = config["obs_scale_gvec"]
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]
    cfg.obs_scale_hist = config["obs_scale_hist"]
    cfg.num_actions = config["num_actions"]
    cfg.clip_observations = config["clip_observations"]
    cfg.clip_actions = config["clip_actions"]
    cfg.action_scale = config["action_scale"]
    cfg.kps = config["kps"]
    cfg.kds = config["kds"]
    cfg.tau_limit = config["tau_limit"]
    cfg.render = config["render"]
    cfg.use_noise = config["use_noise"]
    cfg.init_pos = config["init_pos"]

    # 添加termination相关配置
    cfg.use_termination = config.get("use_termination", True)
    cfg.termination_gravity_x = config.get("termination_gravity_x", 0.85)
    cfg.termination_gravity_y = config.get("termination_gravity_y", 0.85)

    # 添加数据处理配置
    cfg.min_episode_length = config.get("min_episode_length", 75)
    cfg.auto_process = config.get("auto_process", True)

    return cfg


def get_mujoco_data(data):
    mujoco_data = {}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = np.array([q[4], q[5], q[6], q[3]])
    r = R.from_quat(quat)
    base_angvel = dq[3:6]
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    mujoco_data['mujoco_dof_pos'] = q[7:]
    mujoco_data['mujoco_dof_vel'] = dq[6:]
    mujoco_data['mujoco_base_angvel'] = base_angvel
    mujoco_data['mujoco_gvec'] = gvec
    mujoco_data['base_height'] = q[2]
    return mujoco_data


def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    
    if cfg.use_noise:
        noise_base_ang_vel = (np.random.rand(3) * 2. - 1.) * 0.3
        noise_projected_gravity = (np.random.rand(3) * 2. - 1.) * 0.2
        noise_dof_pos = (np.random.rand(23) * 2. - 1.) * 0.01
        noise_dof_vel = (np.random.rand(23) * 2. - 1.) * 1
    else:
        noise_base_ang_vel = np.zeros(3)
        noise_projected_gravity = np.zeros(3)
        noise_dof_pos = np.zeros(23)
        noise_dof_vel = np.zeros(23)
    
    ref_motion_phase = (counter + 1) * cfg.simulation_dt / cfg.cycle_time
    ref_motion_phase = np.clip(ref_motion_phase, 0, 1)
    num_obs_input = (cfg.frame_stack + 1) * cfg.num_single_obs

    obs_all = np.zeros([1, num_obs_input], dtype=np.float32)
    obs_sigle = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
    obs_sigle[0, 0:23] = action
    obs_sigle[0, 23:26] = (mujoco_base_angvel + noise_base_ang_vel) * cfg.obs_scale_base_ang_vel
    dof_pos = mujoco_dof_pos - cfg.default_dof_pos
    obs_sigle[0, 26:49] = (dof_pos + noise_dof_pos) * cfg.obs_scale_dof_pos
    obs_sigle[0, 49:72] = (mujoco_dof_vel + noise_dof_vel) * cfg.obs_scale_dof_vel
    obs_sigle[0, 72:75] = (mujoco_gvec + noise_projected_gravity) * cfg.obs_scale_gvec
    obs_sigle[0, 75] = ref_motion_phase * cfg.obs_scale_refmotion

    obs_all[0, 0:23] = obs_sigle[0, 0:23].copy()
    obs_all[0, 23:26] = obs_sigle[0, 23:26].copy()
    obs_all[0, 26:49] = obs_sigle[0, 26:49].copy()
    obs_all[0, 49:72] = obs_sigle[0, 49:72].copy()
    obs_all[0, 72:376] = hist_obs_c[0] * cfg.obs_scale_hist
    obs_all[0, 376:379] = obs_sigle[0, 72:75].copy()
    obs_all[0, 379] = obs_sigle[0, 75].copy()

    hist_obs_cat = update_hist_obs(hist_dict, obs_sigle)
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)

    return obs_all, hist_obs_cat


def update_hist_obs(hist_dict, obs_sigle):
    slices = {
        'actions': slice(0, 23),
        'base_ang_vel': slice(23, 26),
        'dof_pos': slice(26, 49),
        'dof_vel': slice(49, 72),
        'projected_gravity': slice(72, 75),
        'ref_motion_phase': slice(75, 76)
    }

    for key, slc in slices.items():
        arr = np.delete(hist_dict[key], -1, axis=0)
        arr = np.vstack((obs_sigle[0, slc], arr))
        hist_dict[key] = arr

    hist_obs = np.concatenate([
        hist_dict[key].reshape(1, -1)
        for key in hist_dict.keys()
    ], axis=1).astype(np.float32)
    return hist_obs


def pd_control(target_pos, dof_pos, target_vel, dof_vel, cfg):
    torque_out = (target_pos - dof_pos) * cfg.kps + (target_vel - dof_vel) * cfg.kds
    return torque_out


def parse_dof_axis_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    joints = root.findall('.//joint')
    dof_axis = []
    for j in joints:
        if 'type' in j.attrib and j.attrib['type'] in ['free', 'float']:
            continue
        axis_str = j.attrib.get('axis', None)
        if axis_str is not None:
            axis = [float(x) for x in axis_str.strip().split()]
            dof_axis.append(axis)
    return np.array(dof_axis, dtype=np.float32)


def check_termination(mujoco_data, cfg, counter):
    if not cfg.use_termination:
        return False

    mujoco_gvec = mujoco_data["mujoco_gvec"]
    gravity_x_violation = abs(mujoco_gvec[0]) > cfg.termination_gravity_x
    gravity_y_violation = abs(mujoco_gvec[1]) > cfg.termination_gravity_y
    gravity_termination = gravity_x_violation or gravity_y_violation

    should_terminate = gravity_termination

    if should_terminate:
        print(f"[Termination] Step {counter}: "
              f"gravity_x={mujoco_gvec[0]:.3f}(>{cfg.termination_gravity_x}), "
              f"gravity_y={mujoco_gvec[1]:.3f}(>{cfg.termination_gravity_y})")

    return should_terminate


def process_motion_data(input_path, cfg):
    """
    处理采集到的运动数据，分离episode并过滤短轨迹
    """
    print(f"\n=== 开始处理运动数据 ===")
    print(f"输入文件: {input_path}")
    
    # 读取原始数据
    data = joblib.load(input_path)
    motion = data

    # 取出terminate信号
    terminate = np.array(motion['terminate'])
    T = len(terminate)

    print(f"总帧数: {T}")
    print(f"terminate标志数量: {np.sum(terminate)}")

    # 找到所有终止点
    episode_ends = np.where(terminate)[0]
    if len(episode_ends) == 0 or episode_ends[-1] != T - 1:
        episode_ends = np.append(episode_ends, T - 1)
    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

    print(f"Episode起始点: {episode_starts}")
    print(f"Episode结束点: {episode_ends}")

    # 计算每个episode的长度
    episode_lengths = episode_ends - episode_starts + 1
    print(f"Episode长度: {episode_lengths}")
    print(f"平均长度: {np.mean(episode_lengths):.1f}")
    print(f"最短长度: {np.min(episode_lengths)}")
    print(f"最长长度: {np.max(episode_lengths)}")

    # 切分每个episode
    episodes = []
    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep = {}
        for k in motion.keys():
            if isinstance(motion[k], np.ndarray) and motion[k].shape[0] == T:
                ep[k] = motion[k][start:end + 1]
            else:
                ep[k] = motion[k]
        episodes.append(ep)
        print(f"Episode {i}: {end - start + 1} 帧")

    print(f"\n共切分出{len(episodes)}条episode")

    # 过滤长度不足的episode
    MIN_LENGTH = cfg.min_episode_length
    filtered_episodes = []
    filtered_lengths = []

    print(f"\n=== 过滤长度{MIN_LENGTH}以下的episode ===")
    for i, (ep, length) in enumerate(zip(episodes, episode_lengths)):
        if length >= MIN_LENGTH:
            filtered_episodes.append(ep)
            filtered_lengths.append(length)
            print(f"保留 Episode {i}: {length} 帧")
        else:
            print(f"去除 Episode {i}: {length} 帧 (< {MIN_LENGTH})")

    removed_count = len(episodes) - len(filtered_episodes)
    print(f"\n📊 过滤统计:")
    print(f"原始episode数量: {len(episodes)}")
    print(f"去除的episode数量: {removed_count}")
    print(f"保留的episode数量: {len(filtered_episodes)}")
    if len(episodes) > 0:
        print(f"保留率: {len(filtered_episodes) / len(episodes) * 100:.1f}%")

    if len(filtered_episodes) == 0:
        print("\n❌ 没有符合长度要求的episode!")
        return None

    # 转换数据格式
    filtered_data = {}
    for i, ep in enumerate(filtered_episodes):
        converted_ep = {}
        required_fields = ['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'action',
                          'terminate', 'root_lin_vel', 'root_ang_vel', 'dof_vel', 'motion_times']

        for key, value in ep.items():
            if key in required_fields:
                if isinstance(value, np.ndarray):
                    if key == 'action':
                        # 修复action维度：从(T,1,23)改为(T,23)
                        if len(value.shape) == 3 and value.shape[1] == 1:
                            converted_ep[key] = value.squeeze(1).astype(np.float32)
                        else:
                            converted_ep[key] = value.astype(np.float32)
                    elif key == 'terminate':
                        converted_ep[key] = value.astype(np.int64)
                    elif key in ['root_trans_offset', 'pose_aa', 'dof', 'root_rot',
                                'root_lin_vel', 'root_ang_vel', 'dof_vel', 'motion_times']:
                        converted_ep[key] = value.astype(np.float32)
                    else:
                        converted_ep[key] = value
                else:
                    converted_ep[key] = value

        converted_ep['fps'] = 50.0
        filtered_data[f'motion{i}'] = converted_ep

    # 生成输出文件名
    output_dir = 'processed_pkl_files'
    os.makedirs(output_dir, exist_ok=True)

    input_basename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_basename)[0]
    
    if name_without_ext.startswith('2025'):
        name_without_year = name_without_ext[4:]
    else:
        name_without_year = name_without_ext

    output_filename = f'{name_without_year}_processed_min{MIN_LENGTH}.pkl'
    output_path = os.path.join(output_dir, output_filename)

    # 保存处理后的数据
    joblib.dump(filtered_data, output_path)
    
    filtered_lengths = np.array(filtered_lengths)
    print(f"\n✅ 过滤后的轨迹已保存为: {output_path}")
    print(f"📊 总帧数: {np.sum(filtered_lengths)}")
    print(f"⏱️ 总时长: {np.sum(filtered_lengths) * 0.02:.2f}秒")
    print(f"🔧 数据格式已匹配训练格式")

    return output_path


def run_and_save_mujoco(cfg, save_path):
    current_step = 0
    motions_for_saving = {'root_trans_offset': [], 'pose_aa': [], 'dof': [], 'root_rot': [], 'action': [], 'terminate': [], "root_lin_vel": [],
                          "root_ang_vel": [], "dof_vel": [], "motion_times": []}
    dt = cfg.simulation_dt * cfg.control_decimation
    dof_axis = parse_dof_axis_from_xml(cfg.xml_path)

    termination_stats = {
        'gravity_terminations': 0,
        'height_terminations': 0,
        'torque_terminations': 0,
        'normal_completions': 0,
        'total_episodes': 0
    }

    while True:
        # 回合初始化
        model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        data = mujoco.MjData(model)
        model.opt.timestep = cfg.simulation_dt

        data.qpos[-cfg.num_actions:] = cfg.init_pos
        mujoco.mj_step(model, data)

        # 初始化
        model.opt.timestep = cfg.simulation_dt

        # mujoco可视化设置
        if cfg.render:
            viewer = mujoco_viewer.MujocoViewer(model, data)
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
            viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        # 策略模型加载
        onnx_model_path = cfg.policy_path
        policy = onnxruntime.InferenceSession(onnx_model_path)

        # 变量初始化
        #target_dof_pos = np.zeros((1, len(cfg.default_dof_pos.copy())))
        target_dof_pos = cfg.default_dof_pos.copy()
        action = np.zeros(cfg.num_actions, dtype=np.float32)

        # 初始化历史观测
        hist_dict = {'actions': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
                     'base_ang_vel': np.zeros((cfg.frame_stack, 3), dtype=np.double),
                     'dof_pos': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
                     'dof_vel': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
                     'projected_gravity': np.zeros((cfg.frame_stack, 3), dtype=np.double),
                     'ref_motion_phase': np.zeros((cfg.frame_stack, 1), dtype=np.double),
                     }
        history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                        'dof_vel', 'projected_gravity', 'ref_motion_phase']
        hist_obs = []
        for key in history_keys:
            hist_obs.append(hist_dict[key].reshape(1, -1))
        hist_obs_c = np.concatenate(hist_obs, axis=1)
        counter = 0
        terminate_flag = False
        episode_terminated_early = False

        for step in range(cfg.episode_steps * cfg.control_decimation):
            mujoco_data = get_mujoco_data(data)

            tau = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros_like(cfg.kds), mujoco_data["mujoco_dof_vel"], cfg)
            tau_limit = np.array(cfg.tau_limit)
            tau = np.clip(tau, -tau_limit, tau_limit)
            mujoco_data['torques'] = tau

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            if counter % cfg.control_decimation == 0:
                current_step += 1
                obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg)
                policy_input = {policy.get_inputs()[0].name: obs_buff}
                action = policy.run(["action"], policy_input)[0]
                action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)
                target_dof_pos = action * cfg.action_scale + cfg.default_dof_pos

                should_terminate = check_termination(mujoco_data, cfg, counter)

                # 保存数据
                q = data.qpos.astype(np.double)
                dq = data.qvel.astype(np.double)
                quat = np.array([q[4], q[5], q[6], q[3]])
                root_trans = q[:3]
                root_rot = quat
                dof = q[7:]
                root_rot_vec = R.from_quat(root_rot).as_rotvec()

                joint_aa = dof[:, None] * dof_axis
                num_augment_joint = 3
                pose_aa = np.concatenate([
                    root_rot_vec[None, :],
                    joint_aa,
                    np.zeros((num_augment_joint, 3), dtype=np.float32)
                ], axis=0)
                root_lin_vel = dq[:3]
                root_ang_vel = dq[3:6]
                dof_vel = dq[6:]

                if not cfg.render:
                    motions_for_saving['root_trans_offset'].append(root_trans)
                    motions_for_saving['root_rot'].append(root_rot)
                    motions_for_saving['dof'].append(dof)
                    motions_for_saving['pose_aa'].append(pose_aa)
                    motions_for_saving['action'].append(action)
                    motions_for_saving['root_lin_vel'].append(root_lin_vel)
                    motions_for_saving['root_ang_vel'].append(root_ang_vel)
                    motions_for_saving['dof_vel'].append(dof_vel)
                    motion_times = counter * cfg.simulation_dt
                    motions_for_saving['motion_times'].append(motion_times)
                    motions_for_saving['fps'] = 1.0 / dt

                    if should_terminate:
                        motions_for_saving['terminate'].append(True)
                        print(f"[Early Termination] Episode terminated at step {counter}/{cfg.episode_steps}")
                        episode_terminated_early = True
                    elif ((current_step) % cfg.episode_steps) == 0:
                        motions_for_saving['terminate'].append(True)
                    else:
                        motions_for_saving['terminate'].append(False)

                    print(f"current_step:{current_step}/total_step:{cfg.total_steps}")

                if should_terminate:
                    break

            counter += 1
            if cfg.render:
                viewer.render()

        # 更新统计信息
        termination_stats['total_episodes'] += 1
        if episode_terminated_early:
            termination_stats['gravity_terminations'] += 1
        else:
            termination_stats['normal_completions'] += 1

        if current_step >= cfg.total_steps:
            break
        if cfg.render:
            viewer.close()

    # 打印termination统计
    print(f"\n[Termination Statistics]")
    print(f"Total Episodes: {termination_stats['total_episodes']}")
    print(f"Normal Completions: {termination_stats['normal_completions']}")
    print(f"Early Terminations: {termination_stats['gravity_terminations']}")
    print(f"Early Termination Rate: {termination_stats['gravity_terminations'] / termination_stats['total_episodes'] * 100:.1f}%")

    if not cfg.render:
        # 拼接所有list为ndarray
        result = {}
        for k in motions_for_saving:
            if k != 'fps':
                result[k] = np.array(motions_for_saving[k])
        result['fps'] = motions_for_saving['fps']

        # 保存原始数据
        save_f = open(save_path, 'wb')
        pickle.dump(result, save_f)
        save_f.close()

        print(f"✅ Motion data saved to: {save_path}")
        print(f"📊 Total frames saved: {len(result['motion_times'])}")
        print(f"⏱️ Total duration: {result['motion_times'][-1]:.2f}s")
        print(f"🎬 FPS: {result['fps']:.1f}")

        return save_path

    return None


def main():
    """
    主函数：运行完整的轨迹采集和处理流程
    """
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, "g1_config", "mujoco_config_p.yaml")
    cfg = read_conf(config_file)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_save_path = os.path.join(current_directory, f"{timestamp}_mujoco_motion_raw.pkl")
    
    print(f"=== 开始轨迹采集 ===")
    print(f"原始数据将保存到: {raw_save_path}")

    # 步骤1：运行仿真并保存原始数据
    saved_path = run_and_save_mujoco(cfg, raw_save_path)
    
    if saved_path and cfg.auto_process:
        print(f"\n=== 开始自动处理数据 ===")
        # 步骤2：自动处理数据
        processed_path = process_motion_data(saved_path, cfg)
        
        if processed_path:
            print(f"\n🎉 完整流程完成!")
            print(f"📁 原始数据: {saved_path}")
            print(f"📁 处理后数据: {processed_path}")
            
            # 可选：删除原始文件以节省空间
            # os.remove(saved_path)
            # print(f"🗑️ 原始文件已删除")
        else:
            print(f"\n⚠️ 数据处理失败，原始数据保留在: {saved_path}")
    else:
        print(f"\n✅ 轨迹采集完成，数据保存在: {saved_path}")
        if not cfg.auto_process:
            print("💡 如需处理数据，请在配置文件中设置 auto_process: true")

    print("-----done------")


if __name__ == '__main__':
    main() 