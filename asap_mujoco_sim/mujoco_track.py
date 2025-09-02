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
    cfg.init_pos=config["init_pos"]
    cfg.test=config["test"]

    # 添加termination相关配置
    cfg.use_termination = config.get("use_termination", True)
    cfg.termination_gravity_x = config.get("termination_gravity_x", 0.85)
    cfg.termination_gravity_y = config.get("termination_gravity_y", 0.85)

    return cfg


def get_mujoco_data(data):
    mujoco_data = {}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = np.array([q[4], q[5], q[6], q[3]])
    # v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # 把 base 的线速度（data.qvel[:3]）从全局坐标系变换到base局部坐标系（用四元数逆变换）。
    r = R.from_quat(quat)
    base_angvel = dq[3:6]
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    mujoco_data['mujoco_dof_pos'] = q[7:]
    mujoco_data['mujoco_dof_vel'] = dq[6:]
    mujoco_data['mujoco_base_angvel'] = base_angvel
    mujoco_data['mujoco_gvec'] = gvec

    # 添加基座高度信息
    mujoco_data['base_height'] = q[2]  # z坐标作为高度

    return mujoco_data


def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    ''' obs:
    action #  23
    base_ang_vel # 3
    dof_pos # 23
    dof_vel # 23
    history_actor # 4 * (23+3+23+23+3+1)=4*76=304
    projected_gravity # 3
    ref_motion_phase # 1
    '''
    '''noise_scales: {
        base_rot: 0.0,
        base_pos: 0.0,
        base_lin_vel: 0.0,
        base_ang_vel: 0.3,
        projected_gravity: 0.2,
        dof_pos: 0.01,
        dof_vel: 1.0,
        actions: 0.0,
        dif_local_rigid_body_pos: 0.0,
        local_ref_rigid_body_pos: 0.0,
        ref_motion_phase: 0.0,
        history_actor: 0.0,
        history_delta_actor: 0.0,
        history_critic: 0.0,
        z_actions: 0.0,
    }'''
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    '''0.3,0.2,0.01,1'''
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
    # obs_sigle[0, 23:26] = mujoco_base_angvel * cfg.obs_scale_base_ang_vel

    obs_sigle[0, 23:26] = (mujoco_base_angvel + noise_base_ang_vel) * cfg.obs_scale_base_ang_vel
    dof_pos = mujoco_dof_pos - cfg.default_dof_pos
    obs_sigle[0, 26:49] = (dof_pos + noise_dof_pos) * cfg.obs_scale_dof_pos
    obs_sigle[0, 49:72] = (mujoco_dof_vel + noise_dof_vel) * cfg.obs_scale_dof_vel
    # obs_sigle[0, 72:75] = mujoco_gvec * cfg.obs_scale_gvec
    obs_sigle[0, 72:75] = (mujoco_gvec + noise_projected_gravity) * cfg.obs_scale_gvec
    obs_sigle[0, 75] = ref_motion_phase * cfg.obs_scale_refmotion

    obs_all[0, 0:23] = obs_sigle[0, 0:23].copy()
    obs_all[0, 23:26] = obs_sigle[0, 23:26].copy()
    obs_all[0, 26:49] = obs_sigle[0, 26:49].copy()
    obs_all[0, 49:72] = obs_sigle[0, 49:72].copy()
    # 72:164 action;
    # 164:176 base_ang_vel
    # 176:268 dof_pos
    # 268:360 dof_vel
    # 360:372 gravity
    # 372:376 phase
    obs_all[0, 72:376] = hist_obs_c[0] * cfg.obs_scale_hist
    obs_all[0, 376:379] = obs_sigle[0, 72:75].copy()
    obs_all[0, 379] = obs_sigle[0, 75].copy()

    hist_obs_cat = update_hist_obs(hist_dict, obs_sigle)
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)

    return obs_all, hist_obs_cat


def update_hist_obs(hist_dict, obs_sigle):
    '''
        history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                    dof_vel', 'projected_gravity', 'ref_motion_phase']
    '''
    slices = {
        'actions': slice(0, 23),
        'base_ang_vel': slice(23, 26),
        'dof_pos': slice(26, 49),
        'dof_vel': slice(49, 72),
        'projected_gravity': slice(72, 75),
        'ref_motion_phase': slice(75, 76)
    }

    for key, slc in slices.items():
        # Remove oldest entry and add new observation
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
    # 找到所有joint节点
    joints = root.findall('.//joint')
    dof_axis = []
    for j in joints:
        # 跳过freejoint/floatjoint（通常是base）
        if 'type' in j.attrib and j.attrib['type'] in ['free', 'float']:
            continue
        axis_str = j.attrib.get('axis', None)
        if axis_str is not None:
            axis = [float(x) for x in axis_str.strip().split()]
            dof_axis.append(axis)
    return np.array(dof_axis, dtype=np.float32)


def check_termination(mujoco_data, cfg, counter):
    """
    检查是否需要终止当前回合
    基于重力投影和基座高度判断机器人是否倒地
    """
    if not cfg.use_termination:
        return False

    # 1. 重力投影判断（主要termination条件）
    mujoco_gvec = mujoco_data["mujoco_gvec"]
    gravity_x_violation = abs(mujoco_gvec[0]) > cfg.termination_gravity_x
    gravity_y_violation = abs(mujoco_gvec[1]) > cfg.termination_gravity_y
    gravity_termination = gravity_x_violation or gravity_y_violation

    # # 2. 基座高度判断（辅助termination条件）
    # base_height = mujoco_data.get("base_height", 1.0)  # 默认高度
    # height_termination = base_height < cfg.termination_min_height

    # # 3. 扭矩过大判断（可选）
    # torque_termination = False
    # if "torques" in mujoco_data:
    #     torques = mujoco_data["torques"]
    #     max_torque_ratio = np.max(np.abs(torques) / cfg.tau_limit)
    #     torque_termination = max_torque_ratio > cfg.termination_max_torque_ratio

    # 综合判断
    should_terminate = gravity_termination

    # 调试信息
    if should_terminate:
        print(f"[Termination] Step {counter}: "
              f"gravity_x={mujoco_gvec[0]:.3f}(>{cfg.termination_gravity_x}), "
              f"gravity_y={mujoco_gvec[1]:.3f}(>{cfg.termination_gravity_y})")

    return should_terminate


def run_and_save_mujoco(cfg, save_path):
    current_step = 0
    motions_for_saving = {'root_trans_offset': [], 'pose_aa': [], 'dof': [], 'root_rot': [], 'action': [], 'terminate': [], "root_lin_vel": [],
                          "root_ang_vel": [], "dof_vel": [], "motion_times": []}
    dt = cfg.simulation_dt * cfg.control_decimation
    dof_axis = parse_dof_axis_from_xml(cfg.xml_path)

    # 添加termination统计
    termination_stats = {
        'gravity_terminations': 0,
        'height_terminations': 0,
        'torque_terminations': 0,
        'normal_completions': 0,
        'total_episodes': 0
    }

    while True:  # 回合的循环
        # 回合初始化
        model = mujoco.MjModel.from_xml_path(cfg.xml_path)  # 从 XML 文件（通常是 .xml 或 .mjcf 格式的机器人模型文件）加载 Mujoco 物理模型，生成 model 对象。
        data = mujoco.MjData(model)  # 基于上面加载的 model，创建一个 data 对象，存储仿真过程中的所有状态（如关节位置、速度、力等）。
        model.opt.timestep = cfg.simulation_dt  # 每一步物理仿真的时间间隔。

        data.qpos[-cfg.num_actions:] = cfg.init_pos  # 将机器人所有关节的初始位置（qpos的最后若干个元素）设置为默认关节角度。
        mujoco.mj_step(model, data)  # 执行一次物理仿真步，确保初始状态下所有缓冲区都被正确初始化（有些仿真器需要先step一次）。

        # 初始化
        model.opt.timestep = cfg.simulation_dt

        # mujoco可视化设置
        if cfg.render:
            viewer = mujoco_viewer.MujocoViewer(model, data)
            # 摄像机的设置
            viewer.cam.distance = 5.0  # 设置摄像机距离场景中心的距离（越大越远，越小越近）。
            viewer.cam.azimuth = 90  # 设置摄像机的水平旋转角度（方位角），单位为度。90度通常是侧视。
            viewer.cam.elevation = -45  # 设置摄像机的俯仰角（仰视/俯视），单位为度。-45度是从上往下看。
            viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])  # 设置摄像机的焦点（即摄像机看向场景中的哪个点），这里是一个三维坐标。

        # 策略模型加载
        onnx_model_path = cfg.policy_path
        policy = onnxruntime.InferenceSession(onnx_model_path)  # 用 onnxruntime 加载 ONNX 格式的策略模型，这样后续就可以用 policy.run(...) 来进行神经网络推理

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

        for step in range(cfg.episode_steps * cfg.control_decimation):  # 回合内循环
            mujoco_data = get_mujoco_data(data)

            # 添加扭矩信息用于termination检查
            tau = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"],
                             np.zeros_like(cfg.kds), mujoco_data["mujoco_dof_vel"], cfg)
            tau_limit = np.array(cfg.tau_limit)  # 将列表转换为numpy数组
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

                # 检查termination条件（移到数据保存之前）
                should_terminate = check_termination(mujoco_data, cfg, counter)

                # 保存数据
                q = data.qpos.astype(np.double)  # 当前仿真状态下的所有关节位置（包括base的平移和旋转）
                dq = data.qvel.astype(np.double)  # 当前仿真状态下的所有关节速度（包括base的线速度和角速度）
                quat = np.array([q[4], q[5], q[6], q[3]])  # 正确xyzw顺序
                # quat = np.array([q[3], q[4], q[5], q[6]])  # 正确xyzw顺序
                root_trans = q[:3]
                root_rot = quat
                dof = q[7:]
                # base四元数转axis-angle
                root_rot_vec = R.from_quat(root_rot).as_rotvec()  # shape (3,)

                # 关节角度与旋转轴相乘
                joint_aa = dof[:, None] * dof_axis  # shape (23, 3)
                # print(dof_axis)#这个从xml文件里面读取，经过检查应该是没有问题的
                #  拼接
                num_augment_joint = 3
                pose_aa = np.concatenate([
                    root_rot_vec[None, :],  # (1, 3)
                    joint_aa,  # (num_dof, 3)
                    np.zeros((num_augment_joint, 3), dtype=np.float32)  # (num_augment_joint, 3)，三个虚拟关节
                ], axis=0)  # shape (1+num_dof+3, 3)
                root_lin_vel = dq[:3]
                root_ang_vel = dq[3:6]
                dof_vel = dq[6:]

                if not cfg.render:
                    motions_for_saving['root_trans_offset'].append(root_trans)  # gene[-0.38,1.16] mujoco[-0.03,0.97]
                    motions_for_saving['root_rot'].append(root_rot)  # gene[-0.89,0.94],mujoco[-0.05,1]
                    motions_for_saving['dof'].append(dof)  # gene[-2.19,1.83],mujoco[-1.72,0.84]
                    motions_for_saving['pose_aa'].append(pose_aa)  # gene[-2.76,2.64],mujoco[-3.09,3.14]
                    motions_for_saving['action'].append(action)

                    motions_for_saving['root_lin_vel'].append(root_lin_vel)  # gene[-3.02,3.04],mujoco[-2.36,1.43]
                    motions_for_saving['root_ang_vel'].append(root_ang_vel)  # gene[-9.6,10.26],mujoco[-1.95,5.00]
                    motions_for_saving['dof_vel'].append(dof_vel)  # gene [-22,13],mujoco[-5,9]
                    motion_times = counter * cfg.simulation_dt
                    # print(motion_times)
                    motions_for_saving['motion_times'].append(motion_times)
                    motions_for_saving['fps'] = 1.0 / dt
                    # print(num_steps)
                    # print(1.0 / cfg.simulation_dt)

                    # 根据termination状态设置terminate标志
                    if should_terminate:
                        motions_for_saving['terminate'].append(True)
                        print(f"[Early Termination] Episode terminated at step {counter}/{cfg.episode_steps}")
                        episode_terminated_early = True
                    elif ((current_step) % cfg.episode_steps) == 0:
                        motions_for_saving['terminate'].append(True)
                    else:
                        motions_for_saving['terminate'].append(False)

                    print(f"current_step:{current_step}/total_step:{cfg.total_steps}")

                # 如果需要终止，在保存数据后再break
                if should_terminate:
                    break

            counter += 1
            if cfg.render:
                viewer.render()

        # 更新统计信息
        termination_stats['total_episodes'] += 1
        if episode_terminated_early:
            # 这里可以进一步细分termination原因
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

    if not cfg.render and not cfg.test:
        # 拼接所有list为ndarray
        result = {}
        for k in motions_for_saving:
            if k != 'fps':  # fps is not a list
                result[k] = np.array(motions_for_saving[k])
        result['fps'] = motions_for_saving['fps']

        # 按照期望的方式保存数据
        save_f = open(save_path, 'wb')
        pickle.dump(result, save_f)
        save_f.close()

        print(f"✅ Motion data saved to: {save_path}")
        print(f"📊 Total frames saved: {len(result['motion_times'])}")
        print(f"⏱️ Total duration: {result['motion_times'][-1]:.2f}s")
        print(f"🎬 FPS: {result['fps']:.1f}")


if __name__ == '__main__':
    from datetime import datetime

    current_directory = os.getcwd()
    config_file = current_directory + "/g1_config/mujoco_config_p.yaml"
    cfg = read_conf(config_file)

    # 生成带时间戳的文件名，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(current_directory, f"{timestamp}_mujoco_motion_Hooks_punch.pkl")
    print(f"数据将保存到: {save_path}")

    run_and_save_mujoco(cfg, save_path)
    print("-----done------")