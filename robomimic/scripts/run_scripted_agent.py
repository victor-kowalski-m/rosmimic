"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.env_gazebo import EnvGazebo
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from scipy.spatial.transform import Rotation as R, Slerp
from pynput import keyboard

success_key = False
def on_press(key):
    global success_key
    try:
        if str(key) == "'e'":
            print("success key pressed")
            success_key = True
    except AttributeError:
        pass

def quat_w_first_to_last(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])

def quat_w_last_to_first(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])

def calc_line(start, goal, len_t=50):
    # Receives list of [x, y, z, qw, qx, qy, qz]
    # Returns list of [x, y, z, qw, qx, qy, qz]

    # Convert start and goal to x, y, z, qx, qy, qz, qw
    start = np.array([start[0], start[1], start[2], start[4], start[5], start[6], start[3]])
    goal = np.array([goal[0], goal[1], goal[2], goal[4], goal[5], goal[6], goal[3]])
    
    vec = goal[:3] - start[:3]
    t = np.linspace(0, 1, len_t)
    points = (vec[..., np.newaxis] * t).T + start[:3]
    orients = np.array([start[3:], goal[3:]])
    orients = R.from_quat(orients)
    sl = Slerp([0, 1], orients)
    orients = sl(t).as_quat()
    orients = [np.concatenate((np.array([robot_quat[-1]]), robot_quat[:3])) for robot_quat in orients]  # to w, x, y, z
    return np.concatenate([points, orients], axis=1)

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, teleop_device=None, keyboard_listener=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    global success_key

    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    # assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    # policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    # obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    # try:

    if teleop_device is None:
        robot_quat = ( 
                R.from_euler("xyz", env.block_pose[3:]) *
                R.from_quat([1.0, 0.0, 0.0, 0.0])
            ).as_quat()
        robot_quat = np.concatenate((np.array([robot_quat[-1]]), robot_quat[:3]))  # to w, x, y, z
        approach_t = 20
        pick_t = 20
        pause_pick_t = 10
        grasp_t = 20
        lift_t = 20 
        approach_poses = calc_line(
            [*obs["robot_ee_pos"], *obs["robot_ee_quat"]],
            [*env.block_pose[:3]+ np.array([0, 0, 0.1]), *robot_quat],
            approach_t
        )
        pick_poses = calc_line(
            [*env.block_pose[:3]+ np.array([0, 0, 0.1]), *robot_quat],
            [*env.block_pose[:3]+ np.array([0, 0, -0.01]), *robot_quat],
            pick_t
        )
        lift_poses = calc_line(
            [*env.block_pose[:3]+ np.array([0, 0, -0.01]), *robot_quat],
            [*env.block_pose[:3]+ np.array([0, 0, 0.1]), *robot_quat],
            lift_t
        )
    else:
        sensitivity = np.array([0.1]*3 + [0.5]*3)
        last_gripper_cmd = [0.08]  # open

    success_key = False

    for step_i in range(horizon):

        if teleop_device is not None:
            tdc = teleop_device.control
            teleop_cmd = [tdc[0], tdc[1], tdc[2], tdc[4], -tdc[3], tdc[5]]
            gripper_cmd = [0.08] if tdc[6]==1 else [0.00] if tdc[6]==2 else last_gripper_cmd
            last_gripper_cmd = gripper_cmd

            # Get the newest action
            action = teleop_cmd * sensitivity

            robot_quat = (
                R.from_quat(quat_w_first_to_last(obs["robot_ee_quat"])) *
                R.from_euler("xyz", action[3:])
            ).as_quat()
            robot_quat = np.concatenate((np.array([robot_quat[-1]]), robot_quat[:3]))  # to w, x, y, z

            act = np.concatenate((obs["robot_ee_pos"] + action[:3], robot_quat, gripper_cmd))
        else:
            if step_i < approach_t:
                print("approaching")
                act = np.concatenate((approach_poses[step_i], np.array([0.08])))
            elif step_i < approach_t + pick_t:
                print("picking")
                act = np.concatenate((pick_poses[step_i - approach_t], np.array([0.08])))
            elif step_i < approach_t + pick_t + pause_pick_t:
                print("pausing at pick")
                act = np.concatenate((pick_poses[-1], np.array([0.08])))
            elif step_i < approach_t + pick_t + pause_pick_t + grasp_t:
                print("grasping")
                act = np.concatenate((pick_poses[-1], np.array([0.0])))
            elif step_i < approach_t + pick_t + pause_pick_t + grasp_t + lift_t:
                print("lifting")
                act = np.concatenate((lift_poses[step_i - approach_t - pick_t - pause_pick_t  - grasp_t], np.array([0.0])))
            else:
                act = np.concatenate((lift_poses[-1], np.array([0.0])))

        # play action
        next_obs, r, done, _ = env.step(act)

        # compute reward
        total_reward += r
        success = env.is_success()["task"]

        # # visualization
        # if render:
        #     env.render(mode="human", camera_name=camera_names[0])
        if video_writer is not None:
            print("step", step_i)
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        # collect transition
        traj["actions"].append(act)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["states"].append(state_dict["states"])
        if return_obs:
            traj["obs"].append(obs)
            traj["next_obs"].append(next_obs)

        # break if done or if success
        if done or success:
            break

        if success_key:
            print("success key pressed, terminating episode with success")
            success_key = False
            break

        # update for next iter
        obs = deepcopy(next_obs)
        state_dict = env.get_state()

    # except env.rollout_exceptions as e:
    #     print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    # ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # # restore policy
    # policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    # if rollout_horizon is None:
        # read horizon from config
        # config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        # rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    # env, _ = FileUtils.env_from_checkpoint(
    #     ckpt_dict=ckpt_dict, 
    #     env_name=args.env, 
    #     render=args.render, 
    #     render_offscreen=(args.video_path is not None), 
    #     verbose=True,
    # )
    env = EnvGazebo("PickBlock")

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=10)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    if args.motion_mode == "spacemouse":
        from robomimic.utils.spacemouse import SpaceMouse

        teleop_device = SpaceMouse(
            env=env,
            vendor_id=0x256f,
            product_id=0xc635,
            # pos_sensitivity=1,#args.pos_sensitivity,
            # rot_sensitivity=1,#args.rot_sensitivity,
        )
        teleop_device.start_control()

    global success_key
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    rollout_stats = []
    for i in range(rollout_num_episodes):
        print("********** ROLLOUT {} **********".format(i))
        stats, traj = rollout(
            policy=None, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
            teleop_device=teleop_device if args.motion_mode == "spacemouse" else None,
            keyboard_listener=listener
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            # import pdb; pdb.set_trace()
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=10,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=600,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["fr3_camera_image", "external_camera_image"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    parser.add_argument(
        "--motion_mode",
        type=str,
        default="spacemouse",
        help="'scripted' (robot follows programmed path) or 'spacemouse' (3D mouse)",
    )

    args = parser.parse_args()
    run_trained_agent(args)

