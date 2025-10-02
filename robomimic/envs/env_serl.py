"""
This file contains the Franka environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import sys
from copy import deepcopy

import numpy as np
import requests

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB

# Import the original FrankaEnv and config
# sys.path.append("/workspace/serl_robot_infra")
# from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig


class EnvSERL(EB.EnvBase):
    """Wrapper class for Franka environments"""

    def __init__(
        self,
        env_name="FrankaEnv",
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
        config=None,
        hz=10,
        fake_env=False,
        save_video=False,
        set_load=False,
        open_threads=True,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment
            render (bool): if True, environment supports on-screen rendering
            render_offscreen (bool): if True, environment supports off-screen rendering
            use_image_obs (bool): if True, environment is expected to render rgb image
            observations
            use_depth_obs (bool): if True, environment is expected to render depth image
            observations
            config: FrankaEnv configuration object
            hz (int): control frequency
            fake_env (bool): if True, creates a fake environment for testing
            save_video (bool): if True, saves video recordings
            set_load (bool): if True, sets load parameters
            open_threads (bool): if True, opens camera threads
        """
        self._env_name = env_name
        self.use_image_obs = use_image_obs
        self.use_depth_obs = use_depth_obs

        # Set default config if none provided
        if config is None:
            config = DefaultEnvConfig()

        # Update config based on image observation requirements
        if use_image_obs and not hasattr(config, 'IMAGE_KEYS'):
            config.IMAGE_KEYS = list(config.REALSENSE_CAMERAS.keys()) + list(config.GENERIC_CAMERAS.keys())
        elif not use_image_obs:
            config.IMAGE_KEYS = []

        self.config = config
        self._init_kwargs = {
            'hz': hz,
            'fake_env': fake_env,
            'save_video': save_video,
            'config': config,
            'set_load': set_load,
            'open_threads': open_threads,
            **kwargs
        }

        # Create the underlying Franka environment
        self.env = FrankaEnv(
            hz=hz,
            fake_env=fake_env,
            save_video=save_video,
            config=config,
            set_load=set_load,
            open_threads=open_threads
        )

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, reward, done, _, info = self.env.step(action)

        # Convert observation to robomimic format
        obs = self.get_observation(obs)

        # Convert reward to float
        reward = float(reward)

        # Add success information
        info["is_success"] = self.is_success()

        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        obs, _ = self.env.reset(**kwargs)
        return self.get_observation(obs)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state

        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        # For now, just do a regular reset since Franka doesn't support state restoration
        # This could be extended to support specific pose resets
        if "pose" in state:
            # Could implement pose-specific reset here
            pass
        return self.reset()

    def render(self, mode="human", height=None, width=None, camera_name=None):
        """
        Render from simulation.

        Args:
            mode (str): render mode
            height (int): height of rendered image
            width (int): width of rendered image
            camera_name (str): camera name for rendering
        """
        if mode == "rgb_array":
            # Get image observations
            images = self.env.get_im()
            if camera_name and camera_name in images:
                return images[camera_name]
            elif images:
                # Return first available image
                return list(images.values())[0]
            else:
                # Return dummy image if no cameras
                return np.zeros((height or 128, width or 128, 3), dtype=np.uint8)
        else:
            # For human mode, Franka handles its own display
            return None

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            obs (dict): raw observation from Franka environment
        """
        if obs is None:
            obs = self.env._get_obs()

        ret = {}

        # Process state observations
        if "state" in obs:
            state = obs["state"]

            # Add TCP pose (position + quaternion)
            if "tcp_pose" in state:
                ret["robot0_eef_pos"] = state["tcp_pose"][:3]
                ret["robot0_eef_quat"] = state["tcp_pose"][3:]

            # Add TCP velocity
            if "tcp_vel" in state:
                ret["robot0_eef_vel"] = state["tcp_vel"]

            # Add gripper state
            if "gripper_pose" in state:
                ret["robot0_gripper_qpos"] = state["gripper_pose"]

            # Add force/torque observations
            if "tcp_force" in state:
                ret["robot0_tcp_force"] = state["tcp_force"]
            if "tcp_torque" in state:
                ret["robot0_tcp_torque"] = state["tcp_torque"]

            # Add base policy goal if available (for residual learning)
            if "base_policy_goal" in state:
                ret["base_policy_goal"] = state["base_policy_goal"]

        # Process image observations
        if "images" in obs and self.use_image_obs:
            for camera_name, image in obs["images"].items():
                # Convert from BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    ret[f"{camera_name}_image"] = image[..., ::-1]  # BGR to RGB
                else:
                    ret[f"{camera_name}_image"] = image

        return ret

    def get_state(self):
        """
        Get current environment simulator state.
        """
        # For Franka, we can save the current pose and gripper state
        self.env._update_currpos()
        state = {
            "pose": self.env.currpos.copy(),
            "gripper_pos": self.env.curr_gripper_pos.copy(),
            "velocity": self.env.currvel.copy(),
        }
        return state

    def get_reward(self):
        """
        Get current reward.
        """
        obs = self.env._get_obs()
        return float(self.env.compute_reward(obs))

    def get_goal(self):
        """
        Get goal observation. Returns target pose.
        """
        target_pose = self.config.REAL_TARGET_POSE
        return {"target_pose": target_pose}

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification.
        """
        if "target_pose" in kwargs:
            self.config.REAL_TARGET_POSE = kwargs["target_pose"]
        return True

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        # Check if we've reached max episode length or if termination was triggered
        return (self.env.curr_path_length >= self.env.max_episode_length or
                getattr(self.env, 'terminate', False))

    def is_success(self):
        """
        Check if the task condition(s) is reached.
        """
        obs = self.env._get_obs()
        success = self.env.compute_reward(obs)
        return {"task": bool(success)}

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_space.shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        """
        return EB.EnvType.FRANKA_TYPE

    @property
    def version(self):
        """
        Returns version of Franka environment.
        """
        return "1.0.0"

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(
        cls,
        camera_names=None,
        camera_height=128,
        camera_width=128,
        reward_shaping=True,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=False,
        env_name="FrankaEnv",
        config=None,
        **kwargs,
    ):
        """
        Create environment for processing datasets.
        """
        if camera_names is None:
            camera_names = []
        has_camera = len(camera_names) > 0

        # Set up config with camera information
        if config is None:
            config = DefaultEnvConfig()

        # Configure image keys based on camera names
        if has_camera:
            config.IMAGE_KEYS = camera_names
        else:
            config.IMAGE_KEYS = []

        # Initialize obs utils with modality specifications
        image_modalities = [f"{cn}_image" for cn in camera_names] if has_camera else []
        depth_modalities = [f"{cn}_depth" for cn in camera_names] if (has_camera and use_depth_obs) else []

        obs_modality_specs = {
            "obs": {
                "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "rgb": image_modalities,
            }
        }
        if use_depth_obs:
            obs_modality_specs["obs"]["depth"] = depth_modalities

        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(
            env_name=env_name,
            render=(False if render is None else render),
            render_offscreen=(has_camera if render_offscreen is None else render_offscreen),
            use_image_obs=(has_camera if use_image_obs is None else use_image_obs),
            use_depth_obs=use_depth_obs,
            config=config,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to handle during rollouts.
        """
        return (requests.exceptions.RequestException, ConnectionError, TimeoutError)

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def close(self):
        """
        Clean up environment resources.
        """
        if hasattr(self.env, 'close'):
            self.env.close()

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return (
            self.name + "\n" +
            json.dumps(self._init_kwargs, sort_keys=True, indent=4, default=str)
        )
