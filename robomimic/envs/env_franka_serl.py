"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy
import sys

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

import rospy
from sensor_msgs.msg import Image, JointState, CompressedImage
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from cv_bridge import CvBridge
from gazebo_msgs.srv import SetModelState, DeleteModel, SpawnModel, SetModelConfiguration
import pytransform3d.rotations as pr
from controller_manager_msgs.srv import SwitchController
from actionlib import SimpleActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, \
                             FollowJointTrajectoryGoal, FollowJointTrajectoryResult
from geometry_msgs.msg import  Twist
import tf.transformations
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import random
from scipy.spatial.transform import Rotation as R
import cv2
from franka_msgs.msg import ErrorRecoveryActionGoal

class EnvFrankaSERL(EB.EnvBase):
    """Wrapper class for gym"""
    def __init__(
        self,
        env_name,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): ignored - gym envs always support on-screen rendering

            render_offscreen (bool): ignored - gym envs always support off-screen rendering

            use_image_obs (bool): ignored - gym envs don't typically use images
        """
        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._done = None

        # Initialize ROS node if not already initialized
        if not rospy.core.is_initialized():
            rospy.init_node('robomimic_gazebo_env', anonymous=True)

        # Storage for observation data
        self._fr3_camera_image = None
        self._external_camera_image = None
        self._franka_state = None
        self._gripper_joint_state = None

        self.x_range = (0.4, 0.6)
        self.y_range = (-0.2, 0.2)
        self.z_fixed = 0.035/2
        self.roll_fixed = 0
        self.pitch_fixed = 0
        self.yaw_range = (-np.pi/4, np.pi/4)
        self.block_pose = None

        self._gripper_open = True

        # CV Bridge for image conversion
        self._bridge = CvBridge()

        # Subscribe to observation topics
        self._fr3_camera_sub = rospy.Subscriber(
            '/hand/camera/color/reshaped/compressed',
            CompressedImage,
            self._fr3_camera_callback
        )
        self._external_camera_sub = rospy.Subscriber(
            '/ext/camera/color/reshaped/compressed',
            CompressedImage,
            self._external_camera_callback
        )
        self._franka_state_sub = rospy.Subscriber(
            '/franka_state_controller/franka_states',
            FrankaState,
            self._franka_state_callback
        )
        self._gripper_joint_state_sub = rospy.Subscriber(
            '/franka_gripper/joint_states',
            JointState,
            self._gripper_joint_state_callback
        )

        # Publishers for action topics
        self._equilibrium_pose_pub = rospy.Publisher(
            '/cartesian_impedance_controller/equilibrium_pose',
            PoseStamped,
            queue_size=10
        )
        self._gripper_grasp_pub = rospy.Publisher(
            '/franka_gripper/grasp/goal',
            GraspActionGoal,
            queue_size=10
        )

        self._gripper_move_pub = rospy.Publisher(
            '/franka_gripper/move/goal',
            MoveActionGoal,
            queue_size=10
        )

        self._error_recovery_pub = rospy.Publisher(
            '/franka_control/error_recovery/goal',
            ErrorRecoveryActionGoal,
            queue_size=10

        )

        # Service clients for Gazebo
        # rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        # rospy.wait_for_service('/gazebo/delete_model', timeout=10)
        # rospy.wait_for_service('/gazebo/spawn_urdf_model', timeout=10)
        # rospy.wait_for_service('/gazebo/set_model_configuration', timeout=10)
        
        # self.set_model_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # self.delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        # self.spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        # self.set_model_config_client = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        
        # Controller manager service
        rospy.wait_for_service('/controller_manager/switch_controller', timeout=10)
        self.switch_controller_client = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

        # Get arm_id parameter (usually 'fr3' or 'panda')
        # self.arm_id = rospy.get_param('~arm_id', 'fr3')
        
        # # Default joint positions from launch file
        # self.default_joint_positions = {
        #     f'{self.arm_id}_joint1': 0.0,
        #     f'{self.arm_id}_joint2': -0.785398163,  # -45 degrees
        #     f'{self.arm_id}_joint3': 0.0,
        #     f'{self.arm_id}_joint4': -2.35619449,   # -135 degrees
        #     f'{self.arm_id}_joint5': 0.0,
        #     f'{self.arm_id}_joint6': 1.57079632679, # 90 degrees
        #     f'{self.arm_id}_joint7': 0.785398163397, # 45 degrees
        #     f'{self.arm_id}_finger_joint1': 0.001,
        #     f'{self.arm_id}_finger_joint2': 0.001
        # }

        action = rospy.resolve_name('/effort_joint_trajectory_controller/follow_joint_trajectory/')
        self.joint_traj_client = SimpleActionClient(action, FollowJointTrajectoryAction)
        rospy.loginfo("move_to_start: Waiting for '" + action + "' action to come up")
        self.joint_traj_client.wait_for_server()

        # Service clients
        # rospy.wait_for_service('/gazebo/set_model_state', timeout=30)
        # self.set_model_state_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Wait for initial observations
        rospy.sleep(0.5)

    def _fr3_camera_callback(self, msg):
        # print("Got image!!", msg.encoding)
        """Callback for FR3 camera images."""
        # # Convert ROS Image message to numpy array directly
        # if msg.encoding == 'rgb8':
        #     self._fr3_camera_image = self._bridge.compressed_imgmsg_to_cv2(msg).reshape(msg.height, msg.width, 3)
        # elif msg.encoding == 'bgr8':
        img = self._bridge.compressed_imgmsg_to_cv2(msg) #.reshape(msg.height, msg.width, 3)
        self._fr3_camera_image = img[:, :, ::-1]  # Convert BGR to RGB
        # else:
        #     # For other encodings, keep as is
        #     self._fr3_camera_image = self._bridge.compressed_imgmsg_to_cv2(msg).reshape(msg.height, msg.width, -1)

    def _external_camera_callback(self, msg):
        """Callback for external camera images."""
        # Convert ROS Image message to numpy array directly
        # if msg.encoding == 'rgb8':
        #     self._external_camera_image = self._bridge.compressed_imgmsg_to_cv2(msg).reshape(msg.height, msg.width, 3)
        # elif msg.encoding == 'bgr8':
        img = self._bridge.compressed_imgmsg_to_cv2(msg) #.reshape(msg.height, msg.width, 3)
        self._external_camera_image = img[:, :, ::-1]  # Convert BGR to RGB
        # else:
        #     # For other encodings, keep as is
        #     self._external_camera_image = self._bridge.compressed_imgmsg_to_cv2(msg).reshape(msg.height, msg.width, -1)

    def _franka_state_callback(self, msg):
        """Callback for Franka state."""
        self._franka_state = msg

    def _gripper_joint_state_callback(self, msg):
        """Callback for gripper joint states."""
        self._gripper_joint_state = msg


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
        self._error_recovery_pub.publish(ErrorRecoveryActionGoal())

        # Publish action to equilibrium pose topic
        pose_msg = PoseStamped()

        # Assuming action is [x, y, z, qx, qy, qz, qw, gripper_width, gripper_force]
        if len(action) >= 7:
            pose_msg.pose.position.x = action[0]
            pose_msg.pose.position.y = action[1]
            pose_msg.pose.position.z = action[2]
            pose_msg.pose.orientation.w = action[3]
            pose_msg.pose.orientation.x = action[4]
            pose_msg.pose.orientation.y = action[5]
            pose_msg.pose.orientation.z = action[6]

        self._equilibrium_pose_pub.publish(pose_msg)
        # Control gripper
        if action[7] <= 0.04 and self._gripper_open:  # Assuming a threshold for closing the gripper
            print("Closing gripper")
            grasp_msg = GraspActionGoal()
            grasp_msg.goal.width = 0.0  # Desired gripper width
            grasp_msg.goal.force = 60 # Grasping force
            grasp_msg.goal.speed = 0.1  # Default speed
            grasp_msg.goal.epsilon.inner = 0.2
            grasp_msg.goal.epsilon.outer = 0.2
            self._gripper_grasp_pub.publish(grasp_msg)
            self._gripper_open = False
        elif action[7] >= 0.045 and not self._gripper_open:  # Assuming a threshold for opening the gripper
            print("Opening gripper")
            grasp_msg = MoveActionGoal()
            grasp_msg.goal.width = action[7]  # Desired gripper width
            grasp_msg.goal.speed = 0.2  # Default speed
            self._gripper_move_pub.publish(grasp_msg)
            self._gripper_open = True

        rospy.sleep(0.1)

        # Read current observations from topics
        obs = self._get_obs_dict()
        reward = 0.0  # Compute reward based on your task
        done = False  # Check if episode is done based on your task
        info = {}

        self._current_obs = obs
        self._current_reward = reward
        self._current_done = done
        return self.get_observation(obs), reward, self.is_done(), info

    def _get_obs_dict(self):
        """
        Read current observations from ROS topics.

        Returns:
            obs (dict): observation dictionary containing all sensor data
        """
        obs = {}

        if self._fr3_camera_image is not None:
            obs['fr3_camera_image'] = cv2.resize(self._fr3_camera_image, (84, 84))

        if self._external_camera_image is not None:
            obs['external_camera_image'] = cv2.resize(self._external_camera_image, (84, 84))

        if self._franka_state is not None:
            # Extract relevant state information
            tf = np.array(self._franka_state.O_T_EE).reshape((4, 4), order='F')

            obs['robot_ee_pos'] = tf[:3, 3]
            obs['robot_ee_quat'] = pr.quaternion_from_matrix(tf[:3, :3], strict_check=False)

        if self._gripper_joint_state is not None:
            # obs['gripper_position'] = np.array(self._gripper_joint_state.position)
            obs['gripper_position'] = np.reshape(np.sum(self._gripper_joint_state.position)/0.05, [1])
        return obs
    
    def randomize_block_pose(self):
        """Generate random block position within specified range"""
        x = random.uniform(*self.x_range)
        y = random.uniform(*self.y_range)
        z = self.z_fixed
        roll = self.roll_fixed
        pitch = self.pitch_fixed
        yaw = random.uniform(*self.yaw_range)
        
        return x, y, z, roll, pitch, yaw
    
    def set_block_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Set block to specified position"""
        try:
            block_state = ModelState()
            block_state.model_name = 'pick_block'
            
            # Set position
            block_state.pose.position.x = x
            block_state.pose.position.y = y
            block_state.pose.position.z = z
            
            # Convert RPY to quaternion
            quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            block_state.pose.orientation.x = quaternion[0]
            block_state.pose.orientation.y = quaternion[1]
            block_state.pose.orientation.z = quaternion[2]
            block_state.pose.orientation.w = quaternion[3]
            
            # Zero velocity
            block_state.twist = Twist()
            
            # response = self.set_model_state_client(block_state)
            
            # if response.success:
            #     rospy.loginfo(f"Block positioned at ({x:.3f}, {y:.3f}, {z:.3f})")
            #     return True
            # else:
            #     rospy.logerr(f"Failed to set block position: {response.status_message}")
            #     return False
                
        except Exception as e:
            rospy.logerr(f"Error setting block position: {e}")
            return False

    def move_to_start(self):
        """
        Move robot to starting configuration.
        """
        param = rospy.resolve_name('/joint_pose')
        pose = rospy.get_param(param, None)
        if pose is None:
            rospy.logerr('move_to_start: Could not find required parameter "' + param + '"')
            sys.exit(1)

        topic = rospy.resolve_name('/joint_states')
        rospy.loginfo("move_to_start: Waiting for message on topic '" + topic + "'")
        joint_state = rospy.wait_for_message(topic, JointState)
        initial_pose = dict(zip(joint_state.name, joint_state.position))

        max_movement = max(abs(pose[joint] - initial_pose[joint]) for joint in pose)

        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration.from_sec(
            # Use either the time to move the furthest joint with 'max_dq' or 500ms,
            # whatever is greater
            max(max_movement / rospy.get_param('~max_dq', 0.2), 2)
        )
        goal = FollowJointTrajectoryGoal()

        goal.trajectory.joint_names, point.positions = [list(x) for x in zip(*pose.items())]
        point.velocities = [0] * len(pose)

        goal.trajectory.points.append(point)
        goal.goal_time_tolerance = rospy.Duration.from_sec(2)

        rospy.loginfo('Sending trajectory Goal to move into initial config')
        # import pdb; pdb.set_trace()
        self.joint_traj_client.send_goal_and_wait(goal)

        result = self.joint_traj_client.get_result()
        if result.error_code != FollowJointTrajectoryResult.SUCCESSFUL:
            rospy.logerr('move_to_start: Movement was not successful: ' + {
                FollowJointTrajectoryResult.INVALID_GOAL:
                """
                The joint pose you want to move to is invalid (e.g. unreachable, singularity...).
                Is the 'joint_pose' reachable?
                """,

                FollowJointTrajectoryResult.INVALID_JOINTS:
                """
                The joint pose you specified is for different joints than the joint trajectory controller
                is claiming. Does you 'joint_pose' include all 7 joints of the robot?
                """,

                FollowJointTrajectoryResult.PATH_TOLERANCE_VIOLATED:
                """
                During the motion the robot deviated from the planned path too much. Is something blocking
                the robot?
                """,

                FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED:
                """
                After the motion the robot deviated from the desired goal pose too much. Probably the robot
                didn't reach the joint_pose properly
                """,
            }[result.error_code])

        else:
            rospy.loginfo('move_to_start: Successfully moved into start pose')

    def reset_robot_joints(self):
        """Reset robot joints to default configuration"""
        try:
            rospy.loginfo("Switching to effort_joint_trajectory_controller")

            # Stop current controller
            self.switch_controller_client(
                stop_controllers=['cartesian_impedance_controller'],
                start_controllers=['effort_joint_trajectory_controller'],
                strictness=2,
                timeout=0,
                start_asap=False
            )
            
            rospy.sleep(1)  # Wait for controller to stop
            
            rospy.loginfo("Switched to effort_joint_trajectory_controller")

            # Prepare joint names and positions for Gazebo service
            # joint_names = list(self.default_joint_positions.keys())
            # joint_positions = list(self.default_joint_positions.values())
            
            # # Reset joint configuration
            # response = self.set_model_config_client(
            #     model_name=self.arm_id,
            #     urdf_param_name='robot_description',
            #     joint_names=joint_names,
            #     joint_positions=joint_positions
            # )
            
            # if not response.success:
            #     rospy.logwarn(f"Failed to reset joint configuration: {response.status_message}")
            #     return False
            
            # rospy.sleep(3)  # Wait for joints to settle

            # pose_msg = PoseStamped()

            # pose_msg.pose.position.x = self._current_obs['robot_ee_pos'][0]
            # pose_msg.pose.position.y = self._current_obs['robot_ee_pos'][1]
            # pose_msg.pose.position.z = self._current_obs['robot_ee_pos'][2]
            # pose_msg.pose.orientation.w = self._current_obs['robot_ee_quat'][0]
            # pose_msg.pose.orientation.x = self._current_obs['robot_ee_quat'][1]
            # pose_msg.pose.orientation.y = self._current_obs['robot_ee_quat'][2]
            # pose_msg.pose.orientation.z = self._current_obs['robot_ee_quat'][3]

            # self._equilibrium_pose_pub.publish(pose_msg)
                 
            self.move_to_start()

            # Restart controller
            self.switch_controller_client(
                stop_controllers=['effort_joint_trajectory_controller'],
                start_controllers=['cartesian_impedance_controller'],
                strictness=2,
                timeout=0,
                start_asap=False
            )


            # rospy.loginfo("Robot joints reset to default configuration")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to reset robot joints: {e}")
            # Try to restart controller even if joint reset failed
            try:
                self.switch_controller_client(
                    stop_controllers=['effort_joint_trajectory_controller'],
                    start_controllers=['cartesian_impedance_controller'],
                    strictness=2,
                    timeout=0,
                    start_asap=False
                )
            except:
                pass
            return False

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """


        self._error_recovery_pub.publish(ErrorRecoveryActionGoal())        


        # grasp_msg = MoveActionGoal()
        # grasp_msg.goal.width = 0.05  # Desired gripper width
        # grasp_msg.goal.speed = 0.2  # Default speed
        # self._gripper_move_pub.publish(grasp_msg)

        # self._gripper_open = True

        self._current_obs = self._get_obs_dict()

        pose_msg = PoseStamped()

        # Assuming action is [x, y, z, qx, qy, qz, qw, gripper_width, gripper_force]
        pose_msg.pose.position.x = self._current_obs["robot_ee_pos"][0]
        pose_msg.pose.position.y = self._current_obs["robot_ee_pos"][1]
        pose_msg.pose.position.z = self._current_obs["robot_ee_pos"][2] + 0.05
        pose_msg.pose.orientation.w = self._current_obs["robot_ee_quat"][0]
        pose_msg.pose.orientation.x = self._current_obs["robot_ee_quat"][1]
        pose_msg.pose.orientation.y = self._current_obs["robot_ee_quat"][2]
        pose_msg.pose.orientation.z = self._current_obs["robot_ee_quat"][3]

        self._equilibrium_pose_pub.publish(pose_msg)

        rospy.sleep(1)

        pose_msg = PoseStamped()

        # Assuming action is [x, y, z, qx, qy, qz, qw, gripper_width, gripper_force]
        pose_msg.pose.position.x = 0.58014052 + np.random.uniform(-0.02, 0.02)
        pose_msg.pose.position.y = -0.01655394 + np.random.uniform(-0.02, 0.02)
        pose_msg.pose.position.z = 0.05100882
        pose_msg.pose.orientation.w = -0.01210018
        pose_msg.pose.orientation.x = 0.99975015
        pose_msg.pose.orientation.y = 0.01759502
        pose_msg.pose.orientation.z = 0.00623104

        self._equilibrium_pose_pub.publish(pose_msg)

        rospy.sleep(2)

        grasp_msg = MoveActionGoal()
        grasp_msg.goal.width = 0.05  # Desired gripper width
        grasp_msg.goal.speed = 0.2  # Default speed
        self._gripper_move_pub.publish(grasp_msg)

        self._gripper_open = True
        rospy.sleep(1)


        self.reset_robot_joints()

        # Wait for fresh observations
        # rospy.sleep(1)

        self.block_pose = self.randomize_block_pose()

        self.set_block_pose(*self.block_pose)

        # Read current observations from topics
        self._current_obs = self._get_obs_dict()
        self._current_reward = None
        self._current_done = None
        return self.get_observation(self._current_obs)

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode == "rgb_array":
            # return self._current_obs[camera_name]
            return getattr(self, f"_{camera_name}", None)
        
            if self._external_camera_image is not None:
                return self._external_camera_image
            else:
                raise RuntimeError("No external camera image available for rgb_array rendering")

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            obs (dict): current observation dictionary from ROS topics.
                If not provided, uses self._current_obs.
        """
        if obs is None:
            assert self._current_obs is not None
            obs = self._current_obs
        return obs

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """

        # gym envs generally don't check task success - we only compare returns
        return { "task" : False }

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return 

    @classmethod
    def create_for_data_processing(
        cls, 
    ):
        return
    
    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        return dict(states=False, model=False)

    def get_reward(self):
        """
        Get current reward.
        """
        return 
    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return 

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        return
    
    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return 

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 8  # [x, y, z, qx, qy, qz, qw, gripper_width, gripper_force]

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
        This helps identify this env class.
        """
        return EB.EnvType.SERL_TYPE
    
    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))


    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
