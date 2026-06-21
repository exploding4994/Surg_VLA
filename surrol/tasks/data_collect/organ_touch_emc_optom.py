import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
import cv2
from surrol.utils.robotics import get_matrix_from_euler


class NeedlePick(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    # Default joint positions for ECM
    QPOS_ECM = (0., 0.9, 0.2, 0.)  # Adjust as needed

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid=-1):
        super(NeedlePick, self).__init__(render_mode, cid)
        self.ecm = None  # Initialize ECM as None

    def _env_setup(self):
        super(NeedlePick, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
        p.configureDebugVisualizer(lightPosition=(0.55, 0.0, 5.3))  # GUI light source

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0].mean() + 0.2 * 2 * (np.random.rand() - 0.5),
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1

        # self._setup_soft_tissue(workspace_limits)

        # gauze
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
        #                     (workspace_limits[0].mean() + 3*(np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() + 4*(np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     (0, 0, 0, 1),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        # self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1
        rd = (np.random.rand() - 0.5) * 0.1 * 3
        rd2 = (np.random.rand() - 0.5) * 0.1 * 2

        # Red sphere
        # yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
            (workspace_limits[0].mean() + 0.11 + rd,
             workspace_limits[1].mean() + 0.13 + rd2,
             workspace_limits[2][0] + 0.07),
            p.getQuaternionFromEuler((0, 0, 0)),
            useFixedBase=True,
            globalScaling=4
        )

        # ✅ Correctly set the color for the target sphere
        p.changeVisualShape(obj_id, -1, rgbaColor=(1.0, 0.0, 0.0, 0.4))
        # Optional: Retain or adjust specular highlights
        p.changeVisualShape(obj_id, -1, specularColor=(0.3, 0.3, 0.3))  # Softer highlights

        # Kidney
        # yaw = (np.random.rand() - 0.5) * np.pi

        obj_id2 = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'keydney/keydney.urdf'),
            (workspace_limits[0].mean() + 0.5 + rd,
             workspace_limits[1].mean() + rd2,
             workspace_limits[2][0] - 0.14),
            p.getQuaternionFromEuler((np.pi / 2 + np.pi / 9, 0, np.pi / 2)),
            useFixedBase=True,
            globalScaling=2
        )

        # ✅ Correctly disable all collisions between PSM (body 1) and kidney (obj_id2)
        psm_id = 1
        keydney_id = obj_id2

        # Disable collision between base links
        p.setCollisionFilterPair(psm_id, keydney_id, -1, -1, enableCollision=0)

        # Disable collision between each active PSM link and the kidney base (Critical!)
        for i in range(p.getNumJoints(psm_id)):
            p.setCollisionFilterPair(psm_id, keydney_id, i, -1, enableCollision=0)

        # ✅ Correctly set the color for the kidney model
        p.changeVisualShape(obj_id2, -1, rgbaColor=(0.65, 0.45, 0.45, 1.0))
        # Optional: Retain or adjust specular highlights
        p.changeVisualShape(obj_id2, -1, specularColor=(0.25, 0.25, 0.25))  # Softer highlights
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        # --- 4. Initialize ECM Endoscope ---
        self.ecm = Ecm(
            (0.2, 0., 0.8),  # ECM base position
            scaling=self.SCALING
        )
        self.ecm.reset_joint(self.QPOS_ECM)  # Set initial posture

    def get_ecm_image(self, image_width=640, image_height=480):
        """
        Retrieve RGB and depth images from the ECM camera perspective.
        """
        # Render image (internally sets viewMatrix and projMatrix)
        self.ecm.render_image()

        # Retrieve image data
        _, _, rgb_image, depth_image, mask = p.getCameraImage(
            width=image_width,
            height=image_height,
            viewMatrix=self.ecm.view_matrix,
            projectionMatrix=self.ecm.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # High quality OpenGL rendering
        )

        # Convert depth map
        near, far = 0.02, 1.0
        depth = far * near / (far - (far - near) * depth_image)

        # Convert OpenCV format: RGB -> BGR
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return rgb_image, depth, mask

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        print('[Env] Goal has been set.')
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        # super()._sample_goal_callback()
        self._waypoints = [None, None]  # two waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        self._waypoints[0] = np.array([pos_obj[0] + 0.02, pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0252 + 0.005) * self.SCALING, 0, -0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0] + 0.0, pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0, -0.5])  # approach

        print(self.goal)

        self._steps_per_waypoint = [20, 20, 5, 10]
        self._current_waypoint_index = 0
        self._step_in_waypoint = 0
        self._waypoint_start_state = None

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # sequential waypoint execution
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action

    def _meet_contact_constraint_requirement(self):
        if self._contact_approx:
            return True
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def is_action_completed(self, obs, action, pos_tolerance=0.005, yaw_tolerance=0.1, gripper_tolerance=0.2):
        """
        Check if the current action/waypoint has been completed.
        """
        # If all waypoints are done, return True
        if self._current_waypoint_index >= len(self._waypoints):
            return True

        # Retrieve the target state of the current waypoint
        current_waypoint = self._waypoints[self._current_waypoint_index]

        # If the current waypoint is None, move to the next one
        if current_waypoint is None:
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
            return self.is_action_completed(obs, action, pos_tolerance, yaw_tolerance, gripper_tolerance)

        # Extract target states
        target_pos = current_waypoint[:3]
        target_yaw = current_waypoint[3]
        target_gripper = current_waypoint[4]

        # Extract current states
        current_pos = obs['observation'][:3]
        current_yaw = obs['observation'][5]
        current_gripper = obs['observation'][6]

        # Calculate errors
        pos_error = np.linalg.norm(current_pos - target_pos)
        yaw_error = abs(current_yaw - target_yaw)
        gripper_error = abs(current_gripper - target_gripper)

        # Evaluate completion status
        pos_completed = pos_error < pos_tolerance
        yaw_completed = yaw_error < yaw_tolerance
        gripper_completed = (gripper_error < gripper_tolerance or
                             (action is not None and abs(action[4] - target_gripper) < 0.1))

        completed = pos_completed and yaw_completed and gripper_completed

        if completed:
            print(f"[Env] Waypoint {self._current_waypoint_index} successfully completed.")
            # Mark the current waypoint as done and increment
            self._waypoints[self._current_waypoint_index] = None
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
        else:
            print(f"  [Env] Waypoint {self._current_waypoint_index} pending - "
                  f"Pos error: {pos_error:.4f}, Yaw error: {yaw_error:.4f}, Gripper error: {gripper_error:.4f}")

        return completed

    def are_all_actions_completed(self):
        """
        Verify if all designated waypoints have been executed.
        """
        # Check if all elements in waypoints list are None
        all_completed = all(waypoint is None for waypoint in self._waypoints)

        if all_completed:
            print("[Env] All waypoints completed successfully.")
        else:
            remaining = sum(1 for waypoint in self._waypoints if waypoint is not None)
            print(f"  [Env] Waypoints remaining: {remaining}")
            print(f"  [Env] Current waypoint index: {self._current_waypoint_index}")

        return all_completed


if __name__ == "__main__":
    # Initialize environment
    env = NeedlePick(render_mode='human')

    for idx in range(1, 1001):
        # if idx in target_indices:
        # Create directories to save the collected data
        os.makedirs('rgb/0330-1000-raw-dataset/task3_touch_organ/rgb_images_long_224/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/0330-1000-raw-dataset/task3_touch_organ/depth_images_long_224/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/0330-1000-raw-dataset/task3_touch_organ/depth_images_long_224_32float/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/0330-1000-raw-dataset/task3_touch_organ/mask_images_long_224/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/0330-1000-raw-dataset/task3_touch_organ/state_data_long_224/' + str(idx),
                    exist_ok=True)

        obs = env.reset()
        frame_count = 0
        print(f"=== Starting Episode {idx} ===")

        # Reset waypoint tracking variables
        env._current_waypoint_index = 0
        env._step_in_waypoint = 0

        # Continue execution until all waypoints are resolved
        while not env.are_all_actions_completed():
            frame_count = frame_count + 1

            # Retrieve oracle/expert action
            action = env.get_oracle_action(obs)
            print(f"Executing action: {action}")
            # Verify waypoint completion prior to stepping
            env.is_action_completed(obs, action)

            obs, reward, done, info = env.step(action)
            print("---")

            # Acquire and process ECM imagery
            rgb_img, depth_img, mask_img = env.get_ecm_image(image_width=224, image_height=224)

            depth_img_32 = depth_img.astype(np.float32)
            depth_img = ((depth_img + 1) / (depth_img.max() + 1) * 255).astype(np.uint8)
            mask_img = ((mask_img + 1) / (mask_img.max() + 1) * 255).astype(np.uint8)
            state = env._get_robot_state(idx=0)

            # Save data artifacts...
            rgb_filename = f'rgb/0330-1000-raw-dataset/task3_touch_organ/rgb_images_long_224/' + str(
                idx) + f'/{frame_count:06d}.png'
            cv2.imwrite(rgb_filename, rgb_img)

            depth_filename = f'rgb/0330-1000-raw-dataset/task3_touch_organ/depth_images_long_224/' + str(
                idx) + f'/{frame_count:06d}.png'
            cv2.imwrite(depth_filename, depth_img)

            mask_filename = f'rgb/0330-1000-raw-dataset/task3_touch_organ/mask_images_long_224/' + str(
                idx) + f'/{frame_count:06d}.png'
            cv2.imwrite(mask_filename, mask_img)

            state_filename = f'rgb/0330-1000-raw-dataset/task3_touch_organ/state_data_long_224/' + str(
                idx) + f'/{frame_count:06d}.npy'
            np.save(state_filename, state)
            depth_filename32 = f'rgb/0330-1000-raw-dataset/task3_touch_organ/depth_images_long_224_32float/{idx}/{frame_count:06d}.npy'
            np.save(depth_filename32, depth_img_32)

            # Introduce a minor delay for observation purposes
            time.sleep(0.01)

        print(f"=== Episode {idx} Complete! Total Frames: {frame_count} ===\n")

    env.close()