import os
import time
import numpy as np
import cv2  # 用于图像处理和保存

import pybullet as p
from surrol.utils.robotics import get_matrix_from_euler
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_body_pose,
)
from surrol.utils.utils import RGB_COLOR_255, Boundary, get_centroid
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
import os
import time
import numpy as np
from tqdm import tqdm

import pybullet as p
import pybullet_data
from surrol.utils.pybullet_utils import (
    step,
    get_joints,
    get_link_name,
    reset_camera,
)
from surrol.robots.psm import Psm

# 初始化环境
scaling = 1.0
p.connect(p.GUI)
# p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
reset_camera(yaw=90, pitch=-15, dist=0.9 * scaling)

# p.setPhysicsEngineParameter(contactBreakingThreshold=0.002)


# 加载场景和PSM机器人
p.loadURDF("plane.urdf", [0, 0, -0.001], globalScaling=1)

psm = Psm((0, 0, 0.1524),
          p.getQuaternionFromEuler((0, 0, -90 / 180 * np.pi)),
          scaling=scaling)
psm.reset_joint((0, 0, 0.10, 0, 0, 0))

def move_end_effector_delta(psm, delta_vector, num_steps=30, step_simulation=True):
    dx, dy, dz, drx, dry, drz, grip = delta_vector
    current_pose = psm.get_current_position()
    delta_rot = get_matrix_from_euler([drx, dry, drz])

    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        interp_rotation = current_pose[:3, :3] @ get_matrix_from_euler([drx * alpha, dry * alpha, drz * alpha])
        interp_position = current_pose[:3, 3] + np.array([dx, dy, dz]) * alpha

        target_pose = np.eye(4)
        target_pose[:3, :3] = interp_rotation
        target_pose[:3, 3] = interp_position

        psm.move(target_pose)
        if step_simulation:
            p.stepSimulation()

    # 夹爪控制
    if grip < 0:
        psm.close_jaw()
    else:
        psm.move_jaw(np.deg2rad(40))

if __name__ == "__main__":
    # 让末端向前移动 1cm，向右 1cm，绕 Y 轴转 10°，闭合夹爪
    # move_end_effector_delta(psm,
    #                         dx=0.01, dy=-0.01, dz=0.0,
    #                         dry=np.deg2rad(10),
    #                         grip=-1,
    #                         num_steps=30)

    # 定义基本动作：[dx, dy, dz, drx, dry, drz, grip]
    # 单位：米（m），弧度（rad），grip: -1=闭合, +1=打开

    # 1. 初始状态：夹爪打开
    print("1. Opening gripper")
    psm.move_jaw(np.deg2rad(40))
    p.stepSimulation()
    time.sleep(1)

    # 2. 向前移动，接近缝合针
    delta1 = [0.02, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0]  # 向前2cm，向下1cm #[ 0.44485473  0.57455807  0.24676588 -0.06900252 -1.96030526 -0.41374109]
    move_end_effector_delta(psm, delta1, num_steps=40)
    print("2. Moving forward and downward to approach the target")
    time.sleep(0.5)

    # delta1 = [ 0.44485473,0.57455807,0.24676588,-0.06900252,-1.96030526,-0.41374109,1]  # 向前2cm，向下1cm #[ 0.44485473  0.57455807  0.24676588 -0.06900252 -1.96030526 -0.41374109]
    # move_end_effector_delta(psm, delta1, num_steps=40)
    # print("2. Moving forward and downward to approach the target")
    # time.sleep(0.5)
    # delta1 = [0.44485473, 0.57455807, 0.24676588, -0.06900252, -1.96030526,
    #           -0.41374109,1]  # 向前2cm，向下1cm #[ 0.44485473  0.57455807  0.24676588 -0.06900252 -1.96030526 -0.41374109]
    # move_end_effector_delta(psm, delta1, num_steps=40)
    # print("2. Moving forward and downward to approach the target")
    # time.sleep(0.5)
    # delta1 = [0.44485473, 0.57455807, 0.24676588, -0.06900252, -1.96030526,
    #           -0.41374109,1]  # 向前2cm，向下1cm #[ 0.44485473  0.57455807  0.24676588 -0.06900252 -1.96030526 -0.41374109]
    # move_end_effector_delta(psm, delta1, num_steps=40)
    # print("2. Moving forward and downward to approach the target")
    # time.sleep(0.5)

    # 3. 微调姿态，对齐针的方向
    delta2 = [0.0, -0.005, 0.0, 0.0, np.deg2rad(10), 0.0, 1.0]  # 左移5mm，绕Y转10°
    move_end_effector_delta(psm, delta2, num_steps=30)
    print("3. Fine-tuning pose ")
    time.sleep(0.5)

    # 4. 向下插入，准备抓取
    delta3 = [0.0, 0.0, -0.015, 0.0, 0.0, 0.0, 1.0]  # 向下1.5cm
    move_end_effector_delta(psm, delta3, num_steps=30)
    print("4. Inserting downward")
    time.sleep(0.5)

    # 5. 闭合夹爪，抓取针
    delta4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # 仅闭合夹爪
    move_end_effector_delta(psm, delta4, num_steps=10)
    print("5. Closing gripper")
    time.sleep(0.5)

    # 6. 提起针
    delta5 = [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, -1.0]  # 向上3cm
    move_end_effector_delta(psm, delta5, num_steps=40)
    print("6. Lifting")
    time.sleep(0.5)

    # 7. 平移至目标位置（例如穿环）
    delta6 = [0.03, -0.02, 0.0, 0.0, 0.0, np.deg2rad(-30), -1.0]  # 右移3cm，后移2cm，绕Z转-30°
    move_end_effector_delta(psm, delta6, num_steps=50)
    print("7. Moving to target area")
    time.sleep(0.5)

    # 8. 向下放置
    delta7 = [0.0, 0.0, -0.02, 0.0, 0.0, 0.0, -1.0]  # 向下2cm
    move_end_effector_delta(psm, delta7, num_steps=30)
    print("8. Placing")
    time.sleep(0.5)

    # 9. 打开夹爪，释放针
    delta8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 打开夹爪
    move_end_effector_delta(psm, delta8, num_steps=10)
    print("9. Opening gripper")
    time.sleep(0.5)

    # 10. 抬起机械臂
    delta9 = [0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0]  # 上升3cm
    move_end_effector_delta(psm, delta9, num_steps=30)
    print("10. Operation complete, robotic arm reset")