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

# psm = Psm((0.05, 0.24, 0.8524),
#           p.getQuaternionFromEuler((0, 0, np.deg2rad(-(90+20)))),
#           scaling=scaling)
# psm.reset_joint((0, 0, 0.10, 0, 0, 0))

# for info in get_joints_info(psm.body, psm.joints):
#     print(info)


# 打印关节信息
joints = get_joints(psm.body)
print("There are {} joints.\n".format(len(joints)))

for i in range(0, len(joints)):
    print(get_link_name(psm.body, i))


# 持续运行模拟
p.setRealTimeSimulation(1)

# while True:
 #   p.setGravity(0, 0, -10)
  #  time.sleep(0.01)


# Forward Kinematics
# Test with predefined pose
# original joint position; 0.jpg
psm.move_joint([0, 0, 0.5, 0.0, 0.0, 0.0])
# psm.move_joint([-0.52359879, 0.0, 0.12, 0.0, 0.0, 0.0])
step(0.5)
print(psm.get_current_position())

# Read from dVRK (get_position_current)
# [[-0.0077    0.8686   -0.4955   -0.0567]
#  [ 0.9999    0.0001   -0.0154   -0.0002]
#  [-0.0133   -0.4956   -0.8685   -0.0982]
#  [      0         0         0    1.0000]]

# previously compute
# [[-6.12323415e-17  8.66025396e-01 -5.00000013e-01 -5.67500001e-02]
#  [ 1.00000000e+00 -3.06161708e-17 -1.75493441e-16 -1.35258490e-17]
#  [-1.67289863e-16 -5.00000013e-01 -8.66025396e-01 -9.82938802e-02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
#
#
# # simple joint position 2; 3.jpg
# psm.move_joint([-1.0471975511965976, 0.17453292519943295, 0.1, 0.7853981633974483, 0, 0])
# step(0.5)
# print(psm.get_current_position())
#
# # [[ 0.24721603  0.45989072 -0.85286855 -0.07974321]
# #  [ 0.69636423 -0.69636426 -0.17364817 -0.0162361 ]
# #  [-0.67376636 -0.55097853 -0.49240385 -0.04603976]
# #  [ 0.          0.          0.          1.        ]]
#
#
# # simple joint position 3; 4.jpg
# psm.move_joint([-1.0471975511965976, 0.17453292519943295, 0.1, 0.7853981633974483, 0.5235987755982988, 0.2617993877991494])
# step(0.5)
# print(psm.get_current_position())
#
# # [[-0.21233893  0.66737769 -0.71380613 -0.07982825]
# #  [ 0.51624502 -0.54359788 -0.66180997 -0.01919286]
# #  [-0.82970071 -0.50902688 -0.22910342 -0.0423738 ]
# #  [ 0.          0.          0.          1.        ]]
#
#
# # Open/Close Jaw
# # open jaw test
# psm.open_jaw()
# step(0.5)
# print(psm.get_current_position())
# print(np.rad2deg(psm.get_current_jaw_position()))
#
#
# # close jaw test
# psm.close_jaw()
# step(0.5)
# print(np.rad2deg(psm.get_current_jaw_position()))
#
#
# # Inverse Kinematics
# # Inverse kinematics test
# # simple coord position 0; 5.jpg
# pose = np.array([
#     [4.33293348e-13, 1.00000000e00, 4.52526905e-13, -1.33425265e-17],
#     [1.00000000e00, -4.33293348e-13, 9.95143306e-14, -1.12816941e-17],
#     [9.95143306e-14, 4.52526905e-13, -1.00000000e00, -1.13499997e-01],
#     [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]
# ])
# psm.move(pose)
# step(0.5)
# print(psm.get_current_joint_position())
# print(psm.get_current_position())
#
# # [ 0.00000000e+00  8.65586272e-15  1.19999997e-01  4.33293348e-13
# #  -1.08357767e-13  4.52504218e-13]
#
#
# # simple coord position 1; 6.jpg
# pose = np.array([
#     [4.33293348e-13, 1.00000000e00, 4.52526905e-13, 5.00000000e-02],
#     [1.00000000e00, -4.33293348e-13, 9.95143306e-14, -1.12816941e-17],
#     [9.95143306e-14, 4.52526905e-13, -1.00000000e00, -1.13499997e-01],
#     [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00]
# ])
# psm.move(pose)
# step(0.5)
# print(psm.get_current_joint_position())
# print(psm.get_current_position())

# [ 4.14949688e-01 -7.10542736e-15  1.30525197e-01  4.36665965e-13
#   9.00240590e-14 -4.14949688e-01]


# JIGSAWS Kinematics Record
# jigsaws test
# import kinematics data transformed from JIGSAWS
# (https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
# joint_values = np.load('qs_jigsaws.npy')
# psm.close_jaw()
# step(1)
#
# start_time = time.time()
# for i in tqdm(range(len(joint_values))):
#     psm.move_joint(joint_values[i])
#     psm.close_jaw()
#     step(0.5)
#     _ = p.getCameraImage(128, 128)
#
# end_time = time.time()
# print("Used time: {:.4f}".format(end_time - start_time))


# 截图
_ = p.getCameraImage(512, 512)

while True:
    time.sleep(0.01)