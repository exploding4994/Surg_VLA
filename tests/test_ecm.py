import os
import time
import numpy as np

import pybullet as p
import pybullet_data
from surrol.utils.pybullet_utils import (
    step,
    get_joints,
    get_link_name,
    reset_camera,
)
from surrol.robots.ecm import Ecm
from surrol.const import ASSET_DIR_PATH


def test_ecm_kinematics():
    # 初始化设置
    scaling = 1.0

    # 连接物理引擎
    p.connect(p.GUI)
    # p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    reset_camera(yaw=10, pitch=-15, dist=0.9 * scaling)

    # 加载环境
    p.loadURDF("plane.urdf", [0, 0, -0.001], globalScaling=1)

    # 加载ECM机器人
    ecm = Ecm([0, 0, 0 * scaling], scaling=scaling)
    ecm.reset_joint([0, 0, 0, 0])

    # 打印关节信息
    joints = get_joints(ecm.body)
    print(f"There are {len(joints)} joints.\n")

    for i in range(0, len(joints)):
        print(get_link_name(ecm.body, i))

    # 连续运行模拟
    p.setRealTimeSimulation(1)

    # 正向运动学测试
    print("\nForward Kinematics Tests:")

    # 测试1
    ecm.reset_joint([0, 0, 0, 0])
    pose_rcm = ecm.get_current_position()
    print("Test 1:\n", np.round(pose_rcm, 4))

    # 测试2
    ecm.reset_joint([-0.0024, -0.0023, 0.0025, -0.0007])
    pose_rcm = ecm.get_current_position()
    print("\nTest 2:\n", np.round(pose_rcm, 4))

    # 测试3
    ecm.reset_joint([0.0884, -0.6098, 0.1961, -0.0118])
    pose_rcm = ecm.get_current_position()
    print("\nTest 3:\n", np.round(pose_rcm, 4))

    # 测试4
    ecm.reset_joint([0.5369, 0.1454, 0.0316, 0.1266])
    pose_rcm = ecm.get_current_position()
    print("\nTest 4:\n", np.round(pose_rcm, 4))

    # 逆向运动学测试
    print("\nInverse Kinematics Tests:")

    # 测试1
    pose_rcm = np.array([
        [1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., -1., -0.0007],
        [0., 0., 0., 1.],
    ])
    joints_inv = ecm.move(pose_rcm)
    step(0.5)
    print("\nTest 1:\n", np.round(joints_inv, 4))

    # 测试2
    pose_rcm = np.array([
        [1.0000, 0.0006, -0.0024, -0.0000],
        [0.0007, -1.0000, 0.0023, 0.0000],
        [-0.0024, -0.0023, -1.0000, -0.0032],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ])
    joints_inv = ecm.move(pose_rcm)
    step(0.5)
    print("\nTest 2:\n", np.round(joints_inv, 4))

    # 测试3
    pose_rcm = np.array([
        [0.9954, 0.0623, 0.0724, 0.0142],
        [0.0097, -0.8197, 0.5727, 0.1127],
        [0.0950, -0.5694, -0.8166, -0.1607],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ])
    joints_inv = ecm.move(pose_rcm)
    step(0.5)
    print("\nTest 3:\n", np.round(joints_inv, 4))

    # 测试4
    pose_rcm = np.array([
        [0.8431, -0.182, 0.5061, 0.0163],
        [-0.1249, -0.9815, -0.1449, -0.0047],
        [0.5231, 0.0589, -0.8502, -0.0275],
        [0., 0., 0., 1.],
    ])
    joints_inv = ecm.move(pose_rcm)
    step(0.5)
    print("\nTest 4:\n", np.round(joints_inv, 4))

    while True:
        p.setGravity(0, 0, -9.81)
        time.sleep(0.01)

if __name__ == "__main__":
    test_ecm_kinematics()