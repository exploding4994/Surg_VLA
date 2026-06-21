import os
import logging
import numpy as np
import cv2
from PIL import Image

# from readnpy import file_path

# 配置日志


class CollectData:
    def __init__(self, pose, image):
        self.img = image
        self.gripper = pose[6]
        self.pos = pose[0:6]

    def write(self, path, index):
        """将数据保存为npy文件"""
        # 确保路径存在
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"创建路径：{path}")

        # 收集数据
        data = {
            'pose': np.array(self.pos, dtype=np.float32),
            'image': np.array(self.img,dtype=np.uint8),
            'gripper': self.gripper  # 将gripper转化为字典
        }

        try:
            # 保存为.npy文件
            np.save(os.path.join(path, f"targ{index}"), data)
            logging.info(f"数据已保存到 {path}，索引为 {index}")
        except Exception as e:
            logging.error(f"保存数据失败：{e}")

# 示例数据

# a=1
# b=1
for a in range(1,51):      # 帧数
    for b in range(1, 101):  # 组数
        file_path=f'/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/test/state_data_w_gauze/{b}/{a:06d}.npy'
        data = np.load(file_path, allow_pickle=True)
        pose=data

        index = a
        name1 = f'/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/test/rgb_images_w_gauze/'
        # name2='/'
         # 假设这是你的变量
        image = Image.open(f"{name1}{b}/{a:06d}.png")



        # 创建CollectData对象并调用write方法
        C = CollectData(pose, image)
        path = f"/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/test/needle_process2/{b-1}"

        C.write(path, index)