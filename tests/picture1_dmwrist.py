import os
import logging
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class CollectData:
    def __init__(self, pose, image, depth, mask):
        self.img = image
        self.depth = depth
        self.mask = mask
        self.gripper = pose[6]
        self.pos = pose[0:6]

    def write(self, path, index):
        """将数据保存为npy文件"""
        # 确保路径存在
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"创建路径：{path}")

        # 将PIL图像转换为numpy数组
        img_array = np.array(self.img, dtype=np.uint8)  # 形状: (480, 640, 3)
        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        plt.figure(figsize=(6, 6))
        plt.imshow(gray, cmap='gray')
        # plt.figure(figsize=(8, 6))
        # plt.imshow(gray, cmap='gray')  # 关键：cmap='gray'
        plt.axis('off')  # 关闭坐标轴
        # plt.title('Grayscale Image')
        # plt.tight_layout()
        plt.show()
        depth_array = np.array(self.depth, dtype=np.uint8)  # 形状: (480, 640)
        mask_array = np.array(self.mask, dtype=np.uint8)  # 形状: (480, 640)

        # print(f"灰度图像形状: {gray.shape}")
        # print(f"深度图像形状: {depth_array.shape}")
        # print(f"掩码图像形状: {mask_array.shape}")

        # 为深度和掩码图像添加通道维度，使其变为3维
        if gray.ndim == 2:
            gray = np.expand_dims(gray, axis=2)  # 形状变为: (480, 640, 1)
        if depth_array.ndim == 2:
            depth_array = np.expand_dims(depth_array, axis=2)  # 形状变为: (480, 640, 1)
        if mask_array.ndim == 2:
            mask_array = np.expand_dims(mask_array, axis=2)  # 形状变为: (480, 640, 1)

        # print(f"处理后深度图像形状: {depth_array.shape}")
        # print(f"处理后掩码图像形状: {mask_array.shape}")

        # 在通道维度（第3维，axis=2）拼接
        combined_image = np.concatenate([
            gray,  # 1个通道 (灰度)
            depth_array,  # 1个通道 (深度)
            mask_array  # 1个通道 (掩码)
        ], axis=2)  # 最终形状: (480, 640, 5)

        # print(f"拼接后图像形状: {combined_image.shape}")

        # 收集数据
        data = {
            'pose': np.array(self.pos, dtype=np.float32),
            'image': img_array,
            'wrist': combined_image,# 3通道图像: 灰度 + Depth + Mask
            'gripper': self.gripper
        }

        try:
            # 保存为.npy文件
            np.save(os.path.join(path, f"targ{index}"), data)
            logging.info(f"数据已保存到 {path}，索引为 {index}")
        except Exception as e:
            logging.error(f"保存数据失败：{e}")


# 主循环
for a in range(1, 2):  # 帧数
    for b in range(1, 2):  # 组数
        file_path = f'/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_2/state_data_long/{b}/{a:06d}.npy'
        data = np.load(file_path, allow_pickle=True)
        pose = data

        index = a
        rgb_images = f'/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_2/rgb_images_long/'
        depth_images = f'/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_2/depth_images_long/'
        mask_images = f'/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_2/mask_images_long/'

        # 加载图像
        image = Image.open(f"{rgb_images}{b}/{a:06d}.png")
        depth = Image.open(f"{depth_images}{b}/{a:06d}.png")
        mask = Image.open(f"{mask_images}{b}/{a:06d}.png")

        # 创建CollectData对象并调用write方法
        C = CollectData(pose, image, depth, mask)
        path = f"/mnt/disk/hy/code/rlds_dataset_builder-main/example_dataset/task/needlepick_dep_mask/{b - 1}"
        C.write(path, index)