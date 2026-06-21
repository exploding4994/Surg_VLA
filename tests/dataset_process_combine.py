import os
import logging
import numpy as np
import cv2
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        # plt.figure(figsize=(8, 6))
        # plt.imshow(gray, cmap='gray')  # 关键：cmap='gray'
        # plt.axis('off')  # 关闭坐标轴
        # plt.title('Grayscale Image')
        # plt.tight_layout()
        # plt.show()
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
            'image': combined_image,
            # 'wrist': combined_image,# 3通道图像: 灰度 + Depth + Mask
            'gripper': self.gripper
        }

        try:
            # 保存为.npy文件
            np.save(os.path.join(path, f"targ{index}"), data)
            logging.info(f"数据已保存到 {path}，索引为 {index}")
        except Exception as e:
            logging.error(f"保存数据失败：{e}")


def process_data_with_variable_frames():
    """处理帧数不一致的情况"""

    # 基础路径
    base_state_path = '/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_3/state_data_long'
    base_rgb_path = '/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_3/rgb_images_long'
    base_depth_path = '/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_3/depth_images_long'
    base_mask_path = '/mnt/disk/sjs/just11/mask_processed'
    output_base_path = '/mnt/disk/sjs/just11'

    # 处理每个组
    for group_num in range(1, 51):  # 组数从1到50
        group_state_path = os.path.join(base_state_path, str(group_num))
        group_rgb_path = os.path.join(base_rgb_path, str(group_num))
        group_depth_path = os.path.join(base_depth_path, str(group_num))
        group_mask_path = os.path.join(base_mask_path, str(group_num))
        output_path = os.path.join(output_base_path, str(group_num - 1))

        # 检查所有路径是否存在
        paths_to_check = [group_state_path, group_rgb_path, group_depth_path, group_mask_path]
        missing_paths = [path for path in paths_to_check if not os.path.exists(path)]

        if missing_paths:
            logging.warning(f"组 {group_num} 缺少路径: {missing_paths}")
            continue

        # 获取各文件夹中的文件数量
        try:
            state_files = sorted([f for f in os.listdir(group_state_path) if f.endswith('.npy')])
            rgb_files = sorted([f for f in os.listdir(group_rgb_path) if f.endswith('.png')])
            depth_files = sorted([f for f in os.listdir(group_depth_path) if f.endswith('.png')])
            mask_files = sorted([f for f in os.listdir(group_mask_path) if f.endswith('.png')])
        except Exception as e:
            logging.error(f"读取组 {group_num} 文件列表时出错: {e}")
            continue

        # 找到最小帧数，确保所有数据匹配
        min_frames = min(len(state_files), len(rgb_files), len(depth_files), len(mask_files))

        logging.info(f"处理组 {group_num}: 状态文件={len(state_files)}, RGB文件={len(rgb_files)}, "
                     f"深度文件={len(depth_files)}, 掩码文件={len(mask_files)}, 最小帧数={min_frames}")

        if min_frames == 0:
            logging.warning(f"组 {group_num} 中没有找到数据文件，跳过")
            continue

        # 处理匹配的帧
        processed_count = 0
        for frame_idx in range(min_frames):
            try:
                # 构建文件路径
                state_file_path = os.path.join(group_state_path, state_files[frame_idx])
                rgb_file_path = os.path.join(group_rgb_path, rgb_files[frame_idx])
                depth_file_path = os.path.join(group_depth_path, depth_files[frame_idx])
                mask_file_path = os.path.join(group_mask_path, mask_files[frame_idx])

                # 加载状态数据
                pose_data = np.load(state_file_path, allow_pickle=True)

                # 加载图像
                image = Image.open(rgb_file_path)
                depth = Image.open(depth_file_path)
                mask = Image.open(mask_file_path)

                # 创建CollectData对象并保存
                collector = CollectData(pose_data, image, depth, mask)
                collector.write(output_path, frame_idx + 1)  # 索引从1开始
                processed_count += 1

            except Exception as e:
                logging.error(f"处理组 {group_num} 帧 {frame_idx + 1} 时出错：{e}")
                continue

        logging.info(f"完成处理组 {group_num}，共处理 {processed_count}/{min_frames} 帧数据")

if __name__ == "__main__":
    # 使用方法1：按顺序处理最小帧数
    print("开始处理方法1：按顺序处理...")
    process_data_with_variable_frames()


    print("数据处理完成！")