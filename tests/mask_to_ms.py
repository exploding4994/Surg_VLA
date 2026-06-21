import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm  # 进度条库，如果没有安装请先安装：pip install tqdm

# 输入目录
input_base_dir = "/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/r_d_m_3/mask_images_long"
# 输出目录
output_base_dir = "/mnt/disk/sjs/just11"


def process_single_mask(image_path, output_dir):
    """
    处理单个掩码图像：将非85和255的灰度值全部变成0

    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    try:
        # 打开图像
        img = Image.open(image_path)
        img_array = np.array(img)

        # 如果是彩色图像，转换为灰度
        if len(img_array.shape) == 3:
            img = img.convert('L')
            img_array = np.array(img)

        # 处理：只保留85和255，其他设为0
        processed_array = img_array.copy()
        processed_array[(processed_array != 85) & (processed_array != 255)] = 0

        # 获取相对路径，用于保持目录结构
        rel_path = os.path.relpath(image_path, input_base_dir)
        rel_dir = os.path.dirname(rel_path)

        # 创建对应的输出目录
        output_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(output_subdir, exist_ok=True)

        # 生成输出文件名
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]

        # 保存处理后的PNG
        output_image_path = os.path.join(output_subdir, f"{name_without_ext}.png")
        processed_img = Image.fromarray(processed_array)
        processed_img.save(output_image_path)

        # 保存灰度值到txt（每个文件单独保存）
        output_txt_path = os.path.join(output_subdir, f"{name_without_ext}_grayscale.txt")
        processed_values = np.unique(processed_array)
        #np.savetxt(output_txt_path, processed_values, fmt='%d')

        return True, image_path

    except Exception as e:
        print(f"处理失败 {image_path}: {str(e)}")
        return False, image_path


def batch_process_masks(input_dir, output_dir):
    """
    批量处理所有掩码图像

    参数:
        input_dir: 输入根目录
        output_dir: 输出根目录
    """
    # 查找所有PNG文件
    pattern = os.path.join(input_dir, "**", "*.png")
    image_files = glob.glob(pattern, recursive=True)

    if not image_files:
        print(f"在 {input_dir} 中未找到PNG文件")
        return

    print(f"找到 {len(image_files)} 个PNG文件需要处理")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 创建输出根目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    success_count = 0
    fail_count = 0

    # 使用进度条处理所有文件
    with tqdm(total=len(image_files), desc="处理进度", unit="文件") as pbar:
        for image_path in image_files:
            success, file_path = process_single_mask(image_path, output_dir)

            if success:
                success_count += 1
                # 显示当前处理的文件（可选）
                if pbar.n % 10 == 0:  # 每10个文件显示一次
                    pbar.set_postfix_str(f"当前: {os.path.basename(file_path)}")
            else:
                fail_count += 1
                # 记录失败的文件
                with open(os.path.join(output_dir, "failed_files.txt"), "a") as f:
                    f.write(f"{file_path}\n")

            pbar.update(1)

    # 生成汇总报告
    generate_summary_report(output_dir, success_count, fail_count, len(image_files))

    print("=" * 60)
    print(f"处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"总文件数: {len(image_files)} 个")


def generate_summary_report(output_dir, success_count, fail_count, total_count):
    """
    生成处理汇总报告
    """
    report_path = os.path.join(output_dir, "batch_process_summary.txt")

    with open(report_path, "w") as f:
        f.write("批量处理掩码图像 - 汇总报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"输入目录: {input_base_dir}\n")
        f.write(f"输出目录: {output_base_dir}\n")
        f.write(f"处理时间: {os.path.getctime(__file__)}\n")  # 简化时间显示
        f.write("\n处理统计:\n")
        f.write(f"  总文件数: {total_count}\n")
        f.write(f"  成功处理: {success_count}\n")
        f.write(f"  处理失败: {fail_count}\n")
        f.write(f"  成功率: {success_count / total_count * 100:.2f}%\n")
        f.write("\n处理规则:\n")
        f.write("  只保留灰度值85和255，其他灰度值全部设置为0\n")
        f.write("\n输出文件说明:\n")
        f.write("  1. 处理后的图像: [原文件名].png\n")
        f.write("  2. 灰度值列表: [原文件名]_grayscale.txt\n")
        f.write("  3. 失败文件列表: failed_files.txt\n")
        f.write("  4. 本汇总报告: batch_process_summary.txt\n")


def process_by_folder(folder_paths, output_dir):
    """
    处理指定的文件夹列表

    参数:
        folder_paths: 文件夹路径列表
        output_dir: 输出目录
    """
    all_image_files = []

    # 收集所有文件夹中的PNG文件
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            pattern = os.path.join(folder_path, "*.png")
            folder_files = glob.glob(pattern)
            all_image_files.extend(folder_files)
            print(f"从 {folder_path} 找到 {len(folder_files)} 个PNG文件")
        else:
            print(f"警告: 文件夹不存在 {folder_path}")

    if not all_image_files:
        print("未找到任何PNG文件")
        return

    print(f"总计找到 {len(all_image_files)} 个PNG文件需要处理")
    print("开始处理...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    success_count = 0
    fail_count = 0

    # 处理所有文件
    for image_path in tqdm(all_image_files, desc="处理进度", unit="文件"):
        # 获取文件夹名
        folder_name = os.path.basename(os.path.dirname(image_path))

        # 创建对应的输出子目录
        output_subdir = os.path.join(output_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        success, _ = process_single_mask(image_path, output_subdir)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"处理完成! 成功: {success_count}, 失败: {fail_count}")


def main():
    """主函数"""
    print("掩码图像批量处理工具")
    print("=" * 50)
    print(f"输入目录: {input_base_dir}")
    print(f"输出目录: {output_base_dir}")
    print("处理规则: 只保留灰度值85和255，其他灰度值全部设为0")
    print("=" * 50)

    # 选择处理模式
    print("\n请选择处理模式:")
    print("1. 处理所有子文件夹中的所有PNG文件")
    print("2. 处理指定的几个文件夹")
    print("3. 处理单个文件夹")

    choice = input("请输入选择 (1/2/3): ").strip()

    if choice == "1":
        # 模式1: 处理所有子文件夹
        print("\n开始处理所有子文件夹中的PNG文件...")
        batch_process_masks(input_base_dir, output_base_dir)

    elif choice == "2":
        # 模式2: 处理指定的文件夹
        print("\n请输入要处理的文件夹编号（用逗号分隔，如: 1,2,3）:")
        folder_nums = input("文件夹编号: ").strip().split(',')

        folder_paths = []
        for num in folder_nums:
            num = num.strip()
            folder_path = os.path.join(input_base_dir, num)
            if os.path.exists(folder_path):
                folder_paths.append(folder_path)
            else:
                print(f"警告: 文件夹 {folder_path} 不存在")

        if folder_paths:
            output_subdir = os.path.join(output_base_dir, "selected_folders")
            process_by_folder(folder_paths, output_subdir)
        else:
            print("没有有效的文件夹可处理")

    elif choice == "3":
        # 模式3: 处理单个文件夹
        folder_num = input("请输入文件夹编号: ").strip()
        folder_path = os.path.join(input_base_dir, folder_num)

        if os.path.exists(folder_path):
            output_subdir = os.path.join(output_base_dir, folder_num)
            process_by_folder([folder_path], output_subdir)
        else:
            print(f"错误: 文件夹 {folder_path} 不存在")
    else:
        print("无效的选择，程序退出")


if __name__ == "__main__":
    # 如果需要安装tqdm，取消下面的注释
    # import subprocess
    # subprocess.run(["pip", "install", "tqdm"])

    main()