import os


def remove_files_named(root_path, filename="000000.npy"):
    """
    递归删除 root_path 及其子目录下所有名为 filename 的文件。

    参数:
        root_path (str): 要遍历的根目录路径。
        filename (str): 要删除的文件名，默认为 "000000.png"。
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        if filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")


# 使用示例
if __name__ == "__main__":
    target_directory = "/mnt/disk/hy/code/SurRoL-main/SurRoL-main/surrol/tasks/test/state_data_w_gauze"  # 请替换为你的实际路径
    remove_files_named(target_directory)