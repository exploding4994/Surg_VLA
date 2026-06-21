import os
import logging
import numpy as np
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CollectData:
    def __init__(self, pose, image):
        self.img = image
        self.gripper = pose[6]
        self.pos = pose[0:6]

    def write(self, path, index):
        """
        Save the collected data as a .npy file.

        Args:
            path (str): The destination directory path.
            index (int): The frame index for the saved file.
        """
        # Ensure the output directory exists
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

        # Package the data
        data = {
            'pose': np.array(self.pos, dtype=np.float32),
            'image': np.array(self.img, dtype=np.uint8),
            'gripper': self.gripper
        }

        try:
            # Save as a .npy file
            np.save(os.path.join(path, f"targ{index}"), data)
            logging.info(f"Data saved to {path} with index {index}")
        except Exception as e:
            logging.error(f"Failed to save data: {e}")


def process_data_with_variable_frames():
    """Handle raw dataset processing where frame counts may vary across groups."""

    # Base paths
    base_state_path = '/path/to/state_data_long_224'
    base_rgb_path = '/path/to/rgb_images_long_224'
    output_base_path = '/path/to/firstprocessed_task3_touch_organ_1000'

    # Iterate through each group
    for group_num in range(1, 1001):  # Process groups 1 through 1000
        group_state_path = os.path.join(base_state_path, str(group_num))
        group_rgb_path = os.path.join(base_rgb_path, str(group_num))
        output_path = os.path.join(output_base_path, str(group_num - 1))

        # Validate state data directory
        if not os.path.exists(group_state_path):
            logging.warning(f"State data path does not exist: {group_state_path}")
            continue

        # Validate RGB image directory
        if not os.path.exists(group_rgb_path):
            logging.warning(f"RGB image path does not exist: {group_rgb_path}")
            continue

        # Retrieve all .npy state files
        state_files = sorted([f for f in os.listdir(group_state_path) if f.endswith('.npy')])

        # Retrieve all .png image files
        rgb_files = sorted([f for f in os.listdir(group_rgb_path) if f.endswith('.png')])

        logging.info(f"Processing group {group_num}: State files = {len(state_files)}, RGB files = {len(rgb_files)}")

        # Skip if necessary files are missing
        if len(state_files) == 0 or len(rgb_files) == 0:
            logging.warning(f"No data files found in group {group_num}, skipping.")
            continue

        # Determine the minimum frame count to ensure synchronized state and image data
        min_frames = min(len(state_files), len(rgb_files))

        # Process matching frames
        for frame_idx in range(min_frames):
            try:
                # Construct file paths
                state_file_path = os.path.join(group_state_path, state_files[frame_idx])
                rgb_file_path = os.path.join(group_rgb_path, rgb_files[frame_idx])

                # Load state data
                pose_data = np.load(state_file_path, allow_pickle=True)

                # Load image
                image = Image.open(rgb_file_path)

                # Create CollectData object and save
                collector = CollectData(pose_data, image)
                collector.write(output_path, frame_idx + 1)  # 1-based indexing

            except Exception as e:
                logging.error(f"Error processing group {group_num}, frame {frame_idx + 1}: {e}")
                continue

        logging.info(f"Finished processing group {group_num}, processed {min_frames} frames total.")


if __name__ == "__main__":
    # Method 1: Sequential processing aligning to the minimum frame count
    print("Starting data processing method 1: Sequential processing...")
    process_data_with_variable_frames()

    print("Data processing complete!")