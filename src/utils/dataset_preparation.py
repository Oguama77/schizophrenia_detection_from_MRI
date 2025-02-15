import os
import shutil
import random
import torch
from logger import logger
from src.utils.preprocess import load_nii, resample_image, normalize_data


def prepare_dataset(train_ratio: float = 0.76,
                    raw_pt_dir: str = "data/raw_pt",
                    raw_nii_dir: str = "data/raw_nii",
                    train_set_output_dir: str = "train_set",
                    test_set_output_dir: str = "test_set",
                    resampling_voxel_size: tuple = (2, 2, 2),
                    normalize: bool = True,
                    norm_method: str = "min-max",
                    min_max_min_val: float = 0,
                    min_max_max_val: float = 1) -> None:
    """
    Splits MRI images into training and test sets while maintaining the same split across .pt and .nii formats.
    
    The .pt files are copied into respective train/test folders for first training experiment, while
    .nii files are preprocessed (resampled, optionally normalized) before being saved in .pt format.

    Parameters:
        train_ratio (float): Proportion of data to use for training.
        raw_pt_dir (str): Directory containing raw .pt images.
        raw_nii_dir (str): Directory containing raw .nii images.
        normalize (bool): Whether to normalize the .nii images.
        norm_method (str): Normalization method ("min-max" or other supported methods).
    """
    # Define train/test directories
    train_pt_dir, test_pt_dir = os.path.join(raw_pt_dir, train_set_output_dir), os.path.join(raw_pt_dir, test_set_output_dir)
    train_nii_dir, test_nii_dir = os.path.join(raw_nii_dir, train_set_output_dir), os.path.join(raw_nii_dir, test_set_output_dir)

    # Create required directories
    for directory in [train_pt_dir, test_pt_dir, train_nii_dir, test_nii_dir]:
        os.makedirs(directory, exist_ok=True)

    # Get list of .pt files and shuffle
    pt_images = [f for f in os.listdir(raw_pt_dir) if f.endswith(".pt")]
    random.shuffle(pt_images)

    # Split dataset
    split_idx = int(len(pt_images) * train_ratio)
    train_images, test_images = pt_images[:split_idx], pt_images[split_idx:]

    def copy_files(file_list: list, source_dir: str, target_dir: str) -> None:
        """Copies a list of files from source_dir to target_dir."""
        for file in file_list:
            shutil.copy(os.path.join(source_dir, file),
                        os.path.join(target_dir, file))

    # Copy .pt files to respective directories
    copy_files(train_images, raw_pt_dir, train_pt_dir)
    copy_files(test_images, raw_pt_dir, test_pt_dir)

    def process_and_save_nii(image_list: list, nii_dir: str,
                             save_dir: str) -> None:
        """Processes and saves .nii images as .pt while preserving the dataset split."""
        for img in image_list:
            nii_path = os.path.join(nii_dir, img.replace(".pt", ""))
            if not os.path.exists(nii_path):
                continue

            nii_img = load_nii(nii_path)
            processed_img = resample_image(nii_img, voxel_size=resampling_voxel_size)
            if normalize:
                processed_img = normalize_data(processed_img,
                                               method=norm_method,
                                               min_val=min_max_min_val,
                                               max_val=min_max_max_val)

            torch.save(torch.tensor(processed_img, dtype=torch.float32),
                       os.path.join(save_dir, img))

    # Process .nii images
    process_and_save_nii(train_images, raw_nii_dir, train_nii_dir)
    process_and_save_nii(test_images, raw_nii_dir, test_nii_dir)

    logger.info(f"Dataset prepared: {len(train_images)} images in train_set, {len(test_images)} images in test_set.")
