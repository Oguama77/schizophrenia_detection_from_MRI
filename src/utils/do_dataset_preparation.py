import os
import shutil
import random
import torch
from utils.def_preprocess import load_nii, resample_image, normalize_data

def prepare_dataset(
        train_ratio=0.8, 
        raw_pt_dir="data/raw_pt", 
        raw_nii_dir="data/raw_nii", 
        normalize=True, 
        norm_method="min-max"
        ) -> None:
    """
    Randomly splits images from raw_pt_dir into train and test sets based on train_ratio.
    Copies images into train_set and test_set for raw_pt and ensures the same split for raw_nii,
    while also resampling and normalizing raw_nii images before saving them in .pt format.
    """
    # Create train/test directories for raw_pt
    train_pt_dir = os.path.join(raw_pt_dir, "train_set")
    test_pt_dir = os.path.join(raw_pt_dir, "test_set")
    os.makedirs(train_pt_dir, exist_ok=True)
    os.makedirs(test_pt_dir, exist_ok=True)
    
    # Get list of .pt files in raw_pt_dir
    pt_images = [f for f in os.listdir(raw_pt_dir) if f.endswith(".pt")]
    
    # Shuffle and split
    random.shuffle(pt_images)
    split_idx = int(len(pt_images) * train_ratio)
    train_images, test_images = pt_images[:split_idx], pt_images[split_idx:]
    
    # Copy files to respective directories
    for img in train_images:
        shutil.copy(os.path.join(raw_pt_dir, img), os.path.join(train_pt_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(raw_pt_dir, img), os.path.join(test_pt_dir, img))
    
    # Create train/test directories for raw_nii
    train_nii_dir = os.path.join(raw_nii_dir, "train_set")
    test_nii_dir = os.path.join(raw_nii_dir, "test_set")
    os.makedirs(train_nii_dir, exist_ok=True)
    os.makedirs(test_nii_dir, exist_ok=True)
    
    # Process nii images based on the same split
    for img in train_images + test_images:
        nii_path = os.path.join(raw_nii_dir, img.replace(".pt", ""))
        if not os.path.exists(nii_path):
            continue
        
        nii_img = load_nii(nii_path)
        resampled_img = resample_image(nii_img)
        if normalize:
            resampled_img = normalize_data(resampled_img, method=norm_method)
        
        save_path = os.path.join(train_nii_dir if img in train_images else test_nii_dir, img)
        torch.save(torch.tensor(resampled_img, dtype=torch.float32), save_path)
    
    print(f"Dataset prepared: {len(train_images)} images in train_set, {len(test_images)} images in test_set.")
