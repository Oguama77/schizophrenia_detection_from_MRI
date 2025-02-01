import os
import torch
import random
import numpy as np
import pandas as pd
from utils.augmentation import apply_translation, apply_rotation

def save_as_tensor(image_data, save_dir, filename):
    """Save the augmented image as a PyTorch tensor."""
    os.makedirs(save_dir, exist_ok=True)
    tensor_image = torch.tensor(image_data, dtype=torch.float32)
    save_path = os.path.join(save_dir, f"{filename}.pt")
    torch.save(tensor_image, save_path)
    print(f"Saved: {save_path}")

def augment_and_save(file_path, save_dir, augmentation_type, num_augmentations, value_range):
    """Applies augmentations and saves the output."""
    image_data = torch.load(file_path).numpy()
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    for i in range(1, num_augmentations + 1):
        if augmentation_type == "rotation":
            angle = random.uniform(*value_range)
            augmented_image = apply_rotation(image_data, angle)
        elif augmentation_type == "translation":
            translation = (random.randint(*value_range), random.randint(*value_range))
            augmented_image = apply_translation(image_data, translation)
        else:
            raise ValueError("Invalid augmentation type")
        
        save_as_tensor(augmented_image, save_dir, f"{filename}_{augmentation_type}_{i}")

def main():
    input_dir = "data/processed/extracted_brain"
    output_dir = "data/fully_processed/augmented"
    num_files = 50  # Number of randomly selected files
    num_augmentations = 3
    rotation_range = (-10, 10)
    translation_range = (-10, 10)
    
    # Get all .pt files and randomly select 50
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pt")]
    selected_files = random.sample(all_files, num_files)
    
    # Save selected filenames to a CSV file
    pd.DataFrame(selected_files, columns=["Filename"]).to_csv("selected_files.csv", index=False)
    print("Saved selected filenames to selected_files.csv")
    
    # Apply augmentations
    counter = 0
    for file_path in selected_files:
        augment_and_save(file_path, os.path.join(output_dir, "rotated"), "rotation", num_augmentations, rotation_range)
        augment_and_save(file_path, os.path.join(output_dir, "translated"), "translation", num_augmentations, translation_range)
        counter += 1
        print(f"Augmented {counter} file(s).")
        
if __name__ == "__main__":
    main()
