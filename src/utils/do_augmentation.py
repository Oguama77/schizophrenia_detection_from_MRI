import os
import torch
import random
from utils.def_augmentations import apply_translation, apply_rotation, apply_gaussian_noise

def augment_images(
        augmentations, 
        num_augmentations, 
        input_dir="data/raw_nii/train_set/preprocessed", 
        output_dir="data/raw_nii/train_set/preprocessed/augmented"
        ) -> None:
    """
    Applies user-specified augmentations to images in input_dir and saves them to output_dir.
    
    Parameters:
    - augmentations (list of tuples): List of augmentations to apply. Each tuple should be (augmentation_type, params).
      Example: [("translation", {"shift": (5, 5)}), ("rotation", {"angle": 15})]
    - num_augmentations (int): Number of times to apply the augmentation sequence per image.
    - input_dir (str): Path to the input directory containing .pt images.
    - output_dir (str): Path to save augmented images.
    """

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".pt"):
            file_path = os.path.join(input_dir, file)
            image_data = torch.load(file_path).numpy()

            for i in range(num_augmentations):
                augmented_data = image_data.copy()

                # Apply each augmentation in sequence
                augmentation_description = []
                for aug_type, params in augmentations:
                    if aug_type == "translation":
                        shift_x = params.get("shift_x", random.randint(-10, 10))
                        shift_y = params.get("shift_y", random.randint(-10, 10))
                        augmented_data = apply_translation(augmented_data, (shift_x, shift_y))
                        augmentation_description.append(f"trans_{shift_x}_{shift_y}")

                    elif aug_type == "rotation":
                        angle = params.get("angle", random.uniform(-10, 10))
                        augmented_data = apply_rotation(augmented_data, angle)
                        augmentation_description.append(f"rot_{angle:.1f}")

                    elif aug_type == "gaussian_noise":
                        mean = params.get("mean", 0)
                        std = params.get("std", 0.1)
                        augmented_data = apply_gaussian_noise(augmented_data, mean, std)
                        augmentation_description.append(f"noise_{std:.2f}")

                # Save augmented image with detailed filename
                save_name = f"{os.path.splitext(file)[0]}_{'_'.join(augmentation_description)}_{i}.pt"
                save_path = os.path.join(output_dir, save_name)
                torch.save(torch.tensor(augmented_data, dtype=torch.float32), save_path)

                print(f"Saved: {save_path}")
