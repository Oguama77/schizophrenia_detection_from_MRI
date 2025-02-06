import os
import torch
import random
from utils.augmentations import apply_translation, apply_rotation, apply_gaussian_noise

def augment_images(input_dir, output_dir, augmentation_type, num_augmentations):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".pt"):
            file_path = os.path.join(input_dir, file)
            image_data = torch.load(file_path).numpy()

            for i in range(num_augmentations):
                if augmentation_type == "translation":
                    augmented = apply_translation(image_data, (random.randint(-10, 10), random.randint(-10, 10)))
                elif augmentation_type == "rotation":
                    augmented = apply_rotation(image_data, random.uniform(-10, 10))
                elif augmentation_type == "gaussian_noise":
                    augmented = apply_gaussian_noise(image_data, mean=0, std=0.1)

                save_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_{augmentation_type}_{i}.pt")
                torch.save(torch.tensor(augmented, dtype=torch.float32), save_path)
                print(f"Saved: {save_path}")

