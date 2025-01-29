import os
import torch
import random

from utils.preprocess import load_nii, resample_image
from utils.augmentation import apply_translation, apply_rotation, apply_gaussian_noise

def preprocess_and_augment(file_path, augmentation_type, num_augmentations=1, 
                           translation_range=(-10, 10), rotation_degrees=(-20, 20), noise_mean=0, noise_std=0.1):
    """Apply augmentation (translation, rotation, or gaussian noise) to the image."""
    nii = load_nii(file_path)
    image_data = resample_image(nii, voxel_size=(2, 2, 2), order=4, mode='reflect', cval=0)
    #print(f"Original shape: {image_data.shape}")  # Debugging step
    
    augmented_images = []

    for _ in range(num_augmentations):
        if augmentation_type == 'translation':
            translation = (random.randint(translation_range[0], translation_range[1]), 
                           random.randint(translation_range[0], translation_range[1]))  # Random translation in x and y axes
            augmented_image = apply_translation(image_data, translation)
        elif augmentation_type == 'rotation':
            angle = random.randint(rotation_degrees[0], rotation_degrees[1])  # Random rotation angle between specified degrees
            augmented_image = apply_rotation(image_data, angle)
        elif augmentation_type == 'gaussian_noise':
            augmented_image = apply_gaussian_noise(image_data, mean=noise_mean, std=noise_std)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        
        #print(f"Augmented shape: {augmented_image.shape}")  # Debugging step

        augmented_images.append(augmented_image)
    
    return augmented_images

def save_as_tensor(image_data, save_dir, filename):
    """Save the augmented image as a PyTorch tensor."""
    tensor_image = torch.tensor(image_data, dtype=torch.float32) # , dtype=torch.float32
    save_path = os.path.join(save_dir, f"{filename}.pt")
    torch.save(tensor_image, save_path)
    #print(f"Saved tensor shape: {tensor_image.shape}")  # Debugging
    print(f"Saved augmented image: {save_path}")

def augment_images_in_directory(directory, augmentation_type, num_augmentations=1, 
                                 translation_range=(-10, 10), rotation_degrees=(0, 360), noise_mean=0, noise_std=0.1):
    """Iterate over all .nii.gz files in a directory and apply augmentations."""
    output_dir = 'data/processed/augmented'
    os.makedirs(output_dir, exist_ok=True)

    counter = 0 # For testing purposes
    for file in os.listdir(directory):
        if file.endswith(".nii.gz"):
            file_path = os.path.join(directory, file)

            # Apply augmentations
            augmented_images = preprocess_and_augment(file_path, augmentation_type, num_augmentations, 
                                                      translation_range, rotation_degrees, noise_mean, noise_std)
            # Save each augmented image
            for idx, augmented_image in enumerate(augmented_images):
                # Ensure directory exists for each augmented image
                save_dir = os.path.join(output_dir, augmentation_type)
                os.makedirs(save_dir, exist_ok=True)

                # Save image as tensor
                filename = f"{os.path.splitext(file)[0]}_aug_{idx+1}"
                save_as_tensor(augmented_image, save_dir, filename)
            # Limit processing for testing purposes
            counter += 1
            print(f"Processed {counter} file(s).")
            #if counter == 1:
            #    break

if __name__ == '__main__':
    # Path to data directory
    data_dir = 'data'

    # User-defined options for augmentation
    augmentation_type = 'gaussian_noise'  # 'translation', 'rotation', or 'gaussian_noise'
    num_augmentations = 5  # Number of augmentations per scan
    translation_range = (-10, 10)  # Range for random translation (x and y axes)
    rotation_degrees = (-10, 10)  # Range for random rotation angles in degrees
    noise_mean = 0  # Mean for Gaussian noise
    noise_std = 0.1  # Standard deviation for Gaussian noise

    # Run augmentation on all images in the directory
    augment_images_in_directory(data_dir, augmentation_type, num_augmentations, 
                                 translation_range, rotation_degrees, noise_mean, noise_std)
