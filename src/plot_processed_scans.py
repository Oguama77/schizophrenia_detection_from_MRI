import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import load_nii

# Define the root directory and subfolders
root_dir = 'data/processed'
subfolders = ['original', 'resampled_image', 'normalized_image', 'extracted_brain', 'cropped_image', 'smoothed_normalized_image']

# Initialize a dictionary to store the data
data_dict = {}

# Load and convert .pt files
for subfolder in subfolders:
    subfolder_path = os.path.join(root_dir, subfolder)
    for file in os.listdir(subfolder_path):
        if file.endswith(".pt") or file.endswith(".nii.gz"):
            file_path = os.path.join(subfolder_path, file)
            
            # Load the .pt file
            try:
                tensor = torch.load(file_path)
                np_array = tensor.numpy()
            except:
                np_array = load_nii(file_path)
                np_array = np_array.get_fdata()

            
            # Convert to a NumPy array
            
            
            # Store in the dictionary
            data_dict[subfolder] = np_array
            print(f"Loaded file: {file_path} with shape {np_array.shape}")

# Plot the data on the same figure
fig, axes = plt.subplots(1, len(data_dict), figsize=(15, 5))

for i, (key, array) in enumerate(data_dict.items()):
    # Take a slice from the middle of the volume for visualization (e.g., middle z-axis)
    middle_slice = array[:, :, 140] # array.shape[0] // 2
    
    # Plot the slice
    axes[i].imshow(middle_slice, cmap='gray')
    axes[i].set_title(key)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
