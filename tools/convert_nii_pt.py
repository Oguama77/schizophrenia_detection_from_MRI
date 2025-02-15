import os
import torch
import nibabel as nib
from logger import logger
from utils.preprocess import resample_image

# Paths
input_dir = 'data'
output_dir = 'data/raw_pt'
os.makedirs(output_dir, exist_ok=True)

# Resample parameters
voxel_size = (2, 2, 2)  # Adjust as needed
order = 4  # Nearest-neighbor interpolation (use 1 for trilinear, 3 for cubic)

def process_and_save(file_path, output_path):
    """Loads, resamples, converts, and saves the MRI scan as a PyTorch tensor."""
    nii = nib.load(file_path)  # Load .nii.gz file
    resampled = resample_image(nii, voxel_size=voxel_size, order=order, mode='reflect', cval=0)  # Resample
    tensor = torch.tensor(resampled, dtype=torch.float32)  # Convert to PyTorch tensor
    torch.save(tensor, output_path)  # Save
    logger.info(f"Saved: {output_path}")

# Process all .nii.gz files
counter = 0
for file in os.listdir(input_dir):
    if file.endswith(".nii.gz"):
        file_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.pt")
        process_and_save(file_path, output_path)
        counter += 1
        logger.info(f'Processed and saved {counter} files.')

logger.info("Processing complete!")
