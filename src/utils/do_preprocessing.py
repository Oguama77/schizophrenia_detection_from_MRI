import os
import torch
from utils.preprocess import load_nii, resample_image, normalize_data, extract_brain

def preprocess_images(input_dir, output_dir, steps):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".nii.gz"):
            file_path = os.path.join(input_dir, file)
            nii = load_nii(file_path)

            if "resample" in steps:
                image_data = resample_image(nii, voxel_size=(2, 2, 2))
            if "normalize" in steps:
                image_data = normalize_data(image_data)
            if "extract_brain" in steps:
                brain_data = extract_brain(image_data, {"extracted_brain": "numpy"})
                image_data = brain_data["extracted_brain"]

            torch.save(torch.tensor(image_data, dtype=torch.float32),
                       os.path.join(output_dir, f"{os.path.splitext(file)[0]}.pt"))
            print(f"Preprocessed {file}")
