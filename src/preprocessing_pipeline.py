import os
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from utils.preprocess import (
    load_nii,
    get_data,
    resample_image,
    normalize_data,
    extract_brain,
    crop_to_largest_bounding_box,
    apply_gaussian_smoothing
)

from utils.preprocess_validation import calculate_metrics

def preprocess_pipeline(file_path, 
                        steps_to_take: list, 
                        what_to_return: list, 
                        separation: bool = False):
    """
    Preprocessing pipeline for NIfTI files.
    
    Args:
        file_path (str): Path to the .nii.gz file.
        steps_to_take (list): Steps to apply during preprocessing.
        what_to_return (list): Outputs to save and return.
    
    Returns:
        dict: Dictionary with processed outputs requested in `what_to_return`.
    """
    # Dictionary to store the output
    to_be_returned = {}

    # Load NIfTI file
    nii = load_nii(file_path)

    # Resample image
    if 'resample' in steps_to_take :
        resampled_image = resample_image(nii, voxel_size=(2, 2, 2), order=4, mode='reflect', cval=0)
    else:
        pass
    
    if 'resampled_image' in what_to_return:
        to_be_returned['resampled_image'] = resampled_image

    # Normalize data
    if 'normalize' in steps_to_take:
        normalized_image = normalize_data(resampled_image)
    else:
        pass
    
    if 'normalized_image' in what_to_return:
        to_be_returned['normalized_image'] = normalized_image

    # Brain extraction
    if 'extract_brain' in steps_to_take:
        if separation:
            brain_data = extract_brain(resampled_image, what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'})
        else:
            brain_data = extract_brain(normalized_image, what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'})
        extracted_brain = brain_data['extracted_brain']
        mask = brain_data['mask']
    else:
        mask = None
    
    if 'extracted_brain' in what_to_return:
        to_be_returned['extracted_brain'] = extracted_brain

    # Cropping
    if 'crop' in steps_to_take:
        if separation:
            brain_data = extract_brain(resampled_image, what_to_return={'mask': 'numpy'})
            mask = brain_data['mask']
            cropped_image = crop_to_largest_bounding_box(resampled_image, mask=mask)
        else:    
            cropped_image = crop_to_largest_bounding_box(brain_data, mask=mask)
    else:
        pass
    
    if 'cropped_image' in what_to_return:
        to_be_returned['cropped_image'] = cropped_image

    # Smoothing
    if 'smooth' in steps_to_take:
        if separation:
            smoothed_image = apply_gaussian_smoothing(resampled_image, sigma=0.5, order=0, mode='reflect', cval=0, truncate=3.0)
        else:    
            smoothed_image = apply_gaussian_smoothing(cropped_image, sigma=0.5, order=0, mode='reflect', cval=0, truncate=3.0)
    else:
        pass
    
    if 'smoothed_image' in what_to_return:
        to_be_returned['smoothed_image'] = smoothed_image

    # Normalize smoothed image
    if 'normalize_smoothed' in steps_to_take:
        if separation:
            smoothed_image = apply_gaussian_smoothing(resampled_image, sigma=0.5, order=0, mode='reflect', cval=0, truncate=3.0)
        smoothed_normalized_image = normalize_data(smoothed_image)    
    else:
        pass#smoothed_normalized_image = smoothed_image  # Use smoothed image if no normalization

    if 'smoothed_normalized_image' in what_to_return:
        to_be_returned['smoothed_normalized_image'] = smoothed_normalized_image

    return to_be_returned

def load_pt_file(pt_file_path):
    """
    Load a saved PyTorch tensor file.
    """
    return torch.load(pt_file_path)

if __name__ == '__main__':
    # Paths
    path_to_data = 'data'
    output_dir = 'data/processed/separated'
    os.makedirs(output_dir, exist_ok=True)

    # Steps to take
    steps_to_take = [
        'resample', 
        'normalize', 
        #'extract_brain', 
        #'crop', 
        #'smooth', 
        #'normalize_smoothed'
        ]

    # Outputs to save
    what_to_return = [
                        #'resampled_image',
                        'normalized_image',
                        #'extracted_brain', 
                        #'cropped_image', 
                        #'smoothed_image', 
                        #'smoothed_normalized_image'
                        ]

    # Process files
    counter = 0
    for file in os.listdir(path_to_data):
        if file.endswith(".nii.gz"):
            # Load .nii.gz file
            file_path = os.path.join(path_to_data, file)

            # Preprocess
            processed_data_to_save = preprocess_pipeline(file_path, steps_to_take=steps_to_take, what_to_return=what_to_return)

            # Save outputs
            for key, data in processed_data_to_save.items():
                # Ensure subdirectory exists for each output type
                save_dir = os.path.join(output_dir, key)
                os.makedirs(save_dir, exist_ok=True)

                # Save data as a PyTorch tensor
                pt_file_path = os.path.join(save_dir, f"{os.path.splitext(file)[0]}.pt")
                torch.save(torch.tensor(data, dtype=torch.float32), pt_file_path)
                print(f"Saved: {pt_file_path}")
            
            counter += 1
            print(f"Processed {counter} file(s).")
            
            # Limit processing for testing purposes
            #if counter == 1:
            #    break
    #
