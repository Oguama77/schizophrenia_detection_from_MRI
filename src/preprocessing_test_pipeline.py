import os
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from utils.preprocess import (
    load_nii,
    resample_image,
    normalize_data,
    extract_brain,
    crop_to_largest_bounding_box,
    apply_gaussian_smoothing
)

def preprocess_pipeline(file_path, steps_to_take: list, what_to_return: list):
    """
    Preprocessing pipeline for NIfTI files.
    
    Args:
        file_path (str): Path to the .nii.gz file.
        steps_to_take (list): Steps to apply during preprocessing.
        what_to_return (list): Outputs to save and return.
    
    Returns:
        dict: Dictionary with processed outputs requested in `what_to_return`.
    """
    to_be_returned = {}
    # Load NIfTI file
    nii = load_nii(file_path)

    # Resample image
    if 'resample' in steps_to_take:
        resampled_image = resample_image(nii, voxel_size=(1, 1, 1), order=0, mode='wrap', cval=0)
    else:
        pass#resampled_image = nii  # Default to the original image if not resampling
    
    if 'resampled_image' in what_to_return:
        to_be_returned['resampled_image'] = resampled_image

    # Normalize data
    if 'normalize' in steps_to_take:
        normalized_image = normalize_data(resampled_image)
    else:
        pass#normalized_image = resampled_image  # Use resampled image if no normalization
    
    if 'normalized_image' in what_to_return:
        to_be_returned['normalized_image'] = normalized_image

    # Brain extraction
    if 'extract_brain' in steps_to_take:
        brain_data = extract_brain(normalized_image, what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'})
        extracted_brain = brain_data['extracted_brain']
        mask = brain_data['mask']
    else:
        pass#extracted_brain = normalized_image  # Default if no extraction
        mask = None
    
    if 'extracted_brain' in what_to_return:
        to_be_returned['extracted_brain'] = extracted_brain

    # Cropping
    if 'crop' in steps_to_take and mask is not None:
        cropped_image = crop_to_largest_bounding_box(extracted_brain, mask=mask)
    else:
        pass#cropped_image = extracted_brain  # Use extracted brain if no cropping
    
    if 'cropped_image' in what_to_return:
        to_be_returned['cropped_image'] = cropped_image

    # Smoothing
    if 'smooth' in steps_to_take:
        smoothed_image = apply_gaussian_smoothing(cropped_image, sigma=0.5, order=0, mode='reflect', cval=0, truncate=3.0)
    else:
        pass#smoothed_image = cropped_image  # Use cropped image if no smoothing
    
    if 'smoothed_image' in what_to_return:
        to_be_returned['smoothed_image'] = smoothed_image

    # Normalize smoothed image
    if 'normalize_smoothed' in steps_to_take:
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
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    # Steps to take
    steps_to_take = ['resample', 'normalize', 'extract_brain', 'crop', 'smooth', 'normalize_smoothed']

    # Outputs to save
    what_to_return = [
                        #'resampled_image',
                        #'normalized_image',
                        'extracted_brain', 
                        'cropped_image', 
                        #'smoothed_image', 
                        'smoothed_normalized_image'
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
                torch.save(torch.tensor(data), pt_file_path)
                print(f"Saved: {pt_file_path}")
            
            counter += 1
            print(f"Processed {counter} file(s).")
            
            # Limit processing for testing purposes
            #if counter == 1:
            #    break
    #
    '''pt_file_path_extracted_brain = 'data/processed/extracted_brain/sub-A00023143_ses-20090101_acq-mprage_run-01_T1w.nii.pt'
    loaded_tensor = load_pt_file(pt_file_path_extracted_brain)
    print(f"Loaded tensor shape: {loaded_tensor.shape}")
    # Plot the loaded file, convert to numpy array
    loaded_np = loaded_tensor.numpy()
    plt.imshow(loaded_np[:,:,140], cmap='gray')
    plt.show()

    pt_file_path_smoothed_norm = 'data/processed/smoothed_normalized_image/sub-A00023143_ses-20090101_acq-mprage_run-01_T1w.nii.pt'
    loaded_tensor = load_pt_file(pt_file_path_smoothed_norm)
    loaded_np = loaded_tensor.numpy()
    plt.imshow(loaded_np[:,:,140], cmap='gray')
    plt.show()

    nii_file_path = 'data/sub-A00023143_ses-20090101_acq-mprage_run-01_T1w.nii.gz'
    loaded_nii = load_nii(nii_file_path)
    loaded_data = get_data(loaded_nii)
    plt.imshow(loaded_data[:,:,140], cmap='gray')
    plt.show()'''
    
