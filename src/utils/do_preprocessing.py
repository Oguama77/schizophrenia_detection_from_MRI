import os
import torch
from utils.def_preprocess import normalize_data, extract_brain, crop_to_largest_bounding_box, apply_gaussian_smoothing

def preprocess_images(
        normalize=False, 
        brain_extraction=True, 
        crop=False, 
        smooth=False, 
        re_normalize_after_smooth=False, 
        preprocess_test=False
        ) -> None:
    """
    """
    for dataset in ["train_set", "test_set"] if preprocess_test else ["train_set"]:
        input_dir = f"data/raw_nii/{dataset}"
        output_dir = f"{input_dir}/preprocessed"
        os.makedirs(output_dir, exist_ok=True)
        
        for img in os.listdir(input_dir):
            if not img.endswith(".pt"): 
                continue
            
            img_path = os.path.join(input_dir, img)
            data = torch.load(img_path).numpy()
            
            if normalize:
                data = normalize_data(data)
            if brain_extraction:
                data = extract_brain(
                    data, 
                    modality='t1', 
                    what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'}
                    )
                brain_data = data['extracted_brain']
            if crop: # TODO: handle cropping without brain extraction
                mask_data = data['mask']
                data = crop_to_largest_bounding_box(data=brain_data, mask=mask_data)
            if smooth: # TODO: handle smoothing with/without brain extraction
                data = apply_gaussian_smoothing(data)
                if re_normalize_after_smooth:
                    data = normalize_data(data)
            
            torch.save(torch.tensor(data), os.path.join(output_dir, img))
