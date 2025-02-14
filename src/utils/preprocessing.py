import os
import torch
from src.utils.preprocess import normalize_data, extract_brain, crop_to_largest_bounding_box, apply_gaussian_smoothing

def preprocess_images(
        normalize: bool = False,
        norm_method: str = "min-max",
        min_max_min_val: float = 0,
        min_max_max_val: float = 1,
        brain_extraction: bool = True, 
        brain_extraction_modality: str = "t1",
        brain_extraction_verbose: bool = False,
        crop: bool = False, 
        smooth: bool = False,
        smooth_sigma: float = 1.5,
        smooth_oder: int = 2,
        smooth_mode: str = "constant",
        smooth_cval: int = 1,
        smooth_truncate: float = 2, 
        re_normalize_after_smooth: bool = False, 
        preprocess_test_set: bool = False,
        output_dir: str = "preprocessed",
        ) -> None:
    """
    """
    for dataset in ["train_set", "test_set"] if preprocess_test_set else ["train_set"]:
        input_dir = f"data/raw_nii/{dataset}"
        output_dir = output_dir #f"{input_dir}/preprocessed"
        os.makedirs(output_dir, exist_ok=True)
        
        for img in os.listdir(input_dir):
            if not img.endswith(".pt"): 
                continue
            
            img_path = os.path.join(input_dir, img)
            data = torch.load(img_path).numpy()
            
            if normalize:
                data = normalize_data(data, 
                                      method=norm_method,
                                      min_val=min_max_min_val, 
                                      max_val=min_max_max_val)
            if brain_extraction:
                data = extract_brain(
                    data, 
                    modality=brain_extraction_modality, 
                    what_to_return={'extracted_brain': 'numpy', 'mask': 'numpy'},
                    verbose=brain_extraction_verbose,
                    )
                brain_data = data['extracted_brain']
            if crop: # TODO: handle cropping without brain extraction
                mask_data = data['mask']
                data = crop_to_largest_bounding_box(data=brain_data, mask=mask_data)
            if smooth: # TODO: handle smoothing with/without brain extraction
                data = apply_gaussian_smoothing(data,
                                                sigma=smooth_sigma,
                                                order=smooth_oder,
                                                mode=smooth_mode,
                                                cval=smooth_cval,
                                                truncate=smooth_truncate)
                if re_normalize_after_smooth:
                    data = normalize_data(data)
            
            torch.save(torch.tensor(data), os.path.join(output_dir, img))
