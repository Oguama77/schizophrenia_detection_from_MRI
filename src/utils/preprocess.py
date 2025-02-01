import os
import ants
import antspynet
import numpy as np
import nibabel as nib
from typing import Union
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from nibabel.processing import resample_to_output


class ImagePreprocessor:
    
    @staticmethod
    def load_nii(file_path: str) -> nib.Nifti1Image:
        """
        Load a nifti file from a given file path

        Args:
        file_path: str: path to the nifti file

        Returns:
        nib.Nifti1Image: nifti image object
        """
        if not isinstance(file_path, str):
            raise TypeError(f"Expected a string for file_path, got {type(file_path)}")
    
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            nii = nib.load(file_path)
        except nib.filebasedimages.ImageFileError:
            raise ValueError(f"File is not a valid NIfTI file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading the file: {str(e)}")

        if not isinstance(nii, nib.Nifti1Image):
            raise ValueError(f"Loaded object is not a NIfTI image: {type(nii)}")

        return nii

    @staticmethod
    def get_data(nii: nib.Nifti1Image) -> np.ndarray:
        """
        Get the data from a nifti image object

        Args:
        nii: nib.Nifti1Image: nifti image object

        Returns:
        np.ndarray: nifti image data
        """
        if not isinstance(nii, nib.Nifti1Image):
            raise TypeError(f"Expected nib.Nifti1Image, got {type(nii)}")

        try:
            data = nii.get_fdata()
        except Exception as e:
            raise ValueError(f"Failed to get data from nifti image: {str(e)}")

        if data.size == 0:
            raise ValueError("The nifti image data is empty")
        
        if not np.isfinite(data).all():
            raise ValueError("The nifti image data contains non-finite values")

        return data

    @staticmethod
    def get_affine(nii: nib.Nifti1Image) -> np.ndarray:
        """
        Get the affine matrix from a nifti image object

        Args:
        nii: nib.Nifti1Image: nifti image object

        Returns:
        np.ndarray: affine matrix
        """
        if not isinstance(nii, nib.Nifti1Image):
            raise TypeError(f"Expected nib.Nifti1Image, got {type(nii)}")

        try:
            affine = nii.affine
        except AttributeError:
            raise ValueError("The nifti image object does not have an affine attribute")

        if not isinstance(affine, np.ndarray):
            raise ValueError(f"Affine matrix is not a numpy array, got {type(affine)}")

        if affine.shape != (4, 4):
            raise ValueError(f"Affine matrix has incorrect shape. Expected (4, 4), got {affine.shape}")

        if not np.isfinite(affine).all():
            raise ValueError("The affine matrix contains non-finite values")

        return affine

    @staticmethod
    def match_dimensions(original_image: np.ndarray, modified_image: np.ndarray) -> np.ndarray:
        """
        Pad or crop modified_image to match the dimensions of original_image.

        Args:
        original_image: np.ndarray: original image data
        modified_image: np.ndarray: modified image data

        Returns:
        np.ndarray: resampled and padded/cropped modified image data
        """
        diff = np.array(original_image.shape) - np.array(modified_image.shape)
        pad = [(0, max(0, d)) for d in diff]
        crop = [slice(0, min(s1, s2)) for s1, s2 in zip(original_image.shape, modified_image.shape)]

        matched = np.pad(modified_image, pad_width=pad, mode='constant') if np.any(diff > 0) else modified_image[tuple(crop)]
        return matched[:original_image.shape[0], :original_image.shape[1], :original_image.shape[2]]

    @staticmethod
    def resample_image(data: nib.Nifti1Image,
                       voxel_size: tuple=(2, 2, 2),
                       order: int = 4,
                       mode: str='reflect',
                       cval: float=0.0,
                       output_format: str='numpy'
                       ) -> Union[nib.Nifti1Image, np.ndarray]:
        """
        Resample a nifti image to a given voxel size

        Args:
        data: Union[nib.Nifti1Image, np.ndarray]: nifti image object or numpy array
        voxel_size: tuple: voxel size for resampling
        output_format: str: output format, either 'nib' or 'numpy'

        Returns:
        Union[nib.Nifti1Image, np.ndarray]: resampled nifti image object or numpy array
        """
        if not isinstance(data, nib.Nifti1Image):
            raise TypeError(f"Expected nib.Nifti1Image, got {type(data)}")

        if not isinstance(voxel_size, tuple) or len(voxel_size) != 3:
            raise ValueError(f"voxel_size must be a tuple of length 3, got {voxel_size}")

        if not all(isinstance(x, (int, float)) and x > 0 for x in voxel_size):
            raise ValueError(f"All elements in voxel_size must be positive numbers, got {voxel_size}")

        if not isinstance(output_format, str):
            raise TypeError(f"output_format must be a string, got {type(output_format)}")

        try:
            resampled = resample_to_output(data, voxel_sizes=voxel_size, order=order, mode=mode, cval=cval)
        except Exception as e:
            raise RuntimeError(f"Resampling failed: {str(e)}")

        if output_format.lower() == 'nib':
            return resampled
        elif output_format.lower() == 'numpy':
            try:
                return resampled.get_fdata(caching='unchanged')
            except Exception as e:
                raise RuntimeError(f"Failed to convert resampled image to numpy array: {str(e)}")
        else:
            raise ValueError(f"Invalid output_format: {output_format}. Should be either 'nib' or 'numpy'")

    @staticmethod
    def normalize_data(data: np.ndarray,
                       method: str = "min-max",
                       min_val=0,
                       max_val=1,
                       eps: float = 1e-8) -> np.ndarray:
        """
        Normalize the data using z-score or min-max method

        Args:
        data: np.ndarray: data to be normalized
        method: str: normalization method, either "z-score" or "min-max"
        min_val: (float): Minimum value of the target range.
        max_val: (float): Maximum value of the target range.
        eps: float: a small constant to avoid division by zero

        Returns:
        np.ndarray: normalized data
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
        
        try:
            if method == 'z-score':
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    raise ZeroDivisionError("Standard deviation is zero. Cannot perform z-score normalization")
                return (data - mean) / std
            
            elif method =='min-max':
                min_data_value = np.min(data)
                max_data_value = np.max(data)
                delta_data_value = max_data_value - min_data_value

                if delta_data_value == 0:
                    raise ZeroDivisionError("No variation in the data. Cannot perform min-max normalization")
                
                return ((data - min_data_value) / (delta_data_value + eps)) * (max_val - min_val) + min_val
            
            else:
                raise ValueError("Invalid normalization method. Choose 'z-score' or 'min-max'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while normalizing the data: {str(e)}")

    @staticmethod
    def extract_brain(data: np.ndarray,
                      modality: str = 't1',
                      what_to_return: dict = {'extracted_brain': 'numpy'},
                      verbose: bool = True
                      ) -> dict:
        """
        Extract brain from a given image using deep learning brain extraction.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")

        try:
            # Convert numpy array to ANTs image
            image = ants.from_numpy(data)
            if image is None:
                raise RuntimeError("Failed to initialize ants.Image object")

            # Perform brain extraction
            mask = antspynet.brain_extraction(image, modality=modality, verbose=verbose)
            if mask is None:
                raise RuntimeError("Failed to perform brain extraction")

            # Apply the mask to extract the brain
            extracted_brain = image * mask

            # Prepare results based on 'what_to_return'
            result = {}
            for key, value in what_to_return.items():
                if key == 'image':
                    result['image'] = image.numpy() if value == 'numpy' else image
                elif key == 'mask':
                    result['mask'] = mask.numpy() if value == 'numpy' else mask
                elif key == 'extracted_brain':
                    result['extracted_brain'] = extracted_brain.numpy() if value == 'numpy' else extracted_brain

            return result
        except Exception as e:
            raise RuntimeError(f"Error during brain extraction: {str(e)}")

    @staticmethod
    def get_largest_brain_mask_slice(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Get the slice of the largest brain mask from the input image data
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array for data, got {type(data)}")
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"Expected numpy array for mask, got {type(mask)}")

        # Find the largest connected region in the mask
        from skimage.measure import label, regionprops
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        largest_region = max(regions, key=lambda r: r.area)
        largest_slice = largest_region.slice

        return data[largest_slice]
    
    @staticmethod
    def crop_to_largest_bounding_box(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Crop the image data to the largest bounding box defined by the mask
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array for data, got {type(data)}")
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"Expected numpy array for mask, got {type(mask)}")

        # Find the largest connected region in the mask
        from skimage.measure import label, regionprops
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, minz, maxr, maxc, maxz = largest_region.bbox

        return data[minr:maxr, minc:maxc, minz:maxz]
    
    @staticmethod
    def apply_gaussian_smoothing(data: np.ndarray, sigma: float = 1) -> np.ndarray:
        """
        Apply Gaussian smoothing to the image data
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")

        try:
            smoothed_data = gaussian_filter(data, sigma=sigma)
            return smoothed_data
        except Exception as e:
            raise RuntimeError(f"Failed to apply Gaussian smoothing: {str(e)}")
