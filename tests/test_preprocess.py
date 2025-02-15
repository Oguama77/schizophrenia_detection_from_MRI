import unittest
import tempfile
import os
import numpy as np
import nibabel as nib
from utils.preprocess import (
    load_nii, 
    get_data, 
    get_affine, 
    match_dimensions,
    resample_image, 
    normalize_data, 
    extract_brain,
    get_largest_brain_mask_slice,
    crop_to_largest_bounding_box,
    apply_gaussian_smoothing)

class TestPreprocessFunctions(unittest.TestCase):
    
    def setUpNifti(self):
        """Create a temporary NIfTI file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "test.nii.gz")
        
        data = np.random.rand(64, 64, 64)
        affine = np.eye(4)
        self.nii_img = nib.Nifti1Image(data, affine)
        nib.save(self.nii_img, self.file_path)
    
    def tearDown(self):
        """Cleanup temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_load_nii_valid(self):
        """Test loading a valid NIfTI file."""
        nii = load_nii(self.file_path)
        self.assertIsInstance(nii, nib.Nifti1Image)
    
    def test_load_nii_invalid_path(self):
        """Test loading from an invalid path."""
        with self.assertRaises(FileNotFoundError):
            load_nii("invalid_path.nii.gz")
    
    def test_load_nii_invalid_type(self):
        """Test passing a non-string file path."""
        with self.assertRaises(TypeError):
            load_nii(123)
    
    def test_get_data_valid(self):
        """Test extracting data from a valid NIfTI image."""
        nii = load_nii(self.file_path)
        data = get_data(nii)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (64, 64, 64))
    
    def test_get_data_invalid_type(self):
        """Test passing a non-NIfTI object."""
        with self.assertRaises(TypeError):
            get_data(np.array([1, 2, 3]))
    
    def test_get_affine_valid(self):
        """Test extracting affine matrix from a valid NIfTI image."""
        nii = load_nii(self.file_path)
        affine = get_affine(nii)
        self.assertIsInstance(affine, np.ndarray)
        self.assertEqual(affine.shape, (4, 4))
    
    def test_get_affine_invalid_type(self):
        """Test passing a non-NIfTI object."""
        with self.assertRaises(TypeError):
            get_affine(np.array([1, 2, 3]))
    
    def test_match_dimensions_crop(self):
        """Test cropping a larger image to match dimensions."""
        original = np.random.rand(32, 32, 32)
        modified = np.random.rand(64, 64, 64)  # Larger image
        result = match_dimensions(original, modified)
        self.assertEqual(result.shape, original.shape)
    
    def test_match_dimensions_pad(self):
        """Test padding a smaller image to match dimensions."""
        original = np.random.rand(64, 64, 64)
        modified = np.random.rand(32, 32, 32)  # Smaller image
        result = match_dimensions(original, modified)
        self.assertEqual(result.shape, original.shape)
    
    def test_resample_image(self):
        # Valid resampling
        resampled_nib = resample_image(self.nii_img, voxel_size=(2, 2, 2), output_format='nib')
        self.assertIsInstance(resampled_nib, nib.Nifti1Image)

        resampled_np = resample_image(self.nii_img, voxel_size=(2, 2, 2), output_format='numpy')
        self.assertIsInstance(resampled_np, np.ndarray)
        
        # Invalid voxel size
        with self.assertRaises(ValueError):
            resample_image(self.nii_img, voxel_size=(2, 2), output_format='numpy')
        
        # Invalid output format
        with self.assertRaises(ValueError):
            resample_image(self.nii_img, voxel_size=(2, 2, 2), output_format='invalid')

    def setUpData(self):
        # Create a dummy 3D MRI scan (random data for testing)
        self.data = np.random.rand(128, 128, 64).astype(np.float32)  # Example shape

    def test_normalize_data(self):
        # Z-score normalization
        normalized = normalize_data(self.data, method='z-score')
        self.assertAlmostEqual(np.mean(normalized), 0, delta=0.1)
        self.assertAlmostEqual(np.std(normalized), 1, delta=0.1)
        
        # Min-max normalization
        normalized_minmax = normalize_data(self.data, method='min-max', min_val=0, max_val=1)
        self.assertGreaterEqual(normalized_minmax.min(), 0)
        self.assertLessEqual(normalized_minmax.max(), 1)
        
        # Invalid method
        with self.assertRaises(ValueError):
            normalize_data(self.data, method='invalid')
        
        # Zero variation error
        constant_data = np.ones((10, 10, 10))
        with self.assertRaises(ZeroDivisionError):
            normalize_data(constant_data, method='z-score')
        
        with self.assertRaises(ZeroDivisionError):
            normalize_data(constant_data, method='min-max')

    def test_extract_brain(self):
        # Extract brain with valid modality
        result = extract_brain(self.data, modality='t1', what_to_return={'extracted_brain': 'numpy'})
        self.assertIn('extracted_brain', result)
        self.assertIsInstance(result['extracted_brain'], np.ndarray)

        # Invalid input type
        with self.assertRaises(TypeError):
            extract_brain("invalid_input")

        # Invalid return key
        with self.assertRaises(ValueError):
            extract_brain(self.data, what_to_return={'invalid_key': 'numpy'})


    def setUpMask(self):
        np.random.seed(42)
        self.test_mask = np.zeros((100, 100, 50))
        self.test_mask[30:70, 30:70, 25] = 1  # Largest brain mask in slice 25
        self.test_mask[40:60, 40:60, 10] = 1  # Smaller brain mask in slice 10
        
        self.test_data = np.random.rand(100, 100, 50)  # Random MRI data
    
    def test_get_largest_brain_mask_slice(self):
        binary_mask, largest_slice_idx = get_largest_brain_mask_slice(self.test_mask)
        self.assertEqual(largest_slice_idx, 25)
        self.assertTrue(binary_mask.shape == self.test_mask.shape)

    def test_crop_to_largest_bounding_box(self):
        cropped_data = crop_to_largest_bounding_box(self.test_data, mask=self.test_mask)
        self.assertEqual(cropped_data.shape[:2], (40, 40))  # Bounding box should be 40x40
        self.assertEqual(cropped_data.shape[2], self.test_data.shape[2])  # Same depth
    
    def test_apply_gaussian_smoothing(self):
        smoothed_data = apply_gaussian_smoothing(self.test_data, sigma=1.5)
        self.assertEqual(smoothed_data.shape, self.test_data.shape)
        self.assertFalse(np.array_equal(smoothed_data, self.test_data))  # Should be modified        


if __name__ == "__main__":
    unittest.main()
