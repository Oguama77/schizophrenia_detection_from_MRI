# Import libraries
import unittest
import os
import nibabel as nib
import numpy as np
from unittest.mock import patch
from tempfile import NamedTemporaryFile
from unittest.mock import patch, MagicMock

# Import functions
from utils.preprocess import load_nii 
from utils.preprocess import get_data
from utils.preprocess import get_affine
from utils.preprocess import match_dimensions
from utils.preprocess import resample_image
from utils.preprocess import normalize_data, crop_to_largest_bounding_box, get_largest_brain_mask_slice
from utils.preprocess import get_largest_brain_mask_slice
from utils.preprocess import apply_gaussian_smoothing


class TestLoadNii(unittest.TestCase):

    def setUp(self):
        """Create a temporary valid NIfTI file for testing."""
        self.temp_nii_file = NamedTemporaryFile(delete=False, suffix=".nii")
        data = np.zeros((10, 10, 10))  # Dummy data
        affine = np.eye(4)  # Identity affine
        nii_img = nib.Nifti1Image(data, affine)
        nib.save(nii_img, self.temp_nii_file.name)

    def tearDown(self):
        """Clean up the temporary file."""
        if os.path.exists(self.temp_nii_file.name):
            os.remove(self.temp_nii_file.name)

    def test_load_valid_nii(self):
        """Test loading a valid NIfTI file."""
        nii = load_nii(self.temp_nii_file.name)
        self.assertIsInstance(nii, nib.Nifti1Image)

    def test_invalid_file_path_type(self):
        """Test TypeError is raised for a non-string file path."""
        with self.assertRaises(TypeError):
            load_nii(123)  # Passing an integer instead of a string

    def test_non_existent_file(self):
        """Test FileNotFoundError for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_nii("non_existent_file.nii")

    @patch("nibabel.load", side_effect=nib.filebasedimages.ImageFileError)
    def test_invalid_nii_file(self, mock_nib_load):
        """Test ValueError is raised for an invalid NIfTI file."""
        with self.assertRaises(ValueError):
            load_nii(self.temp_nii_file.name)


class TestGetData(unittest.TestCase):
    
    def test_get_data_valid(self):
        # Mocking a valid nib.Nifti1Image with some data
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.get_fdata.return_value = np.array([[1, 2], [3, 4]])

        # Calling the function with the mocked NIfTI image
        result = get_data(mock_nii)

        # Assert the returned data is correct
        np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))
    
    def test_get_data_invalid_type(self):
        # Mocking an object that is not a Nifti1Image
        mock_nii = MagicMock()
        
        # Check if TypeError is raised for invalid input
        with self.assertRaises(TypeError):
            get_data(mock_nii)
    
    def test_get_data_empty(self):
        # Mocking a Nifti1Image with empty data
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.get_fdata.return_value = np.array([])

        # Check if ValueError is raised for empty data
        with self.assertRaises(ValueError):
            get_data(mock_nii)
    
    def test_get_data_non_finite(self):
        # Mocking a Nifti1Image with non-finite data (NaN or Inf)
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.get_fdata.return_value = np.array([[1, 2], [3, np.nan]])

        # Check if ValueError is raised for non-finite values
        with self.assertRaises(ValueError):
            get_data(mock_nii)
    
    def test_get_data_exception_in_get_fdata(self):
        # Mocking a Nifti1Image where get_fdata throws an exception
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.get_fdata.side_effect = Exception("Some error")

        # Check if ValueError is raised when an exception is thrown by get_fdata
        with self.assertRaises(ValueError):
            get_data(mock_nii)
            

class TestGetAffine(unittest.TestCase):
    
    def test_get_affine_valid(self):
        # Mocking a valid nib.Nifti1Image with a valid affine matrix
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.affine = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Calling the function with the mocked NIfTI image
        result = get_affine(mock_nii)

        # Assert the returned affine matrix is correct
        np.testing.assert_array_equal(result, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    
    def test_get_affine_invalid_type(self):
        # Mocking an object that is not a Nifti1Image
        mock_nii = MagicMock()
        
        # Check if TypeError is raised for invalid input
        with self.assertRaises(TypeError):
            get_affine(mock_nii)
    
    def test_get_affine_missing_affine(self):
        # Mocking a Nifti1Image without the affine attribute
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.affine = None  # Simulating missing affine attribute

        # Check if ValueError is raised for missing affine attribute
        with self.assertRaises(ValueError):
            get_affine(mock_nii)
    
    def test_get_affine_invalid_affine_type(self):
        # Mocking a Nifti1Image with an affine attribute that is not a numpy array
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.affine = "Invalid affine matrix"  # Simulating invalid affine type

        # Check if ValueError is raised for incorrect affine type
        with self.assertRaises(ValueError):
            get_affine(mock_nii)
    
    def test_get_affine_invalid_shape(self):
        # Mocking a Nifti1Image with an affine matrix of incorrect shape
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.affine = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Check if ValueError is raised for incorrect affine shape
        with self.assertRaises(ValueError):
            get_affine(mock_nii)
    
    def test_get_affine_non_finite_values(self):
        # Mocking a Nifti1Image with an affine matrix containing non-finite values
        mock_nii = MagicMock(spec=nib.Nifti1Image)
        mock_nii.affine = np.array([[np.nan, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Check if ValueError is raised for non-finite values in the affine matrix
        with self.assertRaises(ValueError):
            get_affine(mock_nii)


class TestMatchDimensions(unittest.TestCase):
    
    def test_match_dimensions_padding(self):
        # original_image is larger than modified_image, so padding is required
        original_image = np.zeros((5, 5, 5))
        modified_image = np.ones((3, 3, 3))  # smaller than original

        # The result should be a padded modified_image of shape (5, 5, 5)
        result = match_dimensions(original_image, modified_image)

        # Assert that the result shape matches the original image's shape
        self.assertEqual(result.shape, original_image.shape)
        # Assert that the padded regions are zero
        self.assertTrue(np.all(result[:3, :3, :3] == 1))  # Original data
        self.assertTrue(np.all(result[3:, :, :] == 0))  # Padded data
        self.assertTrue(np.all(result[:, 3:, :] == 0))  # Padded data
        self.assertTrue(np.all(result[:, :, 3:] == 0))  # Padded data

    def test_match_dimensions_cropping(self):
        # modified_image is larger than original_image, so cropping is required
        original_image = np.ones((3, 3, 3))
        modified_image = np.ones((5, 5, 5))  # larger than original

        # The result should be a cropped modified_image of shape (3, 3, 3)
        result = match_dimensions(original_image, modified_image)

        # Assert that the result shape matches the original image's shape
        self.assertEqual(result.shape, original_image.shape)
        # Assert that the result data is unchanged from the original modified image region
        self.assertTrue(np.all(result == 1))  # Since the original was 1's and it's cropped

    def test_match_dimensions_same_size(self):
        # modified_image is the same size as original_image, no change needed
        original_image = np.ones((3, 3, 3))
        modified_image = np.ones((3, 3, 3))  # same as original

        # The result should be the same as modified_image
        result = match_dimensions(original_image, modified_image)

        # Assert that the result is the same as modified_image
        np.testing.assert_array_equal(result, modified_image)

    def test_match_dimensions_edge_case_empty(self):
        # original_image and modified_image are both empty
        original_image = np.zeros((0, 0, 0))
        modified_image = np.zeros((0, 0, 0))

        # The result should be empty as well
        result = match_dimensions(original_image, modified_image)

        # Assert that the result is empty
        self.assertEqual(result.shape, (0, 0, 0))

    def test_match_dimensions_non_zero_padding(self):
        # original_image is larger, but with non-zero values
        original_image = np.ones((4, 4, 4))
        modified_image = np.ones((2, 2, 2))  # smaller than original

        # The result should be a padded modified_image of shape (4, 4, 4)
        result = match_dimensions(original_image, modified_image)

        # Assert that the result shape matches the original image's shape
        self.assertEqual(result.shape, original_image.shape)
        # Assert that the padded region is zero
        self.assertTrue(np.all(result[:2, :2, :2] == 1))  # Original data
        self.assertTrue(np.all(result[2:, :, :] == 0))  # Padded data
        self.assertTrue(np.all(result[:, 2:, :] == 0))  # Padded data
        self.assertTrue(np.all(result[:, :, 2:] == 0))  # Padded data


class TestResampleImage(unittest.TestCase):

    @patch('your_module.resample_to_output')  # Mocking the resample_to_output function
    def test_resample_image_numpy(self, mock_resample):
        # Mocking the return value of resample_to_output to return a fake resampled Nifti image
        mock_resampled_nii = MagicMock(spec=nib.Nifti1Image)
        mock_resampled_nii.get_fdata.return_value = np.ones((4, 4, 4))  # Simulated resampled data

        mock_resample.return_value = mock_resampled_nii  # Mock resample_to_output to return our mocked image

        original_data = np.ones((3, 3, 3))  # Original data as a numpy array

        # Testing resampling to a numpy array
        result = resample_image(original_data, voxel_size=(2, 2, 2), output_format='numpy')

        # Check if the result is a numpy array and has the expected shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4, 4))  # Expected shape after resampling

    @patch('your_module.resample_to_output')
    def test_resample_image_nib(self, mock_resample):
        # Mocking the return value of resample_to_output to return a fake resampled Nifti image
        mock_resampled_nii = MagicMock(spec=nib.Nifti1Image)
        mock_resample.return_value = mock_resampled_nii  # Mock resample_to_output to return our mocked image

        original_data = np.ones((3, 3, 3))  # Original data as a numpy array

        # Testing resampling to a Nifti image object
        result = resample_image(original_data, voxel_size=(2, 2, 2), output_format='nib')

        # Check if the result is a nib.Nifti1Image
        self.assertIsInstance(result, nib.Nifti1Image)

    def test_invalid_voxel_size(self):
        original_data = np.ones((3, 3, 3))

        # Testing invalid voxel_size (not a tuple of length 3)
        with self.assertRaises(ValueError):
            resample_image(original_data, voxel_size=(2, 2), output_format='numpy')

        # Testing voxel_size with negative values
        with self.assertRaises(ValueError):
            resample_image(original_data, voxel_size=(-2, 2, 2), output_format='numpy')

        # Testing non-numeric values in voxel_size
        with self.assertRaises(ValueError):
            resample_image(original_data, voxel_size=('a', 2, 2), output_format='numpy')

    def test_invalid_output_format(self):
        original_data = np.ones((3, 3, 3))

        # Testing invalid output_format
        with self.assertRaises(ValueError):
            resample_image(original_data, voxel_size=(2, 2, 2), output_format='invalid_format')

    @patch('your_module.resample_to_output')
    def test_runtime_error_in_resampling(self, mock_resample):
        # Simulating a runtime error during resampling
        mock_resample.side_effect = Exception("Resampling failed")

        original_data = np.ones((3, 3, 3))

        # Check if RuntimeError is raised during resampling
        with self.assertRaises(RuntimeError):
            resample_image(original_data, voxel_size=(2, 2, 2), output_format='nib')

    def test_type_error_invalid_data(self):
        # Testing for invalid data type
        with self.assertRaises(TypeError):
            resample_image("invalid_data", voxel_size=(2, 2, 2), output_format='numpy')


class TestNormalizeData(unittest.TestCase):

    def test_z_score_normalization(self):
        data = np.array([1, 2, 3, 4, 5])
        result = normalize_data(data, method='z-score')

        # Check if the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check if the mean of the result is close to 0 and std is close to 1
        self.assertAlmostEqual(np.mean(result), 0, places=5)
        self.assertAlmostEqual(np.std(result), 1, places=5)

    def test_min_max_normalization(self):
        data = np.array([1, 2, 3, 4, 5])
        result = normalize_data(data, method='min-max', min_val=0, max_val=255)

        # Check if the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check if the result's minimum is 0 and maximum is 255
        self.assertEqual(np.min(result), 0)
        self.assertEqual(np.max(result), 255)

    def test_invalid_method(self):
        data = np.array([1, 2, 3, 4, 5])

        # Testing invalid method
        with self.assertRaises(ValueError):
            normalize_data(data, method='invalid')

    def test_invalid_data_type(self):
        data = "not a numpy array"

        # Testing invalid data type (string instead of numpy array)
        with self.assertRaises(TypeError):
            normalize_data(data, method='min-max')

    def test_zero_variation_z_score(self):
        data = np.array([1, 1, 1, 1, 1])

        # Testing for zero standard deviation in z-score normalization
        with self.assertRaises(ZeroDivisionError):
            normalize_data(data, method='z-score')

    def test_zero_variation_min_max(self):
        data = np.array([1, 1, 1, 1, 1])

        # Testing for zero variation in min-max normalization
        with self.assertRaises(ZeroDivisionError):
            normalize_data(data, method='min-max')

    def test_min_max_custom_range(self):
        data = np.array([1, 2, 3, 4, 5])
        result = normalize_data(data, method='min-max', min_val=10, max_val=20)

        # Check if the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Check if the result's minimum is 10 and maximum is 20
        self.assertEqual(np.min(result), 10)
        self.assertEqual(np.max(result), 20)

    def test_eps_value_in_min_max(self):
        data = np.array([1, 1, 1, 1, 1])
        result = normalize_data(data, method='min-max', eps=1e-6)

        # Ensure that the function still works with a small epsilon even if there's zero variation
        self.assertEqual(np.min(result), 0)
        self.assertEqual(np.max(result), 0)


class TestGetLargestBrainMaskSlice(unittest.TestCase):

    def test_get_largest_brain_mask_slice_basic(self):
        """Test with a simple synthetic mask where the largest slice is known."""
        mask = np.zeros((5, 5, 5))
        
        # Create a larger mask region in slice index 2
        mask[1:4, 1:4, 2] = 1  
        
        # Create a smaller mask region in slice index 3
        mask[2:3, 2:3, 3] = 1  

        binary_mask, largest_slice_index = get_largest_brain_mask_slice(mask)

        # Verify the binary mask is correctly thresholded
        self.assertEqual(binary_mask.dtype, np.uint8)
        self.assertEqual(largest_slice_index, 2)  # Expected largest slice index

    def test_get_largest_brain_mask_slice_multiple_largest(self):
        """Test when multiple slices have the same largest area."""
        mask = np.zeros((5, 5, 5))
        
        # Two slices with equal mask size
        mask[1:4, 1:4, 1] = 1
        mask[1:4, 1:4, 3] = 1  

        binary_mask, largest_slice_index = get_largest_brain_mask_slice(mask)

        # The function should return the first occurrence of the largest area
        self.assertIn(largest_slice_index, [1, 3])

    def test_get_largest_brain_mask_slice_no_brain_mask(self):
        """Test with an empty mask (all zeros)."""
        mask = np.zeros((5, 5, 5))

        binary_mask, largest_slice_index = get_largest_brain_mask_slice(mask)

        self.assertEqual(largest_slice_index, -1)  # No valid slice found
        self.assertTrue((binary_mask == 0).all())  # Binary mask should remain empty

    def test_get_largest_brain_mask_slice_single_slice(self):
        """Test with a mask that contains a single slice with brain tissue."""
        mask = np.zeros((5, 5, 5))
        mask[1:4, 1:4, 2] = 1  # Single slice

        binary_mask, largest_slice_index = get_largest_brain_mask_slice(mask)

        self.assertEqual(largest_slice_index, 2)

    def test_get_largest_brain_mask_slice_invalid_input(self):
        """Test with invalid input types."""
        with self.assertRaises(TypeError):
            get_largest_brain_mask_slice("invalid_data")  # Not a numpy array

        with self.assertRaises(ValueError):
            get_largest_brain_mask_slice(np.array([1, 2, 3]))  # Not 3D


class TestCropToLargestBoundingBox(unittest.TestCase):

    def setUp(self):
        """Create sample 3D MRI data and brain mask for testing."""
        self.data = np.zeros((10, 10, 5))  # 3D MRI data
        self.mask = np.zeros((10, 10, 5))  # 3D brain mask
        
        # Add a brain region in slice index 2
        self.mask[2:7, 3:8, 2] = 1  # Bounding box from (2,3) to (6,7)

        # Add another region in slice index 3 (smaller)
        self.mask[4:6, 5:7, 3] = 1  

        # Process mask using the existing function
        self.processed_mask, self.largest_slice_idx = get_largest_brain_mask_slice(self.mask)

    def test_crop_basic_case(self):
        """Test that the function crops MRI data correctly based on bounding box."""
        cropped_data = crop_to_largest_bounding_box(self.data, 
                                                    processed_mask=self.processed_mask, 
                                                    largest_slice_idx=self.largest_slice_idx, 
                                                    mask=self.mask)

        # Expected shape: (5, 5, 5) → [rows: 2:7, cols: 3:8, all slices]
        self.assertEqual(cropped_data.shape, (5, 5, 5))

    def test_crop_no_brain_region(self):
        """Test when there is no brain region in the mask."""
        empty_mask = np.zeros((10, 10, 5))  # Empty mask
        processed_mask, largest_slice_idx = get_largest_brain_mask_slice(empty_mask)

        cropped_data = crop_to_largest_bounding_box(self.data, 
                                                    processed_mask=processed_mask, 
                                                    largest_slice_idx=largest_slice_idx, 
                                                    mask=empty_mask)

        # Since no valid mask, original dimensions should be unchanged
        self.assertEqual(cropped_data.shape, self.data.shape)

    def test_crop_with_automatic_mask_detection(self):
        """Test when processed_mask and largest_slice_idx are not provided."""
        cropped_data = crop_to_largest_bounding_box(self.data, mask=self.mask)

        self.assertEqual(cropped_data.shape, (5, 5, 5))  # Should match bounding box size

    def test_invalid_input(self):
        """Test with invalid input types."""
        with self.assertRaises(TypeError):
            crop_to_largest_bounding_box("invalid_data", mask=self.mask)  # Not a numpy array

        with self.assertRaises(ValueError):
            crop_to_largest_bounding_box(self.data, mask=np.array([1, 2, 3]))  # Not 3D


class TestGaussianSmoothing(unittest.TestCase):
    def setUp(self):
        # Create a sample 3D MRI-like dataset
        self.test_data = np.random.rand(10, 10, 10)  # 3D array with random values

    def test_output_shape(self):
        """Ensure the output shape is the same as the input shape"""
        smoothed_data = apply_gaussian_smoothing(self.test_data)
        self.assertEqual(smoothed_data.shape, self.test_data.shape)

    def test_data_modification(self):
        """Ensure that the function modifies the data (i.e., it smooths it)"""
        smoothed_data = apply_gaussian_smoothing(self.test_data)
        self.assertFalse(np.array_equal(smoothed_data, self.test_data))  # Data should change

    def test_zero_array(self):
        """Test with a zero-filled array to check for unexpected changes"""
        zero_data = np.zeros((10, 10, 10))
        smoothed_data = apply_gaussian_smoothing(zero_data)
        np.testing.assert_array_equal(smoothed_data, zero_data)  # Should remain zero

    def test_sigma_variation(self):
        """Ensure different sigma values produce different results"""
        smoothed_data_1 = apply_gaussian_smoothing(self.test_data, sigma=0.5)
        smoothed_data_2 = apply_gaussian_smoothing(self.test_data, sigma=2.0)
        self.assertFalse(np.array_equal(smoothed_data_1, smoothed_data_2))  # Smoothing should be different

    def test_invalid_input(self):
        """Check if the function properly handles invalid inputs"""
        with self.assertRaises(TypeError):
            apply_gaussian_smoothing("invalid input")  # Should raise an error for non-array input



if __name__ == '__main__':
    unittest.main()
    