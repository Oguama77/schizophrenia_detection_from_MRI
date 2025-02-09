import pytest
import torch
import os
import pandas as pd
import glob
from unittest.mock import patch, MagicMock
from utils.def_preprocess import load_nii
from utils.def_data_loader import MRIDataset 

def mock_load_nii(file_path):
    """Mock function to simulate loading a NIfTI image."""
    return load_nii(torch.rand((1, 128, 128, 128)))  # Simulating a 3D tensor

def mock_preprocess_image(image):
    """Mock function to simulate preprocessing an image."""
    return image  # Assume it returns the same tensor

@patch("src.utils.data_loader", side_effect=mock_load_nii)
@patch("src.utils.preprocess", side_effect=mock_preprocess_image)
@patch("pandas.read_csv")
@patch("glob.glob")
def test_mri_dataset(mock_glob, mock_read_csv, mock_preprocess, mock_load_nii):
    """Test the MRIDataset class."""
    # Mock the CSV file
    mock_read_csv.return_value = pd.DataFrame({
        "participant_id": ["sub-A000001", "sub-A000002"],
        "dx_encoded": [0, 1]
    }).set_index("participant_id")

    # Mock the file paths found by glob
    mock_glob.return_value = [
        "/fake_path/schizconnect_COBRE_images_22613/COBRE/sub-A000001/ses-01/anat/image1.nii.gz",
        "/fake_path/schizconnect_COBRE_images_22613/COBRE/sub-A000002/ses-01/anat/image2.nii.gz"
    ]

    # Create dataset instance
    dataset = MRIDataset("/fake_path")

    # Test dataset length
    assert len(dataset) == 2

    # Test __getitem__ output
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 128, 128, 128)  # Shape of mock MRI image
    assert label in [0, 1]

    image, label = dataset[1]
    assert isinstance(image, torch.Tensor)
    assert label in [0, 1]
