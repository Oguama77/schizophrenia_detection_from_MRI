import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Function to extract subject number from the file name
def extract_subject_number(file_name):
    match = re.search(r'sub-(A\d+)', file_name)
    if match:
        return match.group(1)
    return None

# Class to return each slice and the label
class MRIDataset(Dataset):
    def __init__(self, image_paths, labels_file, target_shape=(128, 128)):
        """
        Args:
            image_paths (list): List of file paths to MRI tensors.
            labels_file (pd.DataFrame): DataFrame containing participant IDs and labels.
            target_shape (tuple): Target height and width for 2D slices.
        """
        self.image_paths = image_paths
        self.labels_df = labels_file
        self.target_shape = target_shape

    def __len__(self):
        """Calculates the total number of slices across all MRI files."""
        total_slices = 0
        for path in self.image_paths:
            mri_tensor = torch.load(path, weights_only=True)
            total_slices += mri_tensor.shape[0]  # Assume the first dimension is slices
        return total_slices

    def __getitem__(self, idx):
        """Returns a single resized 2D slice and its corresponding label."""
        # Determine which file and slice the index corresponds to
        accumulated_slices = 0
        file_path = None
        slice_index = None

        for path in self.image_paths:
            mri_tensor = torch.load(path, weights_only=True)
            num_slices = mri_tensor.shape[0]  # Number of slices in the current file
            if idx < accumulated_slices + num_slices:
                file_path = path
                slice_index = idx - accumulated_slices
                break
            accumulated_slices += num_slices

        if file_path is None:
            raise IndexError(f"Index {idx} is out of range.")

        # Extract subject number and label
        subject_number = extract_subject_number(file_path)
        label_row = self.labels_df[self.labels_df["participant_id"] == subject_number]
        if label_row.empty:
            #logging.warning(f"No label found for subject: {subject_number}")
            raise ValueError(f"Missing label for subject: {subject_number}")
        label = label_row["dx_encoded"].values[0]

        # Load the MRI tensor and extract the specific slice
        try:
            mri_tensor = torch.load(file_path, weights_only=True)
            slice_2d = mri_tensor[slice_index, :, :]  # Extract the 2D slice
            slice_2d = slice_2d.unsqueeze(0)  # Add channel dimension for PyTorch

            # Resize the slice to the target shape
            slice_2d = F.interpolate(slice_2d.unsqueeze(0), size=self.target_shape, mode="bilinear", align_corners=False)
            slice_2d = slice_2d.squeeze(0)  # Remove the batch dimension
        except Exception as e:
            #logging.error(f"Error loading file {file_path}: {e}")
            raise RuntimeError(f"Error processing file {file_path}: {e}")

        return slice_2d, label
    