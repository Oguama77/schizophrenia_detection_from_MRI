import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, image_paths, labels_file, target_shape=(192, 256, 256), transform=None):
        self.image_paths = image_paths
        self.labels_file = labels_file
        self.target_shape = target_shape
        self.transform = transform

        # Load the labels from CSV
        self.labels_df = labels_file

    def __len__(self):
        return sum(self.target_shape[0] for _ in self.image_paths)  # Total number of slices

    def __getitem__(self, idx):
      file_index = idx // self.target_shape[0]
      slice_index = idx % self.target_shape[0]
      file_path = self.image_paths[file_index]

      subject_label_cache = {}
      subject_number = extract_subject_number(file_path)

      label_row = self.labels_df[self.labels_df["participant_id"] == subject_number]
      if label_row.empty:
          logging.warning(f"No label found for subject: {subject_number}")
          raise ValueError(f"Missing label for subject: {subject_number}")  # Raise error instead of returning None
      label = label_row["dx_encoded"].values[0]

      try:
          img = nib.load(file_path)
          # Apply transformation if specified
          if self.transform:
              img_data = transform_nifti.resample_image(img)
              img_data = transform_nifti.normalize_data(img_data)

          if img_data.shape != self.target_shape:
              if sorted(img_data.shape) == sorted(self.target_shape):
                  permuted_axes = np.argsort(img_data.shape)
                  img_data = np.transpose(img_data, permuted_axes)
              else:
                  pass
              img_data = resize(img_data, self.target_shape, mode="constant", preserve_range=True)

          slice_2d = img_data[slice_index, :, :]
          slice_2d = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

      except Exception as e:
          logging.error(f"Error loading file {file_path}: {e}")
          raise RuntimeError(f"Error processing file {file_path}: {e}")  # Raise error instead of returning None

      return slice_2d, label