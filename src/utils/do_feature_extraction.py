import os
import tqdm
import torch
import numpy as np
import pandas as pd
from google.colab import drive
from utils.def_data_loader import MRIDataset
from torch.utils.data import DataLoader
from models.models import FeatureExtractor

# Function to get image paths
def get_image_paths(folder_path):
    """Gets all .pt file paths from the specified folder."""
    image_paths = [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if file_name.endswith('.pt')
    ]
    return image_paths

# Collate function for data loaders
def collate_fn(batch):
    # Filter out invalid samples
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Handle empty batch case
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels)

# Function to extract features using the pretrained model
def extract_features(loader, model, device): # not used currently
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        # Wrap the DataLoader in tqdm for a progress bar
        for inputs, targets in tqdm(loader, desc="Extracting features"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def feature_extraction_pipeline(
        path_to_project_root_dir,
        path_to_train_set_dir, 
        path_to_test_set_dir,
        path_to_labels_file,
        ):
    """
    JUST NEED TO PROVIDE PATHS
    TO TRAINING AND TESTING FOLDERS
    AND A FILE FOR CLINICAL DATA FOR LABELS
    """
    drive.mount('/content/drive', force_remount=True)

    os.chdir("drive/My Drive/" + path_to_project_root_dir)

    paths_to_train_data_samples = get_image_paths(path_to_train_set_dir)
    paths_to_test_data_samples = get_image_paths(path_to_test_set_dir)
    labels_df = pd.read_csv(path_to_labels_file)

    train_dataset = MRIDataset(paths_to_train_data_samples, )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    test_dataset = MRIDataset(paths_to_test_data_samples, labels_df)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Load feature extractor and move to GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor(base_model_name='resnet18').to(device)

    # Extract or load features for training set # TODO: WHY DO WE NEED TO LOAD THE FEATURES???   
    print("Extracting features for training data...")
    train_features, train_labels = extract_features(train_loader, feature_extractor, device)
    np.save("train_features_sn.npy", train_features)
    np.save("train_labels_sn.npy", train_labels)
   
    # Extract features from test set
    print("Extracting features for test data...")
    test_features, test_labels = extract_features(test_loader, feature_extractor, device)
    np.save("test_features_sn.npy", test_features)
    np.save("test_labels_sn.npy", test_labels)
