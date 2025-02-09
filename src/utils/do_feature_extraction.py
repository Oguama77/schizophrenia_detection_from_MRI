import os
import tqdm
import torch
import numpy as np
import pandas as pd
from google.colab import drive
from utils.def_data_loader import MRIDataset
from torch.utils.data import DataLoader
from itertools import chain
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



####
def feature_extraction_pipeline():
    """
    JUST NEED TO PROVIDE PATHS
    TO TRAINING AND TESTING FOLDERS
    AND A FILE FOR CLINICAL DATA FOR LABELS
    """

    drive.mount('/content/drive', force_remount=True)

    os.chdir("drive/My Drive/python_ml")

    # FROM CONIFIG!!!
    folder_path_train1 = "best_pipeline_001"
    folder_path_train2 = "best_pipeline_002"
    folder_path_test = "best_pipeline_test_set"
    label_file = "new_clinical_data.csv"

    train_path1 = get_image_paths(folder_path_train1)
    train_path2 = get_image_paths(folder_path_train2)
    test_paths = get_image_paths(folder_path_test)
    labels_df = pd.read_csv(label_file)

    # Create datasets
    train_dataset1 = MRIDataset(train_path1, labels_df)
    train_dataset2 = MRIDataset(train_path2, labels_df)
    test_dataset = MRIDataset(test_paths, labels_df)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Create DataLoaders
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True, collate_fn=collate_fn)
    train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True, collate_fn=collate_fn)

    train_loader = chain(train_loader1, train_loader2) # TODO: must be randomly distributed


    # Load feature extractor and move to GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor(base_model_name='resnet18').to(device)

    # CONDITIONAL: LOAD FEATURES OT NOT
    # Extract features for training set
    print("Extracting features for training data...")
    train_features, train_labels = extract_features(train_loader, feature_extractor, device)
    np.save("train_features_sn.npy", train_features)
    np.save("train_labels_sn.npy", train_labels)

    train_features = np.load("train_features_sn.npy")
    train_labels = np.load("train_labels_sn.npy")

    # CONDITIONAL: LOAD FEATURES OR NOT
    # Extract features from test set
    print("Extracting features for test data...")
    test_features, test_labels = extract_features(test_loader, feature_extractor, device)
    np.save("test_features_sn.npy", test_features)
    np.save("test_labels_sn.npy", test_labels)


    test_features = np.load("test_features_sn.npy")
    test_labels = np.load("test_labels_sn.npy")

    pass
