import os
import torch
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import DataLoader
from src.utils.data_loader import MRIDataset
from src.models.cnn import FeatureExtractor

# TODO: ADD CORRECT TRAIN_SET_DIR, TEST_SET_DIR, LABELS_FILE, FEATURE_EXTRACTOR_MODEL, BATCH_SIZE, FEATURES_DIR


# Collate function for DataLoader (handles empty batches)
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.tensor(labels)


# Function to extract features using a pretrained model
def extract_features(loader, model, device):
    """Extracts features from MRI images using the specified model."""
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Extracting features"):
            if batch is None:
                continue  # Skip empty batches

            inputs, targets = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


# Main feature extraction pipeline
def feature_extraction_pipeline(train_set_dir,
                                test_set_dir,
                                labels_dir,
                                extracted_features_dir,
                                batch_size,
                                feature_extractor_model,
                                weights,
                                input_channels: int = 1) -> None:
    os.makedirs(extracted_features_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = MRIDataset(train_set_dir, labels_dir)
    test_dataset = MRIDataset(test_set_dir, labels_dir)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor(
        base_model_name=feature_extractor_model,
        weights=weights,
        input_channels=input_channels).to(device)

    # Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(train_loader, model,
                                                    device)
    np.save(os.path.join(extracted_features_dir, "train_features.npy"), train_features)
    np.save(os.path.join(extracted_features_dir, "train_labels.npy"), train_labels)

    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, model, device)
    np.save(os.path.join(extracted_features_dir, "test_features.npy"), test_features)
    np.save(os.path.join(extracted_features_dir, "test_labels.npy"), test_labels)

    print("Feature extraction completed.")
