import os
import json
import numpy as np
import torch
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

# TODO: ADD CORRECT FEATURES_DIR

def load_features():
    """Loads extracted features and labels."""
    train_features = np.load(os.path.join(FEATURES_DIR, "train_features.npy"))
    train_labels = np.load(os.path.join(FEATURES_DIR, "train_labels.npy"))
    test_features = np.load(os.path.join(FEATURES_DIR, "test_features.npy"))
    test_labels = np.load(os.path.join(FEATURES_DIR, "test_labels.npy"))
    return train_features, train_labels, test_features, test_labels

def train_and_evaluate():
    """Trains a classifier and evaluates it on the test set."""
    train_X, train_y, test_X, test_y = load_features()

    # Train classifier (Support Vector Machine)
    clf = make_pipeline(
        StandardScaler(), 
        SVC(
            kernel=svc_kernel,  # 'rbf'
            probability=True, 
            C = svc_c_value, # 100
            gamma = svc_gamma_value, # 0.0001
            random_state=42
            )
            )
    clf.fit(train_X, train_y)

    joblib.dump(clf, path_to_save_classifier) # "svm_classifier_bp_new.pkl"

    # Make predictions
    test_preds = clf.predict(test_X)
    test_probs = clf.predict_proba(test_X)[:, 1]  # Get probabilities for ROC AUC

    # Compute evaluation metrics
    acc = accuracy_score(test_y, test_preds)
    auc = roc_auc_score(test_y, test_probs)
    report = classification_report(test_y, test_preds, output_dict=True)
    conf_matrix = confusion_matrix(test_y, test_preds)

    # Save results for visualization
    results = {
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }

    results_path = os.path.join(FEATURES_DIR, "classification_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Classification results saved to {results_path}")
