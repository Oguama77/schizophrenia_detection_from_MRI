import os
import json
import numpy as np
import logging
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from src.models.svm import SVMClassifier

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
#FEATURES_DIR = "path/to/save/features"
#MODEL_PATH = "path/to/save/model.pkl"


def load_features(
    features_dir: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads extracted features and labels from disk.

    Parameters:
        features_dir (str): Path to the directory containing feature files.

    Returns:
        tuple: (train_features, train_labels, test_features, test_labels)

    Raises:
        FileNotFoundError: If any of the required feature files are missing.
    """
    try:
        train_features = np.load(os.path.join(features_dir, "train_features.npy"))
        train_labels = np.load(os.path.join(features_dir, "train_labels.npy"))
        test_features = np.load(os.path.join(features_dir, "test_features.npy"))
        test_labels = np.load(os.path.join(features_dir, "test_labels.npy"))
    except FileNotFoundError as e:
        logging.error(f"Feature file not found: {e}")
        raise

    return train_features, train_labels, test_features, test_labels


def train_and_evaluate(extracted_features_dir: str,
                       dir_to_save_clf: str,
                       results_output_dir: str,
                       clf_kernel,
                       clf_c_value,
                       clf_gamma_value) -> None:
    """
    Trains an SVM classifier on extracted features and evaluates it on a test set.

    Parameters:
        model_path (str): Path to save the trained model.
        features_dir (str): Path where feature files are stored.

    Returns:
        None
    """
    logging.info("Loading extracted features...")
    X_train, y_train, X_test, y_test = load_features(extracted_features_dir)

    # Train SVM classifier
    logging.info("Training SVM classifier...")
    classifier = SVMClassifier(kernel=clf_kernel,
                               C=clf_c_value,
                               gamma=clf_gamma_value,
                               random_state=42)
    classifier.train(X_train, y_train)

    # Save trained model
    classifier.save_model(dir_to_save_clf)
    logging.info(f"Model saved to {dir_to_save_clf}")

    # Make predictions and get decision function for ROC AUC
    test_preds = classifier.predict(X_test)
    test_scores = classifier.decision_function(X_test)

    # Compute evaluation metrics
    acc = accuracy_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_scores)
    report = classification_report(y_test, test_preds, output_dict=True)
    conf_matrix = confusion_matrix(y_test, test_preds)

    # Save results
    results = {
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }

    results_path = os.path.join(extracted_features_dir, results_output_dir+'.json') # "classification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Classification results saved to {results_path}")
