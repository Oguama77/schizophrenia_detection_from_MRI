import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class SVMVisualizer:
    def __init__(self, results_json_path):
        """
        Initializes the visualization class and loads classification results.

        Parameters:
            results_json_path (str): Path to the JSON file containing classification results.
        """
        self.results_json_path = results_json_path
        self.results = self._load_results()

    def _load_results(self):
        """Loads classification results from the JSON file."""
        if not os.path.exists(self.results_json_path):
            raise FileNotFoundError(f"Results file not found: {self.results_json_path}")

        with open(self.results_json_path, "r") as f:
            return json.load(f)

    def plot_confusion_matrix(self):
        """Plots the confusion matrix."""
        y_test = np.array(self.results["y_test"])
        test_preds = np.array(self.results["test_preds"])
        conf_matrix = np.array(self.results["confusion_matrix"])

        # Get unique class names if labels are more than just 0 and 1
        unique_classes = sorted(set(y_test))

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=unique_classes, yticklabels=unique_classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self):
        """Plots the ROC curve."""
        y_test = np.array(self.results["y_test"])
        test_scores = np.array(self.results["test_scores"])  # Scores should be probabilities
        
        # Handle binary classification vs. multi-class
        if test_scores.ndim == 1:
            fpr, tpr, _ = roc_curve(y_test, test_scores)
        else:
            fpr, tpr, _ = roc_curve(y_test, test_scores[:, 1])  # Only take positive class scores

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
