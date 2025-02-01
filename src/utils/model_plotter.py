import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class SVMVisualizer:
    def __init__(self):
        """Initializes the visualization tools for SVM classification results."""
        pass
    
    def plot_confusion_matrix(self, true_labels, predicted_labels, class_names=None):
        """Plots the confusion matrix."""
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_roc_curve(self, true_labels, test_scores):
        """Plots the ROC curve."""
        fpr, tpr, _ = roc_curve(true_labels, test_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()