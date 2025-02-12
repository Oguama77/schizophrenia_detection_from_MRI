import torch
import torch.nn as nn
import numpy as np
import joblib
from torchvision import models
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ResNet-18 for Feature Extraction
class FeatureExtractor(nn.Module):
    """
    A feature extraction module based on a pre-trained model (ResNet18).
    This module removes the classification layer and modifies the first convolutional 
    layer to support a custom number of input channels.

    Attributes:
        feature_extractor (nn.Sequential): The modified backbone model without the classification head.
    """

    def __init__(self,
                 base_model_name: str = 'resnet18',
                 weights=models.ResNet18_Weights.IMAGENET1K_V1,
                 input_channels: int = 1) -> None:
        """
        Initializes the FeatureExtractor.

        Args:
            base_model_name (str): The name of the base model to use (default: 'resnet18').
            weights (torchvision.models.Weights): Pretrained weights for the base model.
            input_channels (int): Number of input channels for the first convolutional layer.
        
        Returns:
            None
        """
        super(FeatureExtractor, self).__init__()
        base_model = getattr(models, base_model_name)(weights=weights)
        # Remove the classification layer (fully connected layer)
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1])

        # Modify the first convolutional layer to match the desired number of input channels
        conv1 = self.feature_extractor[0]
        self.feature_extractor[0] = nn.Conv2d(in_channels=input_channels,
                                              out_channels=conv1.out_channels,
                                              kernel_size=conv1.kernel_size,
                                              stride=conv1.stride,
                                              padding=conv1.padding,
                                              bias=conv1.bias)
        # Copy the weights from the first channel of the original conv1 layer
        self.feature_extractor[0].weight.data = conv1.weight.data.mean(
            dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Extracted features in shape (batch_size, feature_dim).
        """
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, feature_dim)


# Support Vector Classifier for image classification
class SVMClassifier:
    """
    A Support Vector Machine (SVM) classifier with built-in preprocessing (StandardScaler).

    This class encapsulates the SVM model, allowing easy training, prediction, 
    and saving/loading of trained models.

    Attributes:
        pipeline (sklearn.pipeline.Pipeline): A pipeline containing StandardScaler and SVC.
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 100,
                 gamma: float = 0.0001,
                 random_state: int = 42) -> None:
        """
        Initializes the SVM classifier with specified hyperparameters.

        Args:
            kernel (str): Specifies the kernel type to be used in the SVM (default: 'rbf').
            C (float): Regularization parameter (default: 100).
            gamma (float): Kernel coefficient (default: 0.0001).
            random_state (int): Controls randomness (default: 42).
        """
        self.pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel,
                probability=True,
                C=C,
                gamma=gamma,
                random_state=random_state))

    def train(self, train_features: np.ndarray,
              train_labels: np.ndarray) -> None:
        """
        Trains the SVM classifier on the given training data.

        Args:
            train_features (np.ndarray): Feature matrix of shape (num_samples, num_features).
            train_labels (np.ndarray): Target labels of shape (num_samples,).
        """
        self.pipeline.fit(train_features, train_labels)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given test features.

        Args:
            test_features (np.ndarray): Feature matrix of shape (num_samples, num_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.pipeline.predict(test_features)

    def predict_proba(self, test_features: np.ndarray) -> np.ndarray:
        """
        Returns probability estimates for the given test features.

        Args:
            test_features (np.ndarray): Feature matrix of shape (num_samples, num_features).

        Returns:
            np.ndarray: Probability estimates for each class.
        """
        return self.pipeline.predict_proba(test_features)

    def save_model(self, file_path: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            file_path (str): Path to save the model.
        """
        joblib.dump(self.pipeline, file_path)

    @classmethod
    def load_model(cls, file_path: str) -> "SVMClassifier":
        """
        Loads a trained model from a file.

        Args:
            file_path (str): Path to the saved model file.

        Returns:
            SVMClassifier: A new instance of SVMClassifier with the loaded model.
        """
        instance = cls()
        instance.pipeline = joblib.load(file_path)
        return instance
