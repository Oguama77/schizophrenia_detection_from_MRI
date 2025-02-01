import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from image_preprocessor import ImagePreprocessor
from image_augmentor import ImageAugmentor
from schizophrenia_eda import SchizophreniaEDA
from svm_classifier import SVMClassifier
from svm_visualizer import SVMVisualizer


def main(args):
    # Initialize components
    preprocessor = ImagePreprocessor()
    augmentor = ImageAugmentor()
    eda = SchizophreniaEDA()
    classifier = SVMClassifier()
    visualizer = SVMVisualizer()

    # Step 1: Load and preprocess images
    print("Loading and preprocessing images...")
    images, labels = preprocessor.load_data(args.data_path)
    processed_images = [preprocessor.process(image) for image in images]

    # Step 2: Data Augmentation (optional)
    if args.augment:
        print("Applying data augmentation...")
        augmented_images, augmented_labels = augmentor.augment_data(processed_images, labels)
        processed_images.extend(augmented_images)
        labels.extend(augmented_labels)

    # Step 3: Exploratory Data Analysis (EDA)
    print("Performing exploratory data analysis...")
    eda.perform_analysis(processed_images, labels)

    # Step 4: Train the SVM Classifier
    print("Training SVM classifier...")
    X_train, X_test, y_train, y_test = classifier.split_data(processed_images, labels)
    classifier.train(X_train, y_train)
    accuracy, report = classifier.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    # Step 5: Visualize SVM results
    print("Visualizing classifier results...")
    visualizer.plot_decision_boundary(X_train, y_train, classifier.model)
    visualizer.plot_confusion_matrix(y_test, classifier.predict(X_test))
    plt.show()

    # Step 6: Save trained model
    if args.save_model:
        classifier.save_model(args.model_path)
        print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schizophrenia MRI Classification Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MRI dataset")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--save_model", action="store_true", help="Save trained SVM model")
    parser.add_argument("--model_path", type=str, default="svm_model.pkl", help="Path to save the model")
    
    args = parser.parse_args()
    main(args)
