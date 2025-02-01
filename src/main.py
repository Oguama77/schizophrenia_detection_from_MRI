import argparse
import matplotlib.pyplot as plt
from utils.preprocess import ImagePreprocessor
from utils.augmentation import ImageAugmentor
from utils.data_visualization import SchizophreniaEDA
from models.models import SVMClassifier
from models.models import FeatureExtractor
from utils.model_plotter import SVMVisualizer


def main(args):
    # Initialize components
    preprocessor = ImagePreprocessor()
    augmentor = ImageAugmentor()
    eda = SchizophreniaEDA()
    feature_extractor= FeatureExtractor()
    svm_classifier = SVMClassifier()
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
    
    # Step 4: Extract features with ResNet-18
    print("Extracting features...")
    train_features, train_labels = extract_features(train_loader, feature_extractor, device)
    
    # Step 5: Train the SVM Classifier
    print("Training SVM classifier...")
    svm_classifier.pipeline()
    svm_classifier.train(train_features, train_labels)
    predicted_labels = svm_classifier.predict(test_features)
    test_scores = svm_classifier.predict_proba(test_features)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    # Step 6: Visualize SVM results
    print("Visualizing classifier results...")
    visualizer.plot_confusion_matrix(test_labels, predicted_labels)
    visualizer.plot_roc_curve(test_labels, test_scores)
    plt.show()

    # Step 7: Save trained model
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
