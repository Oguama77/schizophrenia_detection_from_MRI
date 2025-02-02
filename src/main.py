from utils.preprocessing import preprocess_images
from utils.augmentation import augment_images
from utils.dataset_preparation import prepare_dataset

def main():
    # Define paths
    raw_data_dir = "data/raw"
    preprocessed_dir = "data/processed"
    augmented_dir = "data/fully_processed"
    train_set_dir = "data/train_set"
    
    # User Input
    num_train_images = 50
    preprocessing_steps = ["resample", "normalize", "extract_brain"]
    augmentation_type = "translation"  # Options: 'translation', 'rotation', 'gaussian_noise'
    num_augmentations = 5

    # Step 1: Preprocessing
    preprocess_images(raw_data_dir, preprocessed_dir, preprocessing_steps)

    # Step 2: Augmentation
    augment_images(preprocessed_dir, augmented_dir, augmentation_type, num_augmentations)

    # Step 3: Dataset Preparation
    prepare_dataset(augmented_dir, train_set_dir, num_train_images)

if __name__ == "__main__":
    main()
