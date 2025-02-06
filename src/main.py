from logger import logger
import yaml

from utils.preprocessing import preprocess_images
from utils.augmentation import augment_images
from utils.dataset_preparation import prepare_dataset

def main():
    logger.info("The program started...")

    # Load configuration from a config file
    with open('config.yaml') as c:
        config_data = yaml.safe_load(c)
    
    # Extract the necessary configuration parameters
    # raw_data_dir = config_data['raw_data_dir']
    #

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

    # Step 1: Select the data (train, test)
    """
    Randomly divides (ratio is user defined) images located in /data/raw,
    Creates 2 folders: train_set, test_set,
    Copy-pastes previously divided images into these folders (e.g. 50 images into train, 12 images into test),
    Original images are kept intact in /data/raw.
    """
    prepare_dataset(augmented_dir, train_set_dir, num_train_images)
    logger.info("Dataset preparation completed.")

    # Step 2: Preprocessing
    """
    Preprocesses images located in /data/train_set,
    Resamples to a common voxel size,
    Normalizes intensity values,
    Extracts brain regions using a custom algorithm,
    Saves preprocessed images in /data/train_set/preprocessed.

    Test images located in /data/test_set are also resampled and normalized for consistency,
    Saves preprocessed test images in /data/test_set/preprocessed.
    """
    preprocess_images(raw_data_dir, preprocessed_dir, preprocessing_steps)
    logger.info("Preprocessing completed.")

    # Step 3: Augmentation
    """
    Applies specified augmentation type (translation, rotation, gaussian_noise, or combined)
    to the preprocessed images located in /data/train_set/preprocessed,
    Saves augmented images in /data/train_set/preprocessed/augmented.
    """
    augment_images(preprocessed_dir, augmented_dir, augmentation_type, num_augmentations)
    logger.info("Augmentation completed.")

    # Step 4: Feature extraction stage (ResNet)
    """
    
    Extracts features from both preprocessed and augmented images 
    located in /data/train_set/preprocessed/ and /data/train_set/preprocessed/augmented
    Saves the extracted features in /data/train_set/features.
    """

    # Step 5: Model training and validation (SVC)
    """
    Trains a machine learning model using the extracted features located in /data/train_set/features,
    Validates the trained model using the test set located in /data/test_set/preprocessed,
    Saves the trained model in /models.
    Saves the validation results in /results.
    """



    logger.info("The program completed successfully.")

if __name__ == "__main__":
    main()
