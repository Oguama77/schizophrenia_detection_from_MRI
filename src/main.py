from logger import logger
from omegaconf import OmegaConf
from utils.eda import SchizophreniaEDA
from utils.dataset_preparation import prepare_dataset
from utils.preprocessing import preprocess_images
from utils.augmentations import augment_images
from utils.feature_extraction import feature_extraction_pipeline
from utils.classifier import feature_classification_pipeline


def main():
    logger.info("The program started...")

    # Load configuration from a config file
    config_data = OmegaConf.load('config.yaml')

    # Accessing parameters # TODO: USE CAPITALS FOR CONSTANTS
    CLINICAL_DATA_DIR = config_data.file_paths.clinical_data_dir
    RAW_PT_DATA_DIR = config_data.file_paths.raw_pt_data_dir
    RAW_NII_DATA_DIR = config_data.file_paths.raw_nii_data_dir
    PREPROCESSED_DATA_DIR = config_data.file_paths.preprocessed_dir
    AUGMENTED_DATA_DIR = config_data.file_paths.augmented_dir
    TRAIN_SET_DIR = config_data.file_paths.train_set_dir
    TEST_SET_DIR = config_data.file_paths.test_set_dir

    PREPROCESSING_STEPS = config_data.dataset_preparation.preprocessing_steps  # ["resample", "normalize", "extract_brain"]
    AUGMENTATION_STEPS = config_data.dataset_preparation.augmentation_steps  # "translation"  # Options: 'translation', 'rotation', 'gaussian_noise'
    HOW_MANY_AUGMENTATIONS = config_data.dataset_preparation.how_many_augmentations  # 4
    TRAIN_SET_RATIO = config_data.dataset_preparation.train_set_ratio  # 0.24: 50 - train, 12 - test
    NORMALIZE_DEFAULT = config_data.dataset_preparation.normalize_default  # True False
    NORMALIZATION_METHOD = config_data.dataset_preparation.normalization_method  # min-max

    NORMALIZE_WHEN_PREPROCESS = config_data.preprocessing_params.normalize_when_preprocess
    DO_BRAIN_EXTRACTION = config_data.preprocessing_params.do_brain_exaction
    DO_CROP = config_data.preprocessing_params.do_crop
    DO_SMOOTH = config_data.preprocessing_params.do_smooth
    DO_RE_NORMALIZE_AFTER_SMOOTH = config_data.preprocessing_params.do_re_normalize_after_smooth
    PREPROCESS_TEST_SET = config_data.preprocessing_params.preprocess_test_set

    TARGET_SHAPE = config_data.preprocessing_params.target_shape
    VOXEL_SIZE = config_data.preprocessing_params.voxel_size
    ORDER = config_data.preprocessing_params.order
    MODE = config_data.preprocessing_params.mode
    CVAL = config_data.preprocessing_params.cval
    MIN_VAL = config_data.preprocessing_params.min_val
    MAX_VAL = config_data.preprocessing_params.max_val
    EPS = config_data.preprocessing_params.eps
    MODALITY = config_data.preprocessing_params.modality
    VERBOSE = config_data.preprocessing_params.verbose
    SIGMA = config_data.preprocessing_params.sigma
    ORDER = config_data.preprocessing_params.order
    MODE = config_data.preprocessing_params.mode
    CVAL = config_data.preprocessing_params.cval
    TRUNCATE = config_data.preprocessing_params.truncate
    OUTPUT_FORMAT = config_data.preprocessing_params.output_format

    BASE_MODEL_NAME = config_data.resnet_params.base_model_name
    BASE_MODEL_WEIGHTS = config_data.resnet_params.base_model_weights
    INPUT_CHANNELS = config_data.resnet_params.input_channels

    KERNEL = config_data.svc_params.kernel
    C = config_data.svc_params.C
    GAMMA = config_data.svc_params.gamma
    RANDOM_STATE = config_data.svc_params.random_state

    # Step 0: Explanatory data analysis
    """
    Performs an explanatory data analysis on the raw images and corresponding clinical data.
    This step is optional and can be skipped if not necessary.
    """
    eda = SchizophreniaEDA(CLINICAL_DATA_DIR, RAW_PT_DATA_DIR)
    eda.perform_eda(CLINICAL_DATA_DIR)
    eda.save_raw_images_metrics()
    logger.info("EDA completed.")

    # Step 1: Select the data (train, test)
    """
    Randomly divides (ratio is user defined) images located in /data/raw,
    Creates 2 folders: train_set, test_set,
    Copy-pastes previously divided images into these folders (e.g. 50 images into train, 12 images into test),
    Original images are kept intact in /data/raw.

    UPDATE: for smoother workflow, before copying the images, it resamples and normalizes them.
    """
    prepare_dataset(train_ratio=TRAIN_SET_RATIO,
                    raw_pt_dir=RAW_PT_DATA_DIR,
                    raw_nii_dir=RAW_NII_DATA_DIR,
                    normalize=NORMALIZE_DEFAULT,
                    norm_method=NORMALIZATION_METHOD)
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
    preprocess_images(normalize=NORMALIZE_WHEN_PREPROCESS,
                      brain_extraction=DO_BRAIN_EXTRACTION,
                      crop=DO_CROP,
                      smooth=DO_SMOOTH,
                      re_normalize_after_smooth=DO_RE_NORMALIZE_AFTER_SMOOTH,
                      preprocess_test=PREPROCESS_TEST_SET)
    logger.info("Preprocessing completed.")

    # Step 3: Augmentation
    """
    Applies specified augmentation type (translation, rotation, gaussian_noise, or combined)
    to the preprocessed images located in /data/train_set/preprocessed,
    Saves augmented images in /data/train_set/preprocessed/augmented.
    """
    augment_images(
        augmentations=[
            ("translation", {
                "shift": (5, 5)
            }),
            ("rotation", {
                "angle": 15
            }),
            ("gaussian_noise", {
                "std": 0.1
            }),
        ],
        num_augmentations=HOW_MANY_AUGMENTATIONS,
        input_dir=PREPROCESSED_DATA_DIR,
        output_dir=AUGMENTED_DATA_DIR,
    )
    logger.info("Augmentation completed.")

    # Step 4: Feature extraction stage (ResNet)
    """
    
    Extracts features from both preprocessed and augmented images 
    located in /data/train_set/preprocessed/ and /data/train_set/preprocessed/augmented
    Saves the extracted features in /data/train_set/features.
    """
    feature_extraction_pipeline()
    logger.info("Feature extraction completed.")

    # Step 5: Model training and validation (SVC)
    """
    Trains a machine learning model using the extracted features located in /data/train_set/features,
    Validates the trained model using the test set located in /data/test_set/preprocessed,
    Saves the trained model in /models.
    Saves the validation results in /results.
    """
    feature_classification_pipeline()
    logger.info("Classifier training and validation completed.")

    # Step 6: Plot model (SVC) metrics
    """
    Plots model metrics (accuracy, precision, recall, f1-score) on the test set.
    """

    logger.info("The program completed successfully.")


if __name__ == "__main__":
    main()
