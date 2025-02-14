from logger import logger
from omegaconf import OmegaConf
from utils.eda import SchizophreniaEDA
from utils.dataset_preparation import prepare_dataset
from utils.preprocessing import preprocess_images
from utils.augment_images import augment_images
from utils.feature_extractor import feature_extraction_pipeline
from utils.classifier import train_and_evaluate


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
    IS_NORMALIZE_DEFAULT = config_data.dataset_preparation.normalize_default  # True False
    NORMALIZATION_METHOD = config_data.dataset_preparation.normalization_method  # min-max

    IS_NORMALIZE_WHEN_PREPROCESS = config_data.preprocessing_params.normalize_when_preprocess
    IS_BRAIN_EXTRACTION = config_data.preprocessing_params.do_brain_exaction
    IS_CROP = config_data.preprocessing_params.do_crop
    IS_SMOOTH = config_data.preprocessing_params.do_smooth
    IS_RE_NORMALIZE_AFTER_SMOOTH = config_data.preprocessing_params.do_re_normalize_after_smooth
    IS_PREPROCESS_TEST_SET = config_data.preprocessing_params.preprocess_test_set

    TARGET_SHAPE = config_data.preprocessing_params.target_shape
    VOXEL_SIZE = config_data.preprocessing_params.voxel_size
    #ORDER = config_data.preprocessing_params.order
    #MODE = config_data.preprocessing_params.mode
    #CVAL = config_data.preprocessing_params.cval
    MIN_VAL = config_data.preprocessing_params.min_val
    MAX_VAL = config_data.preprocessing_params.max_val
    #EPS = config_data.preprocessing_params.eps
    MODALITY = config_data.preprocessing_params.modality
    VERBOSE = config_data.preprocessing_params.verbose
    SIGMA = config_data.preprocessing_params.sigma
    ORDER = config_data.preprocessing_params.order
    MODE = config_data.preprocessing_params.mode
    CVAL = config_data.preprocessing_params.cval
    TRUNCATE = config_data.preprocessing_params.truncate

    TRANSLATION_SHIFT = config_data.augmentation_params.translation_shift
    ROTATION_ANGLE = config_data.augmentation_params.rotation_angle
    GAUSSIAN_NOISE_MEAN = config_data.augmentation_params.gaussian_noise_mean
    GAUSSIAN_NOISE_STD = config_data.augmentation_params.gaussian_noise_std

    BASE_MODEL_NAME = config_data.resnet_params.base_model_name
    BASE_MODEL_WEIGHTS = config_data.resnet_params.base_model_weights
    INPUT_CHANNELS = config_data.resnet_params.input_channels

    LABELS_DIR = config_data.feature_extraction_params.labels_dir
    EXTRACTED_FEATURES_DIR = config_data.feature_extraction_params.extracted_features_dir
    BATCH_SIZE = config_data.feature_extraction_params.batch_size

    SAVE_CLF_DIR = config_data.svc_params.save_clf_dir
    RESULTS_OUTPUT_DIR = config_data.svc_params.results_output_dir
    KERNEL = config_data.svc_params.kernel
    C = config_data.svc_params.C
    GAMMA = config_data.svc_params.gamma
    #RANDOM_STATE = config_data.svc_params.random_state

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
                    train_set_output_dir=TRAIN_SET_DIR,
                    test_set_output_dir=TEST_SET_DIR,
                    resampling_voxel_size=VOXEL_SIZE,
                    normalize=IS_NORMALIZE_DEFAULT,
                    norm_method=NORMALIZATION_METHOD,
                    min_max_min_val=MIN_VAL,
                    min_max_max_val=MAX_VAL)
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
    preprocess_images(normalize=IS_NORMALIZE_WHEN_PREPROCESS,
                      norm_metod=NORMALIZATION_METHOD,
                      min_max_min_val=MIN_VAL,
                      min_max_max_val=MAX_VAL,
                      brain_extraction=IS_BRAIN_EXTRACTION,
                      brain_extraction_modality=MODALITY,
                      brain_extraction_verbose=VERBOSE,
                      crop=IS_CROP,
                      smooth=IS_SMOOTH,
                      smooth_sigma=SIGMA,
                      smooth_order=ORDER,
                      smooth_mode=MODE,
                      smooth_cval=CVAL,
                      smooth_truncate=TRUNCATE,
                      re_normalize_after_smooth=IS_RE_NORMALIZE_AFTER_SMOOTH,
                      preprocess_test_set=IS_PREPROCESS_TEST_SET,
                      output_dir=PREPROCESSED_DATA_DIR,
                      )
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
                "shift": TRANSLATION_SHIFT # 5
            }),
            ("rotation", {
                "angle": ROTATION_ANGLE # 10
            }),
            ("gaussian_noise", {
                "mean": GAUSSIAN_NOISE_MEAN,
                "std": GAUSSIAN_NOISE_STD
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
    feature_extraction_pipeline(train_set_dir=TRAIN_SET_DIR,
                                test_set_dir=TEST_SET_DIR,
                                labels_dir=LABELS_DIR,
                                extracted_features_dir=EXTRACTED_FEATURES_DIR,
                                batch_size=BATCH_SIZE,
                                feature_extraction_model=BASE_MODEL_NAME,
                                base_model_weights=BASE_MODEL_WEIGHTS,
                                input_channels=INPUT_CHANNELS)
    logger.info("Feature extraction completed.")

    # Step 5: Model training and validation (SVC)
    """
    Trains a machine learning model using the extracted features located in /data/train_set/features,
    Validates the trained model using the test set located in /data/test_set/preprocessed,
    Saves the trained model in /models.
    Saves the validation results in /results.
    """
    train_and_evaluate(extracted_features_dir=EXTRACTED_FEATURES_DIR,
                       dir_to_save_clf=SAVE_CLF_DIR,
                       results_output_dir=RESULTS_OUTPUT_DIR,
                       clf_kernel=KERNEL,
                       clf_c_value=C,
                       clf_gamma=GAMMA
                       )
    logger.info("Classifier training and validation completed.")

    # Step 6: Plot model (SVC) metrics
    """
    Plots model metrics (accuracy, precision, recall, f1-score) on the test set.
    """

    logger.info("The program completed successfully.")


if __name__ == "__main__":
    main()
