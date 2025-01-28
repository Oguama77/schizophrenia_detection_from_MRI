from utils.preprocess import load_nii, get_data, resample_image, normalize_data
from utils.preprocess_validation import plot_histogram



if __name__ == '__main__':

    #data_path = 'data/sub-A00000541_ses-20110101_acq-mprage_run-01_T1w.nii.gz'
    #data_path = 'data/sub-A00002480_ses-20110101_acq-mprage_run-02_echo-02_T1w.nii.gz'
    #data_path = 'data/sub-A00014636_ses-20090101_acq-mprage_run-01_echo-04_T1w.nii.gz'
    #data_path = 'data/sub-A00020414_ses-20100101_acq-mprage_run-02_echo-02_T1w.nii.gz'
    #data_path = 'data/sub-A00024546_ses-20090101_acq-mprage_run-02_T1w.nii.gz'
    data_path = 'data/sub-A00035751_ses-20130101_acq-mprage_run-01_echo-02_T1w.nii.gz'
    nii = load_nii(data_path)
    data = get_data(nii)
    plot_histogram(data, 50, 'original')

    resampled_data = resample_image(nii, voxel_size=(1, 1, 1), order=0, mode="wrap", cval=0)
    normalized_data = normalize_data(resampled_data, method='z-score') # , method='z-score'
    plot_histogram(normalized_data, 50, 'min max 0 1')

