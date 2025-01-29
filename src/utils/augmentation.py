import numpy as np
from scipy.ndimage import rotate, shift

def apply_translation(image, translation=(0, 0)):
    """Apply translation to the image along x and y axes."""
    return shift(image, shift=(translation[0], translation[1], 0), mode='nearest')

def apply_rotation(image, angle):
    """Apply random rotation to the image."""
    return rotate(image, angle,  axes=(0, 1),  reshape=False, mode='nearest')

def apply_gaussian_noise(image, mean=0, std=0.1):
    """
    Adds Gaussian noise to a 3D image.
    Args:
        image (np.ndarray): The 3D image data.
        mean (float): Mean of the Gaussian noise.
        std_dev (float): Standard deviation of the Gaussian noise.
    Returns:
        np.ndarray: Image with added Gaussian noise.
    """
    std = image.std() * std
    noise = np.random.normal(mean, std, image.shape).astype(image.dtype)
    return image + noise