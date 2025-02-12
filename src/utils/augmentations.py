import numpy as np
from scipy.ndimage import rotate, shift

def apply_translation(image, translation):
    return shift(image, shift=(translation[0], translation[1], 0), mode='nearest')

def apply_rotation(image, angle):
    return rotate(image, angle, axes=(0, 1), reshape=False, mode='nearest')

def apply_gaussian_noise(image, mean, std):
    std = image.std() * std
    noise = np.random.normal(mean, std, image.shape).astype(image.dtype)
    return image + noise
