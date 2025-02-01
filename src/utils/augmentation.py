import numpy as np
from scipy.ndimage import rotate, shift


class ImageAugmentor:

    def __init__(self,
                 apply_translation=True,
                 apply_rotation=True,
                 apply_gaussian_noise=True):
        self.apply_translation = apply_translation
        self.apply_rotation = apply_rotation
        self.apply_gaussian_noise = apply_gaussian_noise

    def augment(self, image, translation=(0, 0), angle=0, mean=0, std=0.1):
        """Applies selected augmentations to the image."""
        if self.apply_translation:
            image = self._apply_translation(image, translation)
        if self.apply_rotation:
            image = self._apply_rotation(image, angle)
        if self.apply_gaussian_noise:
            image = self._apply_gaussian_noise(image, mean, std)
        return image

    def _apply_translation(self, image, translation):
        return shift(image,
                     shift=(translation[0], translation[1], 0),
                     mode='nearest')

    def _apply_rotation(self, image, angle):
        return rotate(image, angle, axes=(0, 1), reshape=False, mode='nearest')

    def _apply_gaussian_noise(self, image, mean, std):
        std = image.std() * std
        noise = np.random.normal(mean, std, image.shape).astype(image.dtype)
        return image + noise
