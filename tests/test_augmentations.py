import unittest
import numpy as np
from utils.augmentations import apply_translation, apply_rotation, apply_gaussian_noise

class TestAugmentations(unittest.TestCase):

    def setUp(self):
        """Create a sample 3D MRI image (5x5x5) for testing."""
        self.image = np.zeros((5, 5, 5), dtype=np.float32)
        self.image[2, 2, 2] = 1  # Add a distinct feature to track transformations

    def test_apply_translation(self):
        """Test translation shifts the image correctly."""
        translated_image = apply_translation(self.image, (1, -1))

        # Ensure the pixel at (2,2,2) moves to (3,1,2)
        self.assertEqual(translated_image[3, 1, 2], 1)
        self.assertEqual(translated_image[2, 2, 2], 0)  # Original should be empty

    def test_apply_rotation(self):
        """Test rotation does not alter the image size."""
        rotated_image = apply_rotation(self.image, 90)
        
        # Shape should remain unchanged
        self.assertEqual(rotated_image.shape, self.image.shape)

        # The feature pixel should have moved (for a 90-degree rotation)
        self.assertNotEqual(rotated_image[2, 2, 2], 1)

    def test_apply_gaussian_noise(self):
        """Test Gaussian noise is added correctly."""
        noisy_image = apply_gaussian_noise(self.image, mean=0, std=0.1)

        # Ensure the shape is preserved
        self.assertEqual(noisy_image.shape, self.image.shape)

        # The pixel at (2,2,2) should be modified by noise
        self.assertNotEqual(noisy_image[2, 2, 2], self.image[2, 2, 2])

        # Check if noise follows expected properties
        self.assertAlmostEqual(noisy_image.std(), self.image.std() * 0.1, delta=0.05)

if __name__ == '__main__':
    unittest.main()
