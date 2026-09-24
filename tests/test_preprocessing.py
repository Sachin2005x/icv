import unittest

import numpy as np

from src.preprocessing import align_to_reference


class PreprocessingTests(unittest.TestCase):
    def test_alignment_resizes_images_without_features(self):
        query = np.zeros((40, 60, 3), dtype=np.uint8)
        reference = np.zeros((80, 100, 3), dtype=np.uint8)
        aligned, transform = align_to_reference(query, reference)
        self.assertEqual(aligned.shape, reference.shape)
        self.assertEqual(transform.shape, (3, 3))
