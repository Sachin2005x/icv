import unittest

import cv2
import numpy as np

from scripts.check_image import evaluate_images


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((120, 160, 3), 240, dtype=np.uint8)
        cv2.rectangle(self.reference, (20, 25), (140, 95), (40, 80, 180), -1)
        cv2.putText(self.reference, "TEST", (38, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def test_identical_image_has_genuine_verdict(self):
        result = evaluate_images(self.reference.copy(), self.reference)
        self.assertEqual(result.verdict, "GENUINE")
        self.assertGreaterEqual(result.overall_score, 99)

    def test_blurred_image_scores_lower_than_identical_image(self):
        identical = evaluate_images(self.reference.copy(), self.reference)
        blurred = cv2.GaussianBlur(self.reference, (17, 17), 0)
        result = evaluate_images(blurred, self.reference)
        self.assertLess(result.overall_score, identical.overall_score)
        self.assertLess(result.components["print_quality"]["score"], 60)
