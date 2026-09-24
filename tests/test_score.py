import unittest

from src.model import verdict_for
from src.score import DEFAULT_WEIGHTS, fuse_components


class ScoreTests(unittest.TestCase):
    def test_fusion_uses_available_component_weights(self):
        components = {"color": {"score": 80}, "ssim": {"score": 60}}
        w = DEFAULT_WEIGHTS
        expected = (w["color"] * 80 + w["ssim"] * 60) / (w["color"] + w["ssim"])
        score, per = fuse_components(components)
        self.assertEqual(score, expected)
        self.assertEqual(per, {"color": 80, "ssim": 60})

    def test_verdict_boundaries(self):
        self.assertEqual(verdict_for(85), "GENUINE")
        self.assertEqual(verdict_for(70), "LIKELY GENUINE")
        self.assertEqual(verdict_for(55), "UNCERTAIN")
        self.assertEqual(verdict_for(35), "LIKELY COUNTERFEIT")
        self.assertEqual(verdict_for(34.9), "COUNTERFEIT")
