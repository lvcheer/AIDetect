import unittest

from aidetect.fusion import (
    BURSTINESS_ONLY_WEIGHTS,
    FULL_FEATURE_WEIGHTS,
    PERPLEXITY_ONLY_WEIGHTS,
    fuse_heuristic_scores,
)


class FusionTests(unittest.TestCase):
    def test_declared_weights_sum_to_one(self):
        for weights in (
            FULL_FEATURE_WEIGHTS,
            PERPLEXITY_ONLY_WEIGHTS,
            BURSTINESS_ONLY_WEIGHTS,
        ):
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_returns_classifier_score_without_auxiliary_features(self):
        self.assertEqual(fuse_heuristic_scores(80.0), 80.0)

    def test_preserves_full_feature_weights(self):
        self.assertEqual(
            fuse_heuristic_scores(80.0, perplexity_score=20.0, burstiness_score=60.0),
            40.0,
        )

    def test_preserves_perplexity_only_weights(self):
        self.assertEqual(
            fuse_heuristic_scores(80.0, perplexity_score=20.0),
            35.0,
        )

    def test_preserves_burstiness_only_weights(self):
        self.assertEqual(
            fuse_heuristic_scores(80.0, burstiness_score=60.0),
            74.0,
        )

    def test_zero_is_treated_as_an_available_feature_score(self):
        self.assertEqual(
            fuse_heuristic_scores(100.0, perplexity_score=0.0, burstiness_score=0.0),
            20.0,
        )

    def test_rounds_like_the_existing_gui(self):
        self.assertEqual(
            fuse_heuristic_scores(33.333, perplexity_score=66.666),
            58.33,
        )


if __name__ == "__main__":
    unittest.main()
