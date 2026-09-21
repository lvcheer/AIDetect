import json
import unittest

from aidetect.schema import DetectionRecord, RESULT_SCHEMA_VERSION, SCORE_SCALE


class DetectionRecordTests(unittest.TestCase):
    def test_stable_record_distinguishes_score_sources(self):
        record = DetectionRecord(
            text="example",
            raw_classifier_score=80.0,
            raw_classifier_is_ai=True,
            perplexity_value=40.0,
            perplexity_heuristic_score=50.0,
            burstiness_cv=0.3,
            burstiness_heuristic_score=66.08,
            fused_score=58.22,
            explanation="example explanation",
        )

        payload = record.to_dict()
        self.assertEqual(payload["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["score_scale"], SCORE_SCALE)
        self.assertEqual(payload["raw_classifier_score"], 80.0)
        self.assertEqual(payload["perplexity_heuristic_score"], 50.0)
        self.assertEqual(payload["burstiness_heuristic_score"], 66.08)
        self.assertEqual(payload["fused_score"], 58.22)
        json.dumps(payload)

    def test_legacy_adapter_preserves_gui_and_csv_fields(self):
        record = DetectionRecord(
            text="example",
            raw_classifier_score=80.0,
            raw_classifier_is_ai=True,
            perplexity_value=40.0,
            perplexity_heuristic_score=50.0,
            burstiness_cv=0.3,
            burstiness_heuristic_score=66.08,
            fused_score=58.22,
            explanation="example explanation",
            error="example error",
        )

        self.assertEqual(
            record.to_legacy_dict(),
            {
                "sentence": "example",
                "ai_prob": 58.22,
                "human_prob": 41.78,
                "is_ai": True,
                "perplexity": 40.0,
                "burstiness_cv": 0.3,
                "explanation": "example explanation",
                "error": "example error",
            },
        )

    def test_legacy_adapter_omits_unavailable_optional_fields(self):
        record = DetectionRecord(
            text="example",
            raw_classifier_score=20.0,
            raw_classifier_is_ai=False,
            perplexity_value=None,
            perplexity_heuristic_score=None,
            burstiness_cv=None,
            burstiness_heuristic_score=None,
            fused_score=20.0,
        )

        self.assertEqual(
            record.to_legacy_dict(),
            {
                "sentence": "example",
                "ai_prob": 20.0,
                "human_prob": 80.0,
                "is_ai": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
