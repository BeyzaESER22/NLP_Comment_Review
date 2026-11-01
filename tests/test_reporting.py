from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from reporting import (
    categorize_record_errors,
    hard_pattern_rows,
    polarity_confusion_rows,
)


class ReportingTests(unittest.TestCase):
    def test_exact_boundary_wrong_polarity_is_polarity_error(self):
        record = {
            "id": 1,
            "words": ["screen"],
            "gold_tags": ["B-ASP_POS"],
            "pred_tags": ["B-ASP_NEG"],
        }
        errors = categorize_record_errors(record, "model")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_type"], "polarity")

    def test_overlapping_different_span_is_boundary_error(self):
        record = {
            "id": 1,
            "words": ["battery", "life"],
            "gold_tags": ["B-ASP_POS", "I-ASP_POS"],
            "pred_tags": ["B-ASP_POS", "O"],
        }
        errors = categorize_record_errors(record, "model")
        self.assertEqual([row["error_type"] for row in errors], ["boundary"])

    def test_polarity_confusion_uses_only_exact_boundaries(self):
        records = [
            {
                "id": 1,
                "words": ["screen"],
                "gold_tags": ["B-ASP_POS"],
                "pred_tags": ["B-ASP_NEG"],
            },
            {
                "id": 2,
                "words": ["battery", "life"],
                "gold_tags": ["B-ASP_POS", "I-ASP_POS"],
                "pred_tags": ["B-ASP_POS", "O"],
            },
        ]
        rows = polarity_confusion_rows(records, "model")
        self.assertEqual(rows, [{"model": "model", "gold": "ASP_POS", "predicted": "ASP_NEG", "count": 1}])

    def test_hard_pattern_counts_negation(self):
        records = [
            {
                "id": 1,
                "words": ["not", "good"],
                "gold_tags": ["B-OPINION", "I-OPINION"],
                "pred_tags": ["O", "B-OPINION"],
            },
            {
                "id": 2,
                "words": ["great"],
                "gold_tags": ["B-OPINION"],
                "pred_tags": ["B-OPINION"],
            },
        ]
        rows = hard_pattern_rows(records, "model")
        negation_pattern = next(
            row for row in rows if row["pattern"] == "negation" and row["group"] == "pattern"
        )
        self.assertEqual(negation_pattern["sentence_count"], 1)
        self.assertEqual(negation_pattern["error_rate"], 1.0)

if __name__ == "__main__":
    unittest.main()

