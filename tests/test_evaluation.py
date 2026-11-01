from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluation import sequence_metrics


class EvaluationTests(unittest.TestCase):
    def test_separate_aspect_metrics_ignore_opinion_errors(self):
        true = [["B-ASP_POS", "O", "B-OPINION"]]
        predicted = [["B-ASP_POS", "O", "O"]]
        self.assertEqual(sequence_metrics(true, predicted, target="aspect")["f1"], 1.0)
        self.assertEqual(sequence_metrics(true, predicted, target="opinion")["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()

