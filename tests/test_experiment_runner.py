from pathlib import Path
import json
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from classical_models import HMMSequenceTagger
from experiment_io import read_predictions, write_predictions
from run_experiment import load_config, split_dataset


def synthetic_dataset(size=20):
    return [
        {
            "id": index,
            "words": ["good" if index % 2 == 0 else "bad", "screen"],
            "pos_tags": ["JJ", "NN"],
            "bio_tags": ["B-OPINION", "B-ASP_POS" if index % 2 == 0 else "B-ASP_NEG"],
        }
        for index in range(size)
    ]


class ExperimentRunnerTests(unittest.TestCase):
    def test_config_overrides_only_non_null_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"seed": 42, "sample_size": 5000}), encoding="utf-8")
            result = load_config(path, {"seed": 9, "sample_size": None})
            self.assertEqual(result, {"seed": 9, "sample_size": 5000})

    def test_split_is_deterministic_and_disjoint(self):
        data = synthetic_dataset(30)
        first = split_dataset(data, 0.2, 0.1, 42)
        second = split_dataset(data, 0.2, 0.1, 42)
        first_ids = [[item["id"] for item in partition] for partition in first]
        second_ids = [[item["id"] for item in partition] for partition in second]
        self.assertEqual(first_ids, second_ids)
        flattened = [item for partition in first_ids for item in partition]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(range(30)))

    def test_hmm_smoke_predictions_match_sequence_lengths(self):
        data = synthetic_dataset(20)
        model = HMMSequenceTagger(unk_threshold=0, smoothing=0.1).fit(data[:15])
        predictions = model.predict(data[15:])
        self.assertEqual(len(predictions), 5)
        self.assertTrue(
            all(len(prediction) == len(item["words"]) for prediction, item in zip(predictions, data[15:]))
        )

    def test_prediction_round_trip(self):
        data = synthetic_dataset(3)
        predictions = [item["bio_tags"] for item in data]
        with tempfile.TemporaryDirectory() as directory:
            path = write_predictions("hmm", data, predictions, directory)
            records = read_predictions(path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["gold_tags"], records[0]["pred_tags"])


if __name__ == "__main__":
    unittest.main()

