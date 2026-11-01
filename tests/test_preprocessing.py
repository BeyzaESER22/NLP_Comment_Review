from pathlib import Path
import sys
import unittest

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_validation import validate_dataframe
from preprocessing import align_aspect_annotations, build_bio_dataset


def fake_pos_tagger(words):
    return [(word, "NN") for word in words]


def make_frame(text, aspects, polarities):
    return pd.DataFrame(
        [{"text": text, "aspect_terms": aspects, "term_polarities": polarities}]
    )


class PreprocessingTests(unittest.TestCase):
    def test_blank_polarity_is_not_silently_neutral(self):
        data, issues = build_bio_dataset(
            make_frame("The battery is fine.", "battery", ""),
            extract_opinions=False,
            return_issues=True,
            pos_tagger=fake_pos_tagger,
        )
        self.assertEqual(data[0]["bio_tags"], ["O", "O", "O", "O"])
        self.assertIn("blank_polarity", set(issues["issue"]))

    def test_unsupported_polarity_is_skipped_and_reported(self):
        aligned, issues = align_aspect_annotations("Good screen", "screen", "mixed")
        self.assertEqual(aligned, [])
        self.assertEqual(issues, [{"issue": "unsupported_polarity", "value": "mixed"}])

    def test_multi_token_aspect_with_attached_punctuation(self):
        data = build_bio_dataset(
            make_frame("The battery life, is excellent.", "battery life", "positive"),
            extract_opinions=False,
            pos_tagger=fake_pos_tagger,
        )
        self.assertEqual(data[0]["bio_tags"], ["O", "B-ASP_POS", "I-ASP_POS", "O", "O"])

    def test_repeated_annotations_consume_successive_occurrences(self):
        data, issues = build_bio_dataset(
            make_frame(
                "Screen is bright but screen is expensive.",
                "screen | screen",
                "positive | negative",
            ),
            extract_opinions=False,
            return_issues=True,
            pos_tagger=fake_pos_tagger,
        )
        self.assertEqual(data[0]["bio_tags"][0], "B-ASP_POS")
        self.assertEqual(data[0]["bio_tags"][4], "B-ASP_NEG")
        self.assertIn("repeated_aspect_ambiguous", set(issues["issue"]))

    def test_overlapping_aspects_do_not_overwrite_existing_tags(self):
        data, issues = build_bio_dataset(
            make_frame("Battery life is good.", "battery life | life", "positive | positive"),
            extract_opinions=False,
            return_issues=True,
            pos_tagger=fake_pos_tagger,
        )
        self.assertEqual(data[0]["bio_tags"][:2], ["B-ASP_POS", "I-ASP_POS"])
        self.assertIn("overlapping_or_exhausted_aspect", set(issues["issue"]))

    def test_validator_reports_blank_polarity_and_repeated_term(self):
        frame = pd.DataFrame(
            [
                {"text": "screen and screen", "aspect_terms": "screen", "term_polarities": "positive"},
                {"text": "battery", "aspect_terms": "battery", "term_polarities": ""},
            ]
        )
        report, issues = validate_dataframe(frame)
        self.assertEqual(report["ambiguous_repeated_aspect_items"], 1)
        self.assertEqual(report["blank_polarity_items"], 1)
        self.assertEqual(set(issues["issue"]), {"repeated_aspect_ambiguous", "blank_polarity"})


if __name__ == "__main__":
    unittest.main()

