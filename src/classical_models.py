"""HMM and feature-engineered CRF sequence models."""

from __future__ import annotations

from collections import Counter

import numpy as np


class HMMSequenceTagger:
    def __init__(self, unk_threshold: int = 1, smoothing: float = 0.01, use_pos: bool = True):
        self.unk_threshold = unk_threshold
        self.smoothing = smoothing
        self.use_pos = use_pos

    def _obs(self, word, pos):
        lowered = word.lower()
        if lowered in self.vocab:
            return lowered
        return f"<UNK>_{pos}" if self.use_pos else "<UNK>"

    def fit(self, data, target_key: str = "bio_tags"):
        counts = Counter(word.lower() for item in data for word in item["words"])
        self.vocab = {word for word, count in counts.items() if count > self.unk_threshold}
        sequences = [
            (
                [self._obs(word, pos) for word, pos in zip(item["words"], item["pos_tags"])],
                item[target_key],
            )
            for item in data
            if item[target_key]
        ]
        if not sequences:
            raise ValueError("HMM training data contains no non-empty sequences.")

        self.tags = sorted({tag for _, tags in sequences for tag in tags})
        self.symbols = sorted({observation for observations, _ in sequences for observation in observations})
        self.tag2i = {tag: index for index, tag in enumerate(self.tags)}
        self.sym2i = {symbol: index for index, symbol in enumerate(self.symbols)}
        tag_count, symbol_count = len(self.tags), len(self.symbols)

        start = np.zeros(tag_count)
        transition = np.zeros((tag_count, tag_count))
        emission = np.zeros((tag_count, symbol_count))
        for observations, tags in sequences:
            start[self.tag2i[tags[0]]] += 1
            for index, (observation, tag) in enumerate(zip(observations, tags)):
                emission[self.tag2i[tag], self.sym2i[observation]] += 1
                if index:
                    transition[self.tag2i[tags[index - 1]], self.tag2i[tag]] += 1

        smoothing = self.smoothing
        self.log_start = np.log((start + smoothing) / (start.sum() + smoothing * tag_count))
        self.log_transition = np.log(
            (transition + smoothing)
            / (transition.sum(1, keepdims=True) + smoothing * tag_count)
        )
        self.log_emission = np.log(
            (emission + smoothing)
            / (emission.sum(1, keepdims=True) + smoothing * (symbol_count + 1))
        )
        self.log_unseen = np.log(
            smoothing / (emission.sum(1) + smoothing * (symbol_count + 1))
        )
        return self

    def decode(self, words, pos_tags):
        columns = []
        for word, pos in zip(words, pos_tags):
            symbol_index = self.sym2i.get(self._obs(word, pos))
            columns.append(
                self.log_emission[:, symbol_index]
                if symbol_index is not None
                else self.log_unseen
            )
        emission_matrix = np.stack(columns, axis=1)
        tag_count, sequence_length = emission_matrix.shape
        scores = np.zeros((tag_count, sequence_length))
        backpointers = np.zeros((tag_count, sequence_length), dtype=int)
        scores[:, 0] = self.log_start + emission_matrix[:, 0]
        for timestep in range(1, sequence_length):
            candidates = scores[:, timestep - 1][:, None] + self.log_transition
            backpointers[:, timestep] = candidates.argmax(axis=0)
            scores[:, timestep] = candidates.max(axis=0) + emission_matrix[:, timestep]

        best = int(scores[:, -1].argmax())
        path = [best]
        for timestep in range(sequence_length - 1, 0, -1):
            best = int(backpointers[best, timestep])
            path.append(best)
        path.reverse()
        return [self.tags[index] for index in path]

    def predict(self, data):
        return [self.decode(item["words"], item["pos_tags"]) for item in data]


def token_features(words, pos_tags, index):
    word, pos = words[index], pos_tags[index]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "pos": pos,
        "pos[:2]": pos[:2],
    }
    for offset in (-2, -1, 1, 2):
        neighbour = index + offset
        if 0 <= neighbour < len(words):
            features[f"{offset}:word.lower"] = words[neighbour].lower()
            features[f"{offset}:pos"] = pos_tags[neighbour]
        else:
            features[f"{offset}:{'BOS' if offset < 0 else 'EOS'}"] = True
    return features


def sentence_features(item):
    return [
        token_features(item["words"], item["pos_tags"], index)
        for index in range(len(item["words"]))
    ]


def train_crf(train_data, c1=0.1, c2=0.1, max_iterations=100):
    try:
        import sklearn_crfsuite
    except ImportError as exc:
        raise ImportError("CRF training requires sklearn-crfsuite.") from exc
    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    model.fit(
        [sentence_features(item) for item in train_data],
        [item["bio_tags"] for item in train_data],
    )
    return model


def predict_crf(model, data):
    return model.predict([sentence_features(item) for item in data])

