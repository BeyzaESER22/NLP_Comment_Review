"""Evaluation helpers for separate aspect and opinion reporting."""

from __future__ import annotations

def _project_tags(sequences, target: str):
    if target not in {"combined", "aspect", "opinion"}:
        raise ValueError("target must be 'combined', 'aspect', or 'opinion'.")
    if target == "combined":
        return sequences

    projected = []
    for sequence in sequences:
        projected_sequence = []
        for tag in sequence:
            keep = (target == "aspect" and "-ASP_" in tag) or (
                target == "opinion" and tag.endswith("-OPINION")
            )
            projected_sequence.append(tag if keep else "O")
        projected.append(projected_sequence)
    return projected


def sequence_metrics(y_true, y_pred, target: str = "combined") -> dict[str, float]:
    true_projected = _project_tags(y_true, target)
    pred_projected = _project_tags(y_pred, target)
    true_entities = set()
    pred_entities = set()
    for sequence_index, sequence in enumerate(true_projected):
        true_entities.update(
            (sequence_index, entity_type, start, end)
            for entity_type, start, end in _entities(sequence)
        )
    for sequence_index, sequence in enumerate(pred_projected):
        pred_entities.update(
            (sequence_index, entity_type, start, end)
            for entity_type, start, end in _entities(sequence)
        )
    true_positive = len(true_entities & pred_entities)
    precision = true_positive / len(pred_entities) if pred_entities else 0.0
    recall = true_positive / len(true_entities) if true_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _entities(sequence):
    """Extract exact BIO entities as (type, start, inclusive_end)."""
    entities = []
    current_type = None
    start = None
    for index, tag in enumerate(list(sequence) + ["O"]):
        if tag.startswith("B-"):
            if current_type is not None:
                entities.append((current_type, start, index - 1))
            current_type = tag[2:]
            start = index
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue
        else:
            if current_type is not None:
                entities.append((current_type, start, index - 1))
                current_type, start = None, None
            if tag.startswith("I-"):
                current_type = tag[2:]
                start = index
    return entities



