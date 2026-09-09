# SPDX-License-Identifier: MIT
"""Pure scoring/pairing logic for the IntPhys 2 violation-of-expectation benchmark
(https://huggingface.co/datasets/facebook/IntPhys2). Each scene contributes two
independent Possible/Impossible pairs (case "1" and case "2"), and a model is scored
per pair by whether its "surprise" (prediction error) is higher for the impossible
video -- the standard methodology this benchmark family uses (see
facebookresearch/jepa-intuitive-physics's evals/intphys_test/eval.py, whose
compute_metrics does the same possible-vs-impossible comparison this module does).

No video decoding or model calls here -- see eval_intphys2.py for the runner that
produces the actual surprise scores this module consumes.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRecord:
    scene_index: str
    name: str
    file_name: str
    condition: str
    difficulty: str
    case: str  # groups the Possible/Impossible pair within a scene, e.g. "1" or "2"
    possible: bool


def load_metadata(csv_path: str) -> list[VideoRecord]:
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            case, label = row["type"].split("_", 1)
            records.append(
                VideoRecord(
                    scene_index=row["SceneIndex"],
                    name=row["name"],
                    file_name=row["file_name"],
                    condition=row["condition"],
                    difficulty=row["Difficulty"],
                    case=case,
                    possible=(label == "Possible"),
                )
            )
    return records


def pair_videos(records: list[VideoRecord]) -> list[tuple[VideoRecord, VideoRecord]]:
    """Groups by (scene_index, case); each group must have exactly one Possible and
    one Impossible video. Returns (possible, impossible) tuples."""
    groups: dict[tuple[str, str], list[VideoRecord]] = defaultdict(list)
    for r in records:
        groups[(r.scene_index, r.case)].append(r)

    pairs = []
    for key, group in groups.items():
        possible = [r for r in group if r.possible]
        impossible = [r for r in group if not r.possible]
        if len(possible) != 1 or len(impossible) != 1:
            raise ValueError(f"Expected exactly one Possible and one Impossible video for {key}, got {group}")
        pairs.append((possible[0], impossible[0]))
    return pairs


def pairwise_accuracy(pairs_with_scores: list[tuple[float, float]]) -> float:
    """pairs_with_scores: (possible_surprise, impossible_surprise) tuples. A pair is
    correct when the impossible video's surprise is strictly higher."""
    if not pairs_with_scores:
        return float("nan")
    correct = sum(1 for possible, impossible in pairs_with_scores if impossible > possible)
    return correct / len(pairs_with_scores)


def accuracy_by_group(pairs, scores: dict[str, float], group_key) -> dict[str, float]:
    """group_key: callable(possible_record) -> group label (e.g. lambda r: r.condition).
    scores: video name -> surprise score. Returns {group_label: pairwise accuracy}."""
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for possible, impossible in pairs:
        buckets[group_key(possible)].append((scores[possible.name], scores[impossible.name]))
    return {key: pairwise_accuracy(rows) for key, rows in buckets.items()}
