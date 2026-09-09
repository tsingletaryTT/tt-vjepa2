# SPDX-License-Identifier: MIT
"""Tests for the pure scoring/pairing logic behind the IntPhys 2 evaluation: parsing
the official metadata.csv, pairing each scene's matched Possible/Impossible videos,
and computing pairwise accuracy. No video decoding, no model, no hardware -- these are
tested separately in eval_intphys2.py, which isn't unit-tested (same convention as
tt/benchmark.py: hardware-touching scripts are run and read, not mocked)."""

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intphys2_scoring import (  # noqa: E402
    VideoRecord,
    accuracy_by_group,
    load_metadata,
    pair_videos,
    pairwise_accuracy,
)

_HEADER = "SceneIndex,name,file_name,game_name,condition,env,type,occluder,Difficulty,Camera\n"


def _write_csv(rows: list[str]) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    f.write(_HEADER)
    f.writelines(rows)
    f.close()
    return f.name


def test_load_metadata_parses_case_and_possible_from_type_column():
    path = _write_csv([
        "0,vidA,Videos/vidA.mp4,Game,solidity,Env_0,1_Possible,None,Easy,Fixed\n",
        "0,vidB,Videos/vidB.mp4,Game,solidity,Env_0,1_Impossible,None,Easy,Fixed\n",
    ])
    records = load_metadata(path)
    assert records[0] == VideoRecord(
        scene_index="0", name="vidA", file_name="Videos/vidA.mp4",
        condition="solidity", difficulty="Easy", case="1", possible=True,
    )
    assert records[1].case == "1"
    assert records[1].possible is False


def test_pair_videos_groups_by_scene_and_case():
    path = _write_csv([
        "0,a1,Videos/a1.mp4,G,solidity,E,1_Possible,None,Easy,Fixed\n",
        "0,a2,Videos/a2.mp4,G,solidity,E,1_Impossible,None,Easy,Fixed\n",
        "0,a3,Videos/a3.mp4,G,solidity,E,2_Possible,None,Easy,Fixed\n",
        "0,a4,Videos/a4.mp4,G,solidity,E,2_Impossible,None,Easy,Fixed\n",
    ])
    records = load_metadata(path)
    pairs = pair_videos(records)
    assert len(pairs) == 2
    names = sorted((p.name, i.name) for p, i in pairs)
    assert names == [("a1", "a2"), ("a3", "a4")]


def test_pair_videos_raises_on_malformed_group():
    path = _write_csv([
        "0,a1,Videos/a1.mp4,G,solidity,E,1_Possible,None,Easy,Fixed\n",
        "0,a2,Videos/a2.mp4,G,solidity,E,1_Possible,None,Easy,Fixed\n",  # two Possible, no Impossible
    ])
    records = load_metadata(path)
    with pytest.raises(ValueError):
        pair_videos(records)


def test_pairwise_accuracy_counts_correctly_ranked_pairs():
    # correct: impossible surprise > possible surprise
    scores = [(0.1, 0.5), (0.3, 0.2), (0.4, 0.9)]  # 2 of 3 correct
    assert pairwise_accuracy(scores) == pytest.approx(2 / 3)


def test_pairwise_accuracy_empty_is_nan():
    import math

    assert math.isnan(pairwise_accuracy([]))


def test_accuracy_by_group_buckets_by_the_given_key():
    path = _write_csv([
        "0,a1,Videos/a1.mp4,G,solidity,E,1_Possible,None,Easy,Fixed\n",
        "0,a2,Videos/a2.mp4,G,solidity,E,1_Impossible,None,Easy,Fixed\n",
        "1,b1,Videos/b1.mp4,G,permanence,E,1_Possible,None,Hard,Fixed\n",
        "1,b2,Videos/b2.mp4,G,permanence,E,1_Impossible,None,Hard,Fixed\n",
    ])
    records = load_metadata(path)
    pairs = pair_videos(records)
    scores = {"a1": 0.1, "a2": 0.9, "b1": 0.5, "b2": 0.3}  # solidity correct, permanence wrong

    result = accuracy_by_group(pairs, scores, group_key=lambda r: r.condition)

    assert result == {"solidity": 1.0, "permanence": 0.0}
