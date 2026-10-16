#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the tooling-only tt-transformers wheel the bundle ships in wheels/.

Why this exists. tt-transformers is not on PyPI yet, and V-JEPA2 needs two small pieces of
its tooling (`tt_transformers.modules.lightweightmodule` and `tt_transformers.tensor_utils`).
tt-transformers also carries the code of every model under `tt_transformers/models/`, and
this bundle must not ship any of it. This script produces the wheel the canonical v6 shape
asks for: the real tt-transformers tooling, with `models/` emptied.

What it does, in order:
  1. `git archive <ref>` of a tt-transformers checkout, so uncommitted edits cannot leak in.
  2. `uv build --wheel` on that clean tree.
  3. Unpack the wheel, then
       - delete everything under `tt_transformers/models/` except `__init__.py`;
       - delete every `Requires-Dist` and `Provides-Extra` line from METADATA. The upstream
         wheel pins ttnn, torch and transformers to exact versions of its own. The bundle's
         `requirements.txt` and `constraints.txt` own the engine pins, and the wheel is
         installed by path after them, so a second set of exact pins in the wheel would
         conflict with `ttnn==0.78.0` and fail the install;
       - delete the stale RECORD.
  4. `wheel pack` (run through `uvx`, so nothing needs installing first) writes a fresh RECORD and the final wheel.

Usage:
    make_tooling_wheel.py <tt-transformers-checkout> <git-ref> <out-dir>

Prints the path of the wheel it wrote. Needs `uv` (which fetches `wheel` on demand) and git.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def build(checkout: Path, ref: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tt-transformers-tooling-") as tmp:
        tmp = Path(tmp)
        src, dist, unpacked = tmp / "src", tmp / "dist", tmp / "unpacked"
        src.mkdir()

        # 1. A clean export of the pinned commit.
        archive = subprocess.run(
            ["git", "-C", str(checkout), "archive", ref], check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)

        # 2. Build the upstream wheel from that export.
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist), str(src)],
            check=True,
            capture_output=True,
        )
        (upstream,) = dist.glob("tt_transformers-*.whl")

        # 3. Unpack and cut it down to tooling.
        with zipfile.ZipFile(upstream) as zf:
            zf.extractall(unpacked)

        models = unpacked / "tt_transformers" / "models"
        for entry in models.iterdir():
            if entry.name == "__init__.py":
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()

        (dist_info,) = unpacked.glob("tt_transformers-*.dist-info")
        metadata = dist_info / "METADATA"
        kept = [
            line
            for line in metadata.read_text().splitlines(keepends=True)
            if not line.startswith(("Requires-Dist:", "Provides-Extra:"))
        ]
        metadata.write_text("".join(kept))
        (dist_info / "RECORD").unlink()

        # 4. Repack. `wheel pack` regenerates RECORD and names the file from the dist-info.
        packed = tmp / "packed"
        packed.mkdir()  # `wheel pack` does not create its destination
        subprocess.run(
            ["uvx", "--from", "wheel", "wheel", "pack", str(unpacked), "--dest-dir", str(packed)],
            check=True,
            capture_output=True,
        )
        (wheel,) = packed.glob("tt_transformers-*.whl")
        final = out_dir / wheel.name
        shutil.copy2(wheel, final)
        return final


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    print(build(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])))
