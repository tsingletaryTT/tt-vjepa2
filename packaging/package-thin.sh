#!/usr/bin/env bash
# Stage the tt-vjepa2 v6 thin bundle in the canonical shape. This is the ONE recipe a
# repackage uses; everything it reads is either in this directory or named below.
#
#   packaging/package-thin.sh <out-dir>
#
# It only STAGES. It never passes a positional repo id to `tt-model package-thin` (with one,
# the command uploads), so nothing is pushed. Publishing is a separate step that follows
# hardware verification.
#
# THE SHAPE (docs/E2E_RECIPE_V6.md in tt-model-manager)
#   engine          stock ttnn==0.78.0 from PyPI (requirements.txt). No forked ttnn.
#   custom kernels  none. V-JEPA2 is written in ttnn ops, so there is no generic_op wheel and
#                   deps.custom_ops stays empty.
#   tooling         tt-transformers as TOOLING ONLY: a wheel with models/ emptied, built by
#                   make_tooling_wheel.py from a pinned commit and shipped through --ops-wheel.
#                   The model uses two of its modules: modules.lightweightmodule and
#                   tensor_utils (see tt/functional_encoder.py and tt/functional_predictor.py).
#   model code      shipped INSIDE the bundle through --extra-code: tt/, gradio_app/ (backends and
#                   planning only) and service/. This replaces the old vjepa2-ttnn wheel plus
#                   a hand-made "models closure" wheel of tt-metal's models/common files.
#   safety          --constraints with fully resolved pins; no index options in requirements.txt.
#   huggingface_hub the served code downloads the pinned checkpoint at first serve (a tt-dit-server
#                   run.sh does no weights prefetch), so the venv needs huggingface_hub. Its
#                   closure (requests, certifi, urllib3) is exactly what the PyTorch CPU index
#                   would supply at stale versions, so it is not in requirements.txt. It ships as
#                   a bundled wheel through --ops-wheel, installed by path after the engine step
#                   and resolved from PyPI under constraints.txt. That is the same route the
#                   tooling wheel takes, for the same reason.
#   front end       kind tt-dit-server: no tokens, no KV cache, no continuous batching, so there
#                   is no vLLM, no plugin wheel and no tool or reasoning parser to declare.
#
# INPUTS
#   TT_TRANSFORMERS_SRC  a tt-transformers git checkout (default ~/code/tt-transformers). Only
#                        the commit named by TT_TRANSFORMERS_REF is used, through `git archive`,
#                        so local edits and untracked files cannot leak into the wheel.
#   TOOLING_WHEEL        optional. A tooling wheel you already built; skips the build. Its
#                        filename must be tt_transformers-*.whl.
#   HF_HUB_WHEEL         optional. A huggingface_hub-<HF_HUB_VERSION> wheel you already have;
#                        skips the download. It is checked against the pinned sha256 either way.
#   TT_MODEL             the tt-model executable (default: tt-model on PATH).
#   WORK_DIR             where intermediate files go (default: a fresh temp dir, removed on exit).
#
# Wheels this bundle ships, and where each comes from. Nothing needs a published HF bundle or a
# hand-built artifact: this bundle has no vLLM wheel and no plugin wheel (kind tt-dit-server).
#   tt_transformers  rebuilt from the pinned tt-transformers commit (make_tooling_wheel.py)
#   huggingface_hub  downloaded from PyPI at the pinned version, sha256 checked
#
# package-thin prints "--ops-wheel not recorded as a custom op" for both. That is the intended
# result: neither is a generic_op kernel wheel, so deps.custom_ops stays empty.
#
# packaging/test_package_thin.py runs this script against a stub tt-model and checks every
# flag below, so a flag that drifts from the code fails a test instead of shipping.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
OUT="${1:?usage: package-thin.sh <out-dir>}"
TT_MODEL="${TT_MODEL:-tt-model}"
TT_TRANSFORMERS_SRC="${TT_TRANSFORMERS_SRC:-$HOME/code/tt-transformers}"
# tenstorrent/tt-transformers main on 2026-10-16 (the commit the tooling wheel is cut from).
TT_TRANSFORMERS_REF="b7d1ab5257da73c588a61720fcfafec246c6fa4f"

# The bundle's identity. The weights revision must equal PINNED_WEIGHTS_REVISION in
# gradio_app/backends.py (a test enforces it), because the served code asks the Hub for that
# exact revision of the checkpoint.
NAME="tt-vjepa2"
WEIGHTS_REPO="episod/vjepa2-ac-vitg-fpc64-256-droid-tt"
WEIGHTS_REVISION="16c90813600ae343866eff37b6b4ad1e834a6a8a"
TTNN_VERSION="0.78.0"
# huggingface_hub 0.36.2 (Hugging Face's own client; a 1.x release would switch it to httpx).
# Must equal HF_HUB_VERSION in regen-constraints.sh (a test checks).
HF_HUB_VERSION="0.36.2"
HF_HUB_SHA256="48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270"
HF_HUB_URL="https://files.pythonhosted.org/packages/a8/af/48ac8483240de756d2438c380746e7130d1c6f75802ef22f3c6d49982787/huggingface_hub-$HF_HUB_VERSION-py3-none-any.whl"
# "3.12", the interpreter every published thin bundle uses. A patch-level pin such as 3.12.14
# fails on a host whose uv predates that release ("No download found for request"); this
# box's uv 0.9.26 is one. The package pins in constraints.txt carry the reproducibility.
PYTHON_VERSION="3.12"

if [ -n "${WORK_DIR:-}" ]; then
  mkdir -p "$WORK_DIR"
else
  WORK_DIR="$(mktemp -d)"
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

# 1. extra_code/: the model's own code, as an explicit allowlist. Tests, the Gradio demo
#    (app.py, moves.py, robot_viz.py), benchmarks and the vendored reference/ tree stay out.
#    The served path imports exactly these files (the packaging test walks their imports).
EXTRA="$WORK_DIR/extra_code"
rm -rf "$EXTRA"
mkdir -p "$EXTRA/tt" "$EXTRA/gradio_app" "$EXTRA/service"
cp "$REPO_ROOT"/tt/{__init__,functional_encoder,functional_predictor}.py "$EXTRA/tt/"
cp "$REPO_ROOT"/gradio_app/{__init__,backends,planning}.py "$EXTRA/gradio_app/"
cp "$REPO_ROOT"/service/{__init__,main,wire,worker}.py "$EXTRA/service/"

# 2. The tooling-only tt-transformers wheel.
if [ -n "${TOOLING_WHEEL:-}" ]; then
  WHEEL="$TOOLING_WHEEL"
else
  WHEEL="$(python3 "$HERE/make_tooling_wheel.py" "$TT_TRANSFORMERS_SRC" "$TT_TRANSFORMERS_REF" "$WORK_DIR/wheels")"
fi

# 2b. The huggingface_hub wheel, from PyPI unless HF_HUB_WHEEL names one. Either way the
#     bytes must match the pinned sha256 before they go into the bundle.
if [ -n "${HF_HUB_WHEEL:-}" ]; then
  HUB_WHEEL="$HF_HUB_WHEEL"
else
  mkdir -p "$WORK_DIR/wheels"
  HUB_WHEEL="$WORK_DIR/wheels/huggingface_hub-$HF_HUB_VERSION-py3-none-any.whl"
  curl -fsSL -o "$HUB_WHEEL" "$HF_HUB_URL"
fi
echo "$HF_HUB_SHA256  $HUB_WHEEL" | sha256sum --check --status \
  || { echo "package-thin.sh: $HUB_WHEEL does not match the pinned sha256" >&2; exit 1; }

# 3. Provenance: what this bundle was cut from. package-thin stages the file as PROVENANCE.json.
PROVENANCE="$WORK_DIR/provenance.json"
REPO_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
python3 - "$PROVENANCE" "$REPO_SHA" "$TT_TRANSFORMERS_REF" "$TTNN_VERSION" "$HF_HUB_VERSION" "$WEIGHTS_REPO" "$WEIGHTS_REVISION" <<'PYEOF'
import json, sys
path, repo_sha, tt_ref, ttnn, hf_hub, weights, rev = sys.argv[1:]
json.dump({
    "tt_vjepa2_commit": repo_sha,
    "tt_transformers_commit": tt_ref,
    "tt_transformers_wheel": "tooling only (tt_transformers/models/ emptied, Requires-Dist removed)",
    "ttnn": ttnn,
    "huggingface_hub": hf_hub,
    "weights": weights,
    "weights_revision": rev,
    "recipe": "packaging/package-thin.sh",
}, open(path, "w"), indent=2)
PYEOF

# 4. Install guards (--verify). They run at the end of install.sh with the venv's python and
#    fail the install, so a wrong layout never reaches the first request. install.sh does not
#    export $HERE, so each guard finds the bundle from sys.prefix: the venv is <bundle>/venv.
#    Pure checks only: none of them imports ttnn.
GUARD_TTNN="import importlib.metadata as m; v=m.version('ttnn'); assert v=='$TTNN_VERSION', 'ttnn '+v+' is not $TTNN_VERSION'; h=m.version('huggingface_hub'); assert h=='$HF_HUB_VERSION', 'huggingface_hub '+h+' is not $HF_HUB_VERSION'"
GUARD_TOOLING="import importlib.util as u, os; d=os.path.dirname(u.find_spec('tt_transformers').origin); left=[e for e in os.listdir(os.path.join(d,'models')) if e not in ('__init__.py','__pycache__')]; assert not left, 'tt_transformers ships model code: '+repr(left); src=open(os.path.join(d,'tensor_utils.py')).read(); assert 'def get_rot_transformation_mat' in src, 'tt_transformers.tensor_utils lacks get_rot_transformation_mat'; assert os.path.isfile(os.path.join(d,'modules','lightweightmodule.py')), 'tt_transformers lacks modules/lightweightmodule.py'"
GUARD_CODE="import os, sys; b=os.path.dirname(sys.prefix); need=['tt/functional_encoder.py','tt/functional_predictor.py','gradio_app/backends.py','gradio_app/planning.py','service/main.py','service/wire.py','service/worker.py']; missing=[f for f in need if not os.path.isfile(os.path.join(b,'extra_code',f))]; assert not missing, 'missing from extra_code: '+repr(missing)"

# 5. Stage. No positional repo id: this writes <out-dir> and uploads nothing.
rm -rf "$OUT"
"$TT_MODEL" package-thin \
  --out "$OUT" --name "$NAME" \
  --kind tt-dit-server --app service.main:app --no-vllm \
  --model-py "$HERE/model.py" \
  --extra-code "$EXTRA" \
  --requirements "$HERE/requirements.txt" \
  --constraints "$HERE/constraints.txt" \
  --python "$PYTHON_VERSION" \
  --ops-wheel "$WHEEL" \
  --ops-wheel "$HUB_WHEEL" \
  --arch blackhole --mesh P150 --device-count 1 \
  --weights "$WEIGHTS_REPO" --weights-revision "$WEIGHTS_REVISION" \
  --env VJEPA2_BACKEND=ttnn \
  --verify "$GUARD_TTNN" \
  --verify "$GUARD_TOOLING" \
  --verify "$GUARD_CODE" \
  --provenance "$PROVENANCE"

echo "staged $NAME (tt-vjepa2 $REPO_SHA, tt-transformers $TT_TRANSFORMERS_REF) in $OUT"
