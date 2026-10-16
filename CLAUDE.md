# tt-vjepa2: working log

V-JEPA2-AC (encoder plus action-conditioned predictor) on Tenstorrent Blackhole through TTNN.
README.md describes the model and how to run it. This file logs what happened, briefly.

## Packaging, v6 thin bundle (2026-10-16)

**Prompt.** Repackage this model to the canonical v6 thin-bundle shape documented in
tt-model-manager PR #183. Stage only: no push, no `hf upload`, no hardware.

**What changed.**
- Added `packaging/package-thin.sh`, the single rerunnable recipe, plus `requirements.txt`,
  `constraints.txt` (from `regen-constraints.sh`), `make_tooling_wheel.py`, `model.py` and
  `test_package_thin.py`.
- The old recipe lived only in the README and used `--models-wheel` for two hand-made wheels:
  the repo's own `vjepa2-ttnn` wheel and a `tt-vjepa2-models-closure` wheel of two tt-metal
  `models/common` files. It had no requirements file, no constraints and no install guards.
  No `episod/tt-vjepa2` bundle was ever published, so there was no earlier artifact to compare.
- Model code now ships as `extra_code/` from a file list in the script. The two tooling
  imports moved from `models.common.*` to `tt_transformers.modules.lightweightmodule` and
  `tt_transformers.tensor_utils`, and tt-transformers ships as a tooling-only wheel through
  `--ops-wheel`.
- `gradio_app/backends.py` no longer reads a checkpoint only from `/home/ttuser/.cache/...`.
  It resolves `$VJEPA2_CKPT`, then that path if present, then `hf_hub_download` of the pinned
  weights revision. The old behaviour could not work on any other host.
- `pyproject.toml` version 0.1.0 to 0.2.0.

**Decisions.**
- ttnn stays at 0.78.0, the release this model was verified on. tt-transformers' own pins
  (ttnn 0.79.0, torch, transformers) are stripped from the tooling wheel so they cannot
  conflict with it.
- Python is pinned as `3.12`. The build box has uv 0.9.26, which cannot fetch 3.12.14.
- A first `constraints.txt` generated with uv's relaxed index strategy failed the real
  `install.sh` (`networkx==3.7` is not on the PyTorch CPU index, which uv ranks first). The
  constraints are now resolved the way the install resolves, in two steps
  (`regen-constraints.sh`). That also surfaced the index-ranking issue for `idna` (the index
  carries only 3.4), so `idna` is a direct-URL requirement. `huggingface_hub` ships as a
  bundled wheel for the same reason. Remove both workarounds when package-thin scopes the
  CPU index to torch.
- No tool or reasoning parser, no vLLM, no plugin wheel, no `generic_op` wheel: the kind is
  `tt-dit-server` and the model has no custom kernels.

**Verified (no hardware).** Recipe stages with the installed tt-model; `install.sh` exits 0 in
a copy with all `--verify` guards run; `import model` loads from `extra_code/` without
importing `ttnn`; `pytest packaging gradio_app/test_backends.py service` passes, and each new
test was seen to fail with its behaviour broken.

**Not verified.** `tt.*` imports against a live ttnn, serving, `/encode` `/predict_step`
`/plan_step` on a card, and the HF cache download path against the real Hub.
