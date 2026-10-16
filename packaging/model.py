# SPDX-License-Identifier: MIT
"""Bundle entry point (`--model-py`) for the tt-vjepa2 v6 thin bundle.

This bundle's kind is "tt-dit-server": `run.sh` serves the ASGI app `service.main:app` with
uvicorn and never loads this file, so there is no vLLM adapter class to re-export. package-thin
still requires a `--model-py`, and `model.py` is also what `deps.custom_ops` is recorded from,
so the file does two small jobs:

  * It makes `import model` a one-line check that the shipped `service` package imports and
    that the served app object exists (the Step 4b import smoke test).
  * It imports no ops wheel and no `tt_transformers` module. V-JEPA2 has no custom kernels,
    so `deps.custom_ops` is correctly empty. An import of `tt_transformers` here would be
    recorded as a custom op, because that wheel is shipped through `--ops-wheel`.

Importing this touches no hardware: `service.main` opens the device only when the app's
lifespan starts, not at import time.
"""

from service.main import app

__all__ = ["app"]
