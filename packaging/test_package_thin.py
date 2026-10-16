# SPDX-License-Identifier: MIT
"""Gates for packaging/package-thin.sh, the one recipe the tt-vjepa2 v6 bundle is staged from.

No hardware, no network except one test that builds a tiny wheel with `uv build`. The tests run
the real script against a stub `tt-model` that records its argv, then check:

  * the canonical shape of the command (stock ttnn, tooling-only tt-transformers through
    --ops-wheel, model code through --extra-code, constraints, no repo id, no vLLM);
  * that every import in the shipped code resolves (the packaging recipe's Step 2c);
  * that each --verify install guard accepts a good layout AND rejects the bad ones it exists
    for (a guard that has never been seen to reject is only a claim);
  * that the weights revision in the script equals the one the served code asks the Hub for.

What these do not cover: install.sh, `import model` from the installed bundle, and serving.
Those need the real venv and a card; the report that accompanies a bundle lists them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING = REPO_ROOT / "packaging"
SCRIPT = PACKAGING / "package-thin.sh"

# Stand-in for `tt-model package-thin`: record argv, one token per line. It also stages
# nothing: these tests read the script's inputs (extra_code, provenance) and leave the
# real tool's output to the manual inspection.
STUB_TT_MODEL = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_ARGS"
"""


def _script_text() -> str:
    return SCRIPT.read_text()


def _script_value(name: str) -> str:
    """The value of `NAME="..."` in package-thin.sh."""
    match = re.search(rf'^{name}="([^"]*)"', _script_text(), re.MULTILINE)
    assert match, f"{name} not set in package-thin.sh"
    return match.group(1)


def _write_stub(tmp: Path) -> Path:
    stub = tmp / "tt-model"
    stub.write_text(STUB_TT_MODEL)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """Run package-thin.sh once against the stub. Returns (argv, work_dir, tooling_wheel, hub_wheel,
    packaging_dir).

    The script checks the huggingface_hub wheel against a pinned sha256, and a test cannot ship
    the real 450 KB wheel. So the script runs from a COPY of the repo whose pinned sha256 is
    that of the fake wheel. The copy still checks every other flag and file, and
    test_hub_wheel_must_match_the_pinned_sha256 runs the original script against a wrong wheel
    to show that the check itself rejects."""
    tmp = tmp_path_factory.mktemp("stage")
    repo = tmp / "repo"
    for name in ("tt", "gradio_app", "service", "packaging"):
        shutil.copytree(REPO_ROOT / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    for git_args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(repo), *git_args], check=True, capture_output=True)

    tooling = tmp / "tt_transformers-2.0.0.dev0-py3-none-any.whl"
    tooling.write_bytes(b"not a real wheel; the stub never opens it")
    hub = tmp / "huggingface_hub-0.36.2-py3-none-any.whl"
    hub.write_bytes(b"not a real wheel either")
    script = repo / "packaging" / "package-thin.sh"
    script.write_text(re.sub(r'^HF_HUB_SHA256="[0-9a-f]+"', f'HF_HUB_SHA256="{hashlib.sha256(hub.read_bytes()).hexdigest()}"',
                             script.read_text(), flags=re.M))
    work = tmp / "work"
    env = {
        **os.environ,
        "TT_MODEL": str(_write_stub(tmp)),
        "TOOLING_WHEEL": str(tooling),
        "HF_HUB_WHEEL": str(hub),
        "WORK_DIR": str(work),
        "STUB_ARGS": str(tmp / "args"),
    }
    subprocess.run(["bash", str(script), str(tmp / "out")], env=env, check=True, capture_output=True, text=True)
    return (tmp / "args").read_text().splitlines(), work, tooling, hub, repo / "packaging"


def _values(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def _only(argv, flag):
    (value,) = _values(argv, flag)
    return value


# ---------------------------------------------------------------- the shape of the command


def test_stages_only_with_no_repo_id(staged):
    """Every token is a flag or the value right after a flag. A bare token would be the
    positional repo id, which makes package-thin upload."""
    argv = staged[0]
    assert argv[0] == "package-thin"
    boolean = {"--no-vllm"}
    i = 1
    while i < len(argv):
        assert argv[i].startswith("--"), f"bare token {argv[i]!r} would be read as a repo id"
        i += 1 if argv[i] in boolean else 2
    assert "--out" in argv


def test_canonical_flags(staged):
    argv, work, tooling, hub, packaging = staged
    assert _only(argv, "--kind") == "tt-dit-server"
    assert _only(argv, "--app") == "service.main:app"
    assert "--no-vllm" in argv
    # No vLLM pieces: a tt-dit-server bundle has nothing for vLLM to do.
    for flag in ("--plugin-wheel", "--vllm-wheel", "--vllm-version", "--tool-parser", "--reasoning-parser"):
        assert flag not in argv
    # Model code ships in the bundle. No whole-models-tree wheel, no vjepa2 wheel.
    assert "--models-wheel" not in argv
    assert _only(argv, "--extra-code") == str(work / "extra_code")
    # The two --ops-wheels are tooling-only tt-transformers and huggingface_hub. There are no
    # generic_op wheels, because the model has no custom kernels.
    assert _values(argv, "--ops-wheel") == [str(tooling), str(hub)]
    assert Path(_only(argv, "--constraints")) == packaging / "constraints.txt"
    assert Path(_only(argv, "--requirements")) == packaging / "requirements.txt"
    assert _only(argv, "--arch") == "blackhole"
    assert _only(argv, "--mesh") == "P150"
    assert _only(argv, "--device-count") == "1"
    assert _only(argv, "--weights") == "episod/vjepa2-ac-vitg-fpc64-256-droid-tt"
    assert "VJEPA2_BACKEND=ttnn" in _values(argv, "--env")


def test_provenance_is_a_path_to_json(staged):
    argv = staged[0]
    path = Path(_only(argv, "--provenance"))
    data = json.loads(path.read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", data["tt_vjepa2_commit"])
    assert data["tt_transformers_commit"] == _script_value("TT_TRANSFORMERS_REF")
    assert data["ttnn"] == _script_value("TTNN_VERSION")


def test_extra_code_is_the_allowlist(staged):
    """The staged extra_code holds the served path and nothing else: no tests, no demo UI, no
    benchmarks."""
    work = staged[1]
    files = sorted(str(p.relative_to(work / "extra_code")) for p in (work / "extra_code").rglob("*.py"))
    assert files == [
        "gradio_app/__init__.py",
        "gradio_app/backends.py",
        "gradio_app/planning.py",
        "service/__init__.py",
        "service/main.py",
        "service/wire.py",
        "service/worker.py",
        "tt/__init__.py",
        "tt/functional_encoder.py",
        "tt/functional_predictor.py",
    ]


# ------------------------------------------------------------- pins and the weights revision


def _pin_names(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def test_requirements_have_no_index_options_and_no_local_ttnn():
    text = (PACKAGING / "requirements.txt").read_text()
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        assert not stripped.startswith("-"), f"index/option line in requirements.txt: {line!r}"
    ttnn = _pin_names(PACKAGING / "requirements.txt")["ttnn"]
    assert "+" not in ttnn, "a +local tag means a forked or custom-built ttnn"
    assert ttnn == _script_value("TTNN_VERSION")


def test_constraints_pin_every_requirement_and_the_networking_packages():
    constraints = _pin_names(PACKAGING / "constraints.txt")
    # requirements.txt mixes exact pins and ranges, so collect the bare names.
    requirement_names = set()
    for line in (PACKAGING / "requirements.txt").read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            requirement_names.add(re.split(r"[=<>!~ ]", stripped, 1)[0].lower().replace("_", "-"))
    assert requirement_names <= set(constraints), requirement_names - set(constraints)
    # The packages the PyTorch CPU index can silently downgrade.
    for name in ("certifi", "urllib3", "requests", "idna"):
        assert name in constraints, f"{name} is not pinned in constraints.txt"
    assert constraints["ttnn"] == _script_value("TTNN_VERSION")


def test_weights_revision_matches_the_served_code():
    served = re.search(
        r'^PINNED_WEIGHTS_REVISION = "([0-9a-f]{40})"', (REPO_ROOT / "gradio_app" / "backends.py").read_text(), re.M
    )
    assert served, "PINNED_WEIGHTS_REVISION missing from gradio_app/backends.py"
    assert served.group(1) == _script_value("WEIGHTS_REVISION")
    repo = re.search(r'^WEIGHTS_REPO = "([^"]+)"', (REPO_ROOT / "gradio_app" / "backends.py").read_text(), re.M)
    assert repo.group(1) == _script_value("WEIGHTS_REPO")


# --------------------------------------------------- every import resolves (recipe Step 2c)

# Imports that stay unresolved in the bundle on purpose. Both are lazy, inside a function that
# the served path never calls. A new entry here is a decision, so it is listed by name.
KNOWN_UNRESOLVED = {
    "httpx": "RemoteBackend, the Gradio demo's HTTP client; the bundle serves the app, it is not a client",
    "src": "ReferenceBackend, which imports Meta's reference/ tree; that tree is not shipped",
}
# import name -> distribution name in constraints.txt
THIRD_PARTY = {
    "ttnn": "ttnn",
    "torch": "torch",
    "numpy": "numpy",
    "PIL": "pillow",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface-hub",
    "fastapi": "fastapi",
    "starlette": "starlette",
    "pydantic": "pydantic",
    "uvicorn": "uvicorn",
}
FIRST_PARTY = {"tt", "gradio_app", "service"}
TOOLING = "tt_transformers"


def _imports(py: Path, root: Path):
    """Absolute module names imported by `py` (relative imports resolved against `root`)."""
    tree = ast.parse(py.read_text())
    package = list(py.relative_to(root).with_suffix("").parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, []
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            yield module, [a.name for a in node.names]


def _exists_in_tree(root: Path, module: str) -> bool:
    parts = module.split(".")
    return (root.joinpath(*parts).with_suffix(".py")).is_file() or root.joinpath(*parts, "__init__.py").is_file()


def _exists_at_ref(checkout: Path, ref: str, module: str) -> bool:
    parts = module.split(".")
    for candidate in ("/".join(["src", *parts]) + ".py", "/".join(["src", *parts, "__init__.py"])):
        result = subprocess.run(
            ["git", "-C", str(checkout), "cat-file", "-e", f"{ref}:{candidate}"], capture_output=True
        )
        if result.returncode == 0:
            return True
    return False


def test_every_shipped_import_resolves(staged):
    work = staged[1]
    extra = work / "extra_code"
    constraints = _pin_names(PACKAGING / "constraints.txt")
    checkout = Path(os.environ.get("TT_TRANSFORMERS_SRC", Path.home() / "code" / "tt-transformers"))
    ref = _script_value("TT_TRANSFORMERS_REF")
    have_checkout = (checkout / ".git").exists() and subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", ref], capture_output=True
    ).returncode == 0

    unresolved, tooling_seen = set(), set()
    # model.py is part of the bundle too: it sits at the bundle root, next to extra_code/.
    sources = [(p, extra) for p in extra.rglob("*.py")] + [(PACKAGING / "model.py", PACKAGING)]
    for py, root in sources:
        for module, names in _imports(py, root):
            top = module.split(".")[0]
            if not module or top == "__future__" or top in sys.stdlib_module_names:
                continue
            if top in FIRST_PARTY:
                assert _exists_in_tree(extra, module), f"{py.name}: {module} is not in extra_code"
            elif top == TOOLING:
                tooling_seen.add(module)
                assert not module.startswith("tt_transformers.models"), (
                    f"{py.name} imports {module}: that is another model's code, not tooling"
                )
            elif top in THIRD_PARTY:
                assert THIRD_PARTY[top] in constraints, f"{top} is not pinned in constraints.txt"
            elif top in KNOWN_UNRESOLVED:
                unresolved.add(top)
            else:
                pytest.fail(f"{py.name} imports {module}: not stdlib, extra_code, tooling or a pinned package")

    assert unresolved <= set(KNOWN_UNRESOLVED)
    # The model uses exactly these two tooling modules.
    assert tooling_seen == {"tt_transformers.modules.lightweightmodule", "tt_transformers.tensor_utils"}
    if not have_checkout:
        pytest.skip(f"no tt-transformers checkout with {ref} to confirm the tooling modules exist")
    for module in tooling_seen:
        assert _exists_at_ref(checkout, ref, module), f"{module} does not exist at tt-transformers {ref}"


def test_the_old_models_common_import_is_gone():
    for py in list((REPO_ROOT / "tt").glob("*.py")) + list((REPO_ROOT / "gradio_app").glob("*.py")):
        if py.name.startswith("test_"):
            continue
        assert "models.common" not in re.sub(r"#.*|\"\"\"[\s\S]*?\"\"\"", "", py.read_text()), py.name


# --------------------------------------------------------------- the --verify install guards


def _guards(argv):
    ttnn, tooling, code = _values(argv, "--verify")
    return ttnn, tooling, code


def _run(statement: str, preamble: str = "", pythonpath: str = "") -> subprocess.CompletedProcess:
    """Run a guard the way install.sh does (`python -c <statement>`), with `preamble` run first
    in the same process to set up the fake environment."""
    env = {**os.environ, "PYTHONPATH": pythonpath}
    return subprocess.run(
        [sys.executable, "-c", preamble + "\n" + statement], env=env, capture_output=True, text=True
    )


def _fake_tt_transformers(root: Path, *, models=(), tensor_fn=True, lightweight=True) -> Path:
    pkg = root / "tt_transformers"
    (pkg / "models").mkdir(parents=True)
    (pkg / "modules").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "models" / "__init__.py").write_text("")
    for name in models:
        (pkg / "models" / name).mkdir()
    (pkg / "tensor_utils.py").write_text("def get_rot_transformation_mat(dhead=32): pass\n" if tensor_fn else "x = 1\n")
    if lightweight:
        (pkg / "modules" / "lightweightmodule.py").write_text("class LightweightModule: pass\n")
    return root


def test_ttnn_guard(staged):
    ttnn_guard = _guards(staged[0])[0]
    pin, hub_pin = _script_value("TTNN_VERSION"), _script_value("HF_HUB_VERSION")

    def pretend(ttnn, hub):
        versions = {"ttnn": ttnn, "huggingface_hub": hub}
        return f"import importlib.metadata as m; m.version = lambda name: {versions!r}[name]"

    assert _run(ttnn_guard, pretend(pin, hub_pin)).returncode == 0
    assert _run(ttnn_guard, pretend("0.77.0", hub_pin)).returncode != 0
    assert _run(ttnn_guard, pretend(pin + "+local", hub_pin)).returncode != 0
    assert _run(ttnn_guard, pretend(pin, "1.0.0")).returncode != 0


def test_tooling_guard(staged, tmp_path):
    guard = _guards(staged[0])[1]
    good = _fake_tt_transformers(tmp_path / "good")
    assert _run(guard, pythonpath=str(good)).returncode == 0
    # __pycache__ in models/ is normal and must not fail it.
    (good / "tt_transformers" / "models" / "__pycache__").mkdir()
    assert _run(guard, pythonpath=str(good)).returncode == 0
    # A tt-transformers that still carries a model.
    with_model = _fake_tt_transformers(tmp_path / "with_model", models=("llama3_8b",))
    bad = _run(guard, pythonpath=str(with_model))
    assert bad.returncode != 0 and "ships model code" in bad.stderr
    # A tt-transformers without the function the model needs.
    no_fn = _fake_tt_transformers(tmp_path / "no_fn", tensor_fn=False)
    assert _run(guard, pythonpath=str(no_fn)).returncode != 0
    no_lw = _fake_tt_transformers(tmp_path / "no_lw", lightweight=False)
    assert _run(guard, pythonpath=str(no_lw)).returncode != 0


def test_extra_code_guard(staged, tmp_path):
    guard = _guards(staged[0])[2]
    work = staged[1]

    def bundle(name, drop=None):
        root = tmp_path / name
        (root / "venv").mkdir(parents=True)
        subprocess.run(["cp", "-r", str(work / "extra_code"), str(root / "extra_code")], check=True)
        if drop:
            (root / "extra_code" / drop).unlink()
        return root

    def run_in(root):
        # install.sh runs the guard with the venv's python, so sys.prefix is <bundle>/venv.
        return _run(guard, f"import sys; sys.prefix = {str(root / 'venv')!r}")

    assert run_in(bundle("good")).returncode == 0
    bad = run_in(bundle("no_backend", drop="gradio_app/backends.py"))
    assert bad.returncode != 0 and "missing from extra_code" in bad.stderr
    assert run_in(bundle("no_worker", drop="service/worker.py")).returncode != 0


def test_hub_version_matches_between_script_and_constraints_generator():
    generator = (PACKAGING / "regen-constraints.sh").read_text()
    assert re.search(rf'^HF_HUB_VERSION="{re.escape(_script_value("HF_HUB_VERSION"))}"', generator, re.M)
    assert _pin_names(PACKAGING / "constraints.txt")["huggingface-hub"] == _script_value("HF_HUB_VERSION")


def test_hub_wheel_must_match_the_pinned_sha256(tmp_path):
    """The ORIGINAL script, given a wheel whose bytes are not the pinned ones, refuses to stage."""
    stub = _write_stub(tmp_path)
    tooling = tmp_path / "tt_transformers-2.0.0.dev0-py3-none-any.whl"
    tooling.write_bytes(b"x")
    wrong = tmp_path / "huggingface_hub-0.36.2-py3-none-any.whl"
    wrong.write_bytes(b"tampered")
    result = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "out")],
        env={**os.environ, "TT_MODEL": str(stub), "TOOLING_WHEEL": str(tooling), "HF_HUB_WHEEL": str(wrong),
             "WORK_DIR": str(tmp_path / "w"), "STUB_ARGS": str(tmp_path / "args")},
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "pinned sha256" in result.stderr
    assert not (tmp_path / "args").exists(), "tt-model ran even though the wheel check failed"


def test_idna_is_a_direct_url_so_the_cpu_index_cannot_supply_it():
    lines = [l.split("#", 1)[0].strip() for l in (PACKAGING / "requirements.txt").read_text().splitlines()]
    (idna,) = [l for l in lines if l.startswith("idna")]
    assert re.fullmatch(r"idna @ https://files\.pythonhosted\.org/packages/.+/idna-[0-9.]+-py3-none-any\.whl", idna)
    assert _pin_names(PACKAGING / "constraints.txt")["idna"] == re.search(r"idna-([0-9.]+)-", idna).group(1)


# ------------------------------------------------------------- the tooling wheel builder


@pytest.mark.skipif(subprocess.run(["which", "uv"], capture_output=True).returncode != 0, reason="needs uv")
def test_tooling_wheel_keeps_tooling_and_drops_models_and_pins(tmp_path):
    """Builds a tiny tt-transformers look-alike in a throwaway git repo (needs uv, and the
    network the first time uv fetches setuptools)."""
    repo = tmp_path / "tt-transformers"
    pkg = repo / "src" / "tt_transformers"
    (pkg / "models" / "llama3_8b").mkdir(parents=True)
    (pkg / "modules").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "tensor_utils.py").write_text("def get_rot_transformation_mat(): pass\n")
    (pkg / "modules" / "__init__.py").write_text("")
    (pkg / "modules" / "lightweightmodule.py").write_text("class LightweightModule: pass\n")
    (pkg / "models" / "__init__.py").write_text("")
    (pkg / "models" / "llama3_8b" / "__init__.py").write_text("")
    (pkg / "models" / "llama3_8b" / "model.py").write_text("SECRET = 1\n")
    (repo / "pyproject.toml").write_text(textwrap.dedent("""\
        [build-system]
        requires = ["setuptools"]
        build-backend = "setuptools.build_meta"
        [project]
        name = "tt-transformers"
        version = "2.0.0.dev0"
        dependencies = ["ttnn==0.79.0", "torch==2.11.0"]
        [project.optional-dependencies]
        test = ["pytest"]
        [tool.setuptools]
        package-dir = {"" = "src"}
        [tool.setuptools.packages.find]
        where = ["src"]
        """))
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    ref = git("rev-parse", "HEAD").stdout.strip()
    # An uncommitted file must not reach the wheel: the builder uses `git archive`.
    (pkg / "untracked.py").write_text("LEAK = 1\n")

    out = subprocess.run(
        [sys.executable, str(PACKAGING / "make_tooling_wheel.py"), str(repo), ref, str(tmp_path / "out")],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        metadata = zf.read(next(n for n in names if n.endswith("METADATA"))).decode()
    assert "tt_transformers/tensor_utils.py" in names
    assert "tt_transformers/modules/lightweightmodule.py" in names
    assert "tt_transformers/models/__init__.py" in names
    assert not [n for n in names if n.startswith("tt_transformers/models/") and not n.endswith("__init__.py")]
    assert "tt_transformers/untracked.py" not in names
    assert "Requires-Dist" not in metadata and "Provides-Extra" not in metadata
    assert any(n.endswith("RECORD") for n in names)


# --------------------------------------------------------------- the model.py entry point


def test_model_py_imports_no_ops_wheel_and_no_tooling():
    """`deps.custom_ops` is recorded from model.py's imports. V-JEPA2 has no custom kernels, and
    an import of tt_transformers here would be recorded as one."""
    imported = {m.split(".")[0] for m, _ in _imports(PACKAGING / "model.py", PACKAGING)}
    assert imported == {"service"}
