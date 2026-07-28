"""Shared test machinery: golden-fixture loading, tolerant comparison, skips.

Why comparison needs its own helper
------------------------------------
``numpy.testing.assert_allclose`` on a dict of ~150 keys (a full
``PhysicsConsistency.report()`` flattened, say) fails on the FIRST mismatch it
finds, in whatever order ``dict`` iteration happens to give you -- usually not
the most informative one. A port bug that is off by 1e-4 in ``mean_jerk_mps3``
and a port bug that returns the wrong SHAPE for ``bone_lengths`` look
identical from "AssertionError: not close" with no further context. Sorting
every mismatching key by its worst relative error and printing that table
(:func:`_assert_close_dict_impl`) turns a bisection session into a five-second
read.

Why the skip machinery is markers, not ``pytest.importorskip``
------------------------------------------------------------------
The four environment dependencies this benchmark can be missing (a CUDA
device, network, a trained checkpoint, ffmpeg) are each conditions on the
*environment*, not on whether a Python import succeeds -- ``torch`` is always
importable whether or not a GPU exists. Markers let a test declare "I need a
GPU" once via ``@pytest.mark.gpu`` and get skipped everywhere that's false,
instead of every GPU test repeating an ``if not torch.cuda.is_available():
pytest.skip(...)`` guard (and inevitably a few forgetting to).
"""
from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest

__all__ = ["assert_close_dict", "load_golden"]

GOLDEN_DIR = Path(__file__).parent / "golden"

#: Env var pointing at a Marionette-ciasc checkout, so `ckpt`-marked tests can
#: probe for real checkpoints without this file hardcoding anyone's home
#: directory. Unset (the common case in CI / a fresh clone) => `ckpt` tests
#: skip with an actionable message rather than erroring on a missing path.
CKPT_ROOT_ENV = "KINESCORE_CIASC_ROOT"

#: Env var pointing at a Marionette-fkjepa checkout -- a DIFFERENT source
#: repo from CKPT_ROOT_ENV's Marionette-ciasc (see docs/PROVENANCE.md's A/B
#: split). This is where the GR-1 ReadoutV2Head production checkpoint lives
#: (`model_ckpt/readout_v2_gr1.pt`), so `ckpt`-marked tests that need it
#: (tests/test_checkpoint_v2_real_ckpt.py) probe this instead.
FKJEPA_ROOT_ENV = "KINESCORE_FKJEPA_ROOT"


# ===========================================================================
# assert_close_dict
# ===========================================================================

def _assert_close_dict_impl(got: dict, want: dict, rtol: float = 1e-5,
                            atol: float = 1e-6, keys=None) -> None:
    """Compare two ``{key: array-like}`` dicts; raise with the worst key first.

    Parameters
    ----------
    got, want:
        Dicts of anything ``np.asarray`` accepts (Python floats, numpy
        arrays, 0-d values straight out of an ``.npz`` file). ``want`` is
        the golden/legacy side; ``got`` is the side under test -- this
        ordering only affects which is labelled which in the failure
        message, not the tolerance check (which is symmetric via
        ``np.allclose``).
    rtol, atol:
        Passed straight to :func:`numpy.allclose` for the pass/fail
        decision. The printed ``rel`` column uses ``atol`` as the
        denominator floor (``abs(want) + atol``) so a key whose golden
        value is exactly 0 doesn't divide by zero or produce a spurious
        ``inf``.
    keys:
        Restrict comparison to these keys (default: the intersection of
        ``got`` and ``want``). Keys present in ``want`` but absent from
        ``got`` are always a hard failure, listed separately -- a missing
        key is a stronger regression signal than a numeric drift and
        shouldn't get buried in the mismatch table.
    """
    want_keys = set(keys) if keys is not None else set(want)
    missing = sorted(want_keys - set(got))
    if missing:
        raise AssertionError(
            f"assert_close_dict: {len(missing)} key(s) present in `want` but "
            f"missing from `got`: {missing[:20]}"
            + (" ... (truncated)" if len(missing) > 20 else "")
        )

    compare_keys = sorted(want_keys & set(got))
    failures: list[tuple[str, float, float, np.ndarray, np.ndarray]] = []
    for k in compare_keys:
        g = np.asarray(got[k])
        w = np.asarray(want[k])
        if g.shape != w.shape:
            failures.append((k, float("inf"), float("inf"), g, w))
            continue
        if not (np.issubdtype(w.dtype, np.number) or w.dtype == bool):
            # Non-numeric golden arrays (pose_names, cfg_json, ckpt_sha256,
            # quantity_keys -- several fixtures carry string/bool metadata
            # alongside the numeric outputs). No tolerance concept applies;
            # exact equality is the only meaningful check, reported the same
            # way (abs/rel columns become 0/inf so the table stays uniform).
            if not np.array_equal(g, w):
                failures.append((k, float("nan"), float("inf"), g, w))
            continue
        gf = g.astype(np.float64, copy=False)
        wf = w.astype(np.float64, copy=False)
        abs_diff = np.abs(gf - wf)
        rel_diff = abs_diff / (np.abs(wf) + atol)
        max_abs = float(np.nanmax(abs_diff)) if abs_diff.size else 0.0
        max_rel = float(np.nanmax(rel_diff)) if rel_diff.size else 0.0
        if not np.allclose(gf, wf, rtol=rtol, atol=atol, equal_nan=True):
            failures.append((k, max_abs, max_rel, g, w))

    if not failures:
        return

    failures.sort(key=lambda f: f[2], reverse=True)  # worst relative error first
    lines = [
        f"assert_close_dict: {len(failures)}/{len(compare_keys)} keys "
        f"mismatched (rtol={rtol}, atol={atol})"
    ]
    for k, max_abs, max_rel, g, w in failures[:15]:
        shape_note = "" if g.shape == w.shape else f"  SHAPE got={g.shape} want={w.shape}"
        lines.append(
            f"  {k}: abs={max_abs:.6g} rel={max_rel:.6g}{shape_note}\n"
            f"      got ={np.array2string(g.ravel()[:6], precision=6)}"
            f"{' ...' if g.size > 6 else ''}\n"
            f"      want={np.array2string(w.ravel()[:6], precision=6)}"
            f"{' ...' if w.size > 6 else ''}"
        )
    if len(failures) > 15:
        lines.append(f"  ... and {len(failures) - 15} more mismatching key(s)")
    raise AssertionError("\n".join(lines))


@pytest.fixture
def assert_close_dict():
    """Fixture-factory: inject, then call as ``assert_close_dict(got, want)``.

    A fixture (not a bare top-level function) so every test gets it via
    normal dependency injection with no import path to get wrong; it returns
    a plain callable so the call site reads exactly like the function it
    wraps -- see :func:`_assert_close_dict_impl` for the comparison itself
    and the failure-report format.
    """
    return _assert_close_dict_impl


# ===========================================================================
# golden fixture loading
# ===========================================================================

def _load_golden_impl(name: str) -> dict[str, np.ndarray]:
    """Load ``tests/golden/{name}.npz`` into a plain ``dict[str, ndarray]``.

    ``name`` may be given with or without the ``.npz`` suffix and with or
    without a leading ``golden_`` (``"golden_fk"``, ``"fk"``, and
    ``"golden_fk.npz"`` all resolve to the same file) -- purely a call-site
    convenience, since every golden fixture in this repo follows the
    ``golden_<thing>.npz`` naming convention and typing it in full every time
    a test references it is friction with no payoff.

    Returns a plain ``dict`` (fully materialised, not a lazy ``NpzFile``) so
    the file handle is closed before the test body runs -- these fixtures are
    all well under a megabyte (see ``MANIFEST.json``), so eager loading costs
    nothing and avoids a class of "works until the file handle limit" bugs in
    long test sessions.
    """
    stem = name
    if stem.endswith(".npz"):
        stem = stem[: -len(".npz")]
    if not stem.startswith("golden_"):
        stem = f"golden_{stem}"
    path = GOLDEN_DIR / f"{stem}.npz"
    if not path.is_file():
        available = sorted(p.stem for p in GOLDEN_DIR.glob("golden_*.npz"))
        raise FileNotFoundError(
            f"no golden fixture at {path}. Available: {available}. "
            f"Fixtures are generated by tools/gen_golden.py -- if this repo "
            f"was freshly cloned, run that script (see MANIFEST.json in "
            f"{GOLDEN_DIR} for the exact invocation it was last run with)."
        )
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


@pytest.fixture
def load_golden():
    """Fixture-factory: inject, then call as ``load_golden("golden_fk")``.

    See :func:`_load_golden_impl` for the name-resolution rules and why this
    returns a fully materialised dict rather than the raw ``NpzFile``.
    """
    return _load_golden_impl


# ===========================================================================
# environment probes (shared by the marker skip logic and the URDF fixture)
# ===========================================================================

def _has_network(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.0) -> bool:
    """Best-effort "is there a route to the internet" probe.

    Opens a raw TCP connection to a well-known DNS resolver and immediately
    closes it -- no DNS lookup of our own involved (a lookup would itself
    depend on network), no data sent. False on ANY exception (timeout,
    unreachable, sandboxed-network-namespace-refuses-raw-sockets, etc.) is
    the conservative and correct default: "assume offline" only ever causes
    an unnecessary skip, never a false pass.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ckpt_root() -> Path | None:
    root = os.environ.get(CKPT_ROOT_ENV)
    return Path(root) if root else None


def _fkjepa_root() -> Path | None:
    root = os.environ.get(FKJEPA_ROOT_ENV)
    return Path(root) if root else None


def _ckpt_available() -> bool:
    """``True`` if EITHER real-checkpoint source tree is reachable.

    Two independent checkpoint families live under two independent env
    vars (see ``CKPT_ROOT_ENV``/``FKJEPA_ROOT_ENV`` above) -- a host with
    only one checked out should still run the `ckpt`-marked tests that
    apply to it, so this is an OR, not a requirement that both be present.
    """
    ciasc = _ckpt_root()
    has_ciasc = (ciasc is not None and ciasc.is_dir()
                and any(ciasc.glob("model_ckpt/*/judge.pt")))
    fkjepa = _fkjepa_root()
    has_fkjepa = (fkjepa is not None and fkjepa.is_dir()
                 and (fkjepa / "model_ckpt" / "readout_v2_gr1.pt").is_file())
    return has_ciasc or has_fkjepa


def _ffmpeg_available() -> bool:
    return shutil.which("ffprobe") is not None


# ===========================================================================
# marker -> skip wiring
# ===========================================================================

def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Turn ``@pytest.mark.{gpu,net,ckpt,ffmpeg}`` into skips when unavailable.

    Markers themselves are declared in ``pyproject.toml`` (``[tool.pytest.
    ini_options] markers = [...]``); this hook is what actually enforces
    them. Probes run once per session (not per item) since none of them are
    cheap enough to want repeated 1000 times.
    """
    has_gpu = _cuda_available()
    has_net = _has_network()
    has_ckpt = _ckpt_available()
    has_ffmpeg = _ffmpeg_available()

    skip_gpu = pytest.mark.skip(
        reason="requires a CUDA device (torch.cuda.is_available() is False)")
    skip_net = pytest.mark.skip(
        reason="requires network access; none reachable from this host/sandbox")
    skip_ckpt = pytest.mark.skip(
        reason=f"requires a trained checkpoint on disk; set ${CKPT_ROOT_ENV} to a "
               f"Marionette-ciasc checkout containing model_ckpt/*/judge.pt "
               f"(currently {_ckpt_root() or 'unset'})")
    skip_ffmpeg = pytest.mark.skip(
        reason="requires the ffprobe/ffmpeg binaries on PATH")

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "net" in item.keywords and not has_net:
            item.add_marker(skip_net)
        if "ckpt" in item.keywords and not has_ckpt:
            item.add_marker(skip_ckpt)
        if "ffmpeg" in item.keywords and not has_ffmpeg:
            item.add_marker(skip_ffmpeg)


# ===========================================================================
# Panda URDF availability (FK tests need the robot_descriptions cache)
# ===========================================================================

def _find_cached_panda_urdf() -> Path | None:
    """Locate a cached Panda URDF WITHOUT importing ``robot_descriptions``.

    Accessing ``robot_descriptions.panda_description.URDF_PATH`` triggers a
    network fetch on a cold cache -- exactly what a CPU-only, network-free
    test run must not do. Searching the package's own default cache
    directory first lets us answer "is it already there" without risking
    that fetch.
    """
    cache_root = Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE",
                                     Path.home() / ".cache" / "robot_descriptions"))
    if not cache_root.is_dir():
        return None
    matches = sorted(cache_root.glob("**/panda_description/urdf/panda.urdf"))
    return matches[0] if matches else None


@pytest.fixture(scope="session")
def panda_urdf_path() -> Path:
    """Session fixture: a cached Panda URDF path, or an ACTIONABLE skip.

    Every FK test should depend on this fixture (directly or transitively)
    instead of constructing ``FrankaFK`` and letting a cold-cache network
    fetch either hang (sandboxed/offline CI) or silently pass by touching the
    network in a suite that promises not to (see ``pyproject.toml``:
    ``addopts`` excludes ``net`` by default).
    """
    cached = _find_cached_panda_urdf()
    if cached is not None:
        return cached
    if not _has_network():
        pytest.skip(
            "Panda URDF not found under "
            f"{os.environ.get('ROBOT_DESCRIPTIONS_CACHE', '~/.cache/robot_descriptions')} "
            "and no network reachable to fetch it. Fix ONE of: "
            "(1) run once on a host with network access: `python -c "
            "\"from robot_descriptions import panda_description as p; "
            "print(p.URDF_PATH)\"`, which populates the cache; "
            "(2) copy an existing ~/.cache/robot_descriptions/ directory "
            "from a host that has it and verify it against "
            "tests/golden/MANIFEST.json's `panda_urdf_sha256`; "
            "(3) set ROBOT_DESCRIPTIONS_CACHE to point at such a directory."
        )
    from robot_descriptions import panda_description
    return Path(panda_description.URDF_PATH)
