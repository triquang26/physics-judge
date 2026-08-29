"""Shared test machinery: golden-fixture loading, tolerant comparison, skips.

:func:`assert_close_dict` compares whole dicts of arrays and reports every
mismatching key sorted by relative error, rather than aborting on whichever
key ``dict`` iteration reached first.

The environment dependencies a run can be missing -- a CUDA device, network, a
trained checkpoint, ffmpeg -- are conditions on the environment, not on
whether an import succeeds, so they are markers (``@pytest.mark.gpu``, ...)
wired to skips in :func:`pytest_collection_modifyitems`.
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

#: Where trained readers are written (``kinescore train``). ``ckpt``-marked
#: tests probe it instead of hardcoding a path; unset means they skip.
CKPT_ROOT_ENV = "KINESCORE_CKPT_DIR"


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
        the reference side, ``got`` the side under test; the ordering only
        decides which is labelled which in the failure message.
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
    without a leading ``golden_``: ``"golden_fk"``, ``"fk"`` and
    ``"golden_fk.npz"`` all resolve to the same file.

    Returns a fully materialised ``dict``, not a lazy ``NpzFile``, so the file
    handle is closed before the test body runs.
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
            f"See MANIFEST.json in {GOLDEN_DIR} for how each was produced."
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


def _ckpt_available() -> bool:
    """``True`` if a trained reader checkpoint is on disk."""
    root = _ckpt_root()
    return root is not None and root.is_dir() and any(root.glob("*.pt"))


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
        reason=f"requires a trained reader checkpoint; set ${CKPT_ROOT_ENV} to a "
               f"directory containing <reader_id>.pt "
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
