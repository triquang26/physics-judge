"""``kinescore doctor``'s engine: environment sanity check.

Lives in :mod:`kinescore.bench` rather than a new top-level
``kinescore.doctor`` module: every check here (torch/CUDA, ffprobe, the
Franka/GR-1 asset hashes, ``$KINESCORE_*`` env vars) exists to answer "is
this environment ready to run the benchmark", the same question
:mod:`kinescore.bench.verify` and :mod:`kinescore.bench.noise_floor` answer
for a manifest/run rather than the interpreter itself -- one more benchmark
diagnostic, not a new architectural layer.

The one hard requirement on this module (see ``kinescore.cli`` and
``kinescore.cli.main``'s docstrings, and the test that pins it,
``tests/test_cli_smoke.py``): it must run to completion and print something
useful on an interpreter that has **none** of torch / pytorch_kinematics /
transformers installed, and it must never touch the network (no triggering a
cold ``robot_descriptions`` fetch just to check whether a URDF is cached).
Every check below is therefore individually wrapped so one missing/broken
dependency degrades its own row instead of aborting the whole report.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["build_report", "render_human", "render_markdown"]


def _check_import(name: str) -> dict:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 -- any import failure is a "not available" row
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(mod, "__version__", None)}


def _torch_info() -> dict:
    info = _check_import("torch")
    if not info["available"]:
        return info
    import torch

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        info["cuda_error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["cuda_available"] = cuda_available
    info["cuda_build"] = getattr(torch.version, "cuda", None)
    info["cuda_device_count"] = torch.cuda.device_count() if cuda_available else 0
    if cuda_available:
        try:
            info["cuda_driver_matches_build"] = True
            info["cuda_device_0"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # noqa: BLE001
            info["cuda_device_0_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _ffprobe_info() -> dict:
    path = shutil.which("ffprobe")
    if not path:
        return {"available": False, "reason": "ffprobe not on PATH"}
    try:
        out = subprocess.run([path, "-version"], capture_output=True, text=True,
                             timeout=5)
        first_line = out.stdout.splitlines()[0] if out.stdout else ""
        return {"available": True, "path": path, "version": first_line}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "path": path,
                "reason": f"{type(exc).__name__}: {exc}"}


def _env_vars() -> dict:
    try:
        from kinescore.paths import ENV_VARS
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {k: os.environ.get(k) or None for k in ENV_VARS}


def _cached_panda_urdf() -> Path | None:
    """Find a cached Panda URDF WITHOUT importing ``robot_descriptions``.

    Mirrors ``tests/conftest.py::_find_cached_panda_urdf`` -- importing
    ``robot_descriptions.panda_description`` itself can trigger a network
    fetch on a cold cache, which ``doctor`` must never do.
    """
    cache_root = Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE",
                                     Path.home() / ".cache" / "robot_descriptions"))
    if not cache_root.is_dir():
        return None
    matches = sorted(cache_root.glob("**/panda_description/urdf/panda.urdf"))
    return matches[0] if matches else None


def _asset_hashes() -> dict:
    out: dict = {}

    urdf = _cached_panda_urdf()
    if urdf is None:
        out["franka_panda"] = {
            "available": False,
            "reason": "no cached Panda URDF (robot_descriptions not yet "
                      "fetched); doctor never triggers a network fetch",
        }
    else:
        try:
            from kinescore.robots.urdf import sha256_file
            out["franka_panda"] = {"available": True, "path": str(urdf),
                                   "sha256": sha256_file(urdf)}
        except Exception as exc:  # noqa: BLE001
            out["franka_panda"] = {"available": False,
                                   "reason": f"{type(exc).__name__}: {exc}"}

    try:
        from kinescore.paths import env_path
        assets = env_path("KINESCORE_ASSETS")
    except Exception as exc:  # noqa: BLE001
        out["fourier_gr1"] = {"available": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            from kinescore.robots.gr1.spec import GR1_URDF_RELPATH
            from kinescore.robots.urdf import sha256_file
            urdf_path = assets / GR1_URDF_RELPATH
            if not urdf_path.is_file():
                out["fourier_gr1"] = {"available": False,
                                      "reason": f"{urdf_path} does not exist",
                                      "assets_root": str(assets)}
            else:
                out["fourier_gr1"] = {"available": True, "path": str(urdf_path),
                                      "sha256": sha256_file(urdf_path)}
        except Exception as exc:  # noqa: BLE001
            out["fourier_gr1"] = {"available": False,
                                  "reason": f"{type(exc).__name__}: {exc}",
                                  "assets_root": str(assets)}
    return out


def build_report() -> dict:
    try:
        from kinescore import __version__ as kinescore_version
    except Exception as exc:  # noqa: BLE001
        kinescore_version = f"unknown ({type(exc).__name__}: {exc})"

    return {
        "kinescore_version": kinescore_version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _torch_info(),
        "pytorch_kinematics": _check_import("pytorch_kinematics"),
        "transformers": _check_import("transformers"),
        "pandas": _check_import("pandas"),
        "scipy": _check_import("scipy"),
        "ffprobe": _ffprobe_info(),
        "env_vars": _env_vars(),
        "asset_hashes": _asset_hashes(),
    }


def render_human(report: dict) -> str:
    lines = [
        f"kinescore {report['kinescore_version']}  "
        f"(python {report['python_version']}, {report['platform']})", "",
    ]
    for name in ("torch", "pytorch_kinematics", "transformers", "pandas",
                "scipy", "ffprobe"):
        info = report[name]
        status = "OK" if info.get("available") else "MISSING"
        detail = info.get("version") or info.get("reason") or ""
        lines.append(f"  [{status:7s}] {name:20s} {detail}")
    lines.append("")
    lines.append("  env vars:")
    for k, v in report["env_vars"].items():
        if k == "error":
            continue
        lines.append(f"    {k:24s} {'(unset)' if v is None else v}")
    lines.append("")
    lines.append("  assets:")
    for robot, info in report["asset_hashes"].items():
        if info.get("available"):
            lines.append(f"    {robot:14s} sha256={info['sha256'][:12]}... {info['path']}")
        else:
            lines.append(f"    {robot:14s} unavailable: {info.get('reason')}")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    torch_info = report["torch"]
    if torch_info.get("available"):
        cuda = ("available" if torch_info.get("cuda_available")
               else "not available") + f" (build {torch_info.get('cuda_build')})"
        torch_row = torch_info.get("version") or "?"
    else:
        cuda = "n/a"
        torch_row = "not installed"
    lines = [
        "| Component | Status |", "|---|---|",
        f"| Python | {report['python_version']} |",
        f"| torch | {torch_row} |",
        f"| CUDA | {cuda} |",
        f"| pytorch_kinematics | {'available' if report['pytorch_kinematics'].get('available') else 'not installed'} |",
        f"| transformers | {'available' if report['transformers'].get('available') else 'not installed'} |",
        f"| ffprobe | {'available' if report['ffprobe'].get('available') else 'not on PATH'} |",
    ]
    return "\n".join(lines)
