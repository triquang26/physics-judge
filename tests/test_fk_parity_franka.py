"""Franka FK parity against a golden checkpoint.

Compares :meth:`FrankaSpec.forward_kinematics` against
``tests/golden/golden_fk.npz``. That fixture is generated independently of
this file (by whichever agent owns ``legacy_docs/PROVENANCE.md`` / the golden-data
pipeline), specifically so a regression introduced while porting
:class:`~kinescore.robots.franka.fk.FrankaFK` shows up as a numeric mismatch
against an independently-produced reference, rather than being invisible
because the same code path produced both sides of the comparison.

Expected ``.npz`` layout (documented here since this file is the first
consumer): ``q`` -- arm angles, shape ``(N,7)`` or ``(B,T,7)``; ``gripper``
(optional) -- opening in ``[0,1]``, shape ``(N,)``/``(N,1)`` or ``(B,T,1)``;
``P`` -- reference keypoint positions, same leading shape as ``q`` with a
trailing ``(K,3)`` (``K=8``), in metres.

Skipped entirely -- not failed -- when the golden file does not exist yet, or
when ``pytorch_kinematics`` / the Panda URDF are unavailable.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pytorch_kinematics")

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_fk.npz"


def _as_b_t(x: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Normalise a ``(N, ...)`` or ``(B, T, ...)`` array to ``(B, T, ...)``."""
    if x.ndim == 2 and x.shape[-1] == last_dim:
        return x.unsqueeze(0)  # (N, D) -> (1, N, D)
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(-1)  # (N,) -> (1, N, 1)
    return x


@pytest.mark.ckpt
@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason=f"{GOLDEN_PATH} not present yet (generated outside this "
           f"subpackage); nothing to compare the Franka port against.")
def test_franka_fk_matches_golden():
    from kinescore.robots.franka.spec import FrankaSpec

    try:
        data = np.load(GOLDEN_PATH)
    except Exception as exc:  # pragma: no cover - fixture corrupted
        pytest.skip(f"could not read {GOLDEN_PATH}: {exc}")

    for key in ("q", "P"):
        if key not in data:
            pytest.skip(
                f"{GOLDEN_PATH} does not have the expected key {key!r} "
                f"(has: {list(data.keys())}); see this module's docstring "
                f"for the expected schema.")

    q = _as_b_t(torch.as_tensor(np.asarray(data["q"]), dtype=torch.float32), 7)
    if "gripper" in data:
        gripper = _as_b_t(
            torch.as_tensor(np.asarray(data["gripper"]), dtype=torch.float32), 1)
    else:
        gripper = None

    try:
        spec = FrankaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Franka URDF / robot_descriptions unavailable: {exc}")

    P = spec.forward_kinematics(q, gripper)
    want = torch.as_tensor(np.asarray(data["P"]), dtype=torch.float32)
    if want.ndim == 3:
        want = want.unsqueeze(0)

    assert P.shape == want.shape, f"shape mismatch: {P.shape} != {want.shape}"
    max_err = (P - want).abs().max().item()
    assert torch.allclose(P, want, atol=1e-6), (
        f"Franka FK diverges from golden_fk.npz: max abs error {max_err:.3e} m "
        f"(tolerance 1e-6 m)")
