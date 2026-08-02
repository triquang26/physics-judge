"""ALOHA bimanual FK regression freeze against ``golden_fk_aloha_bimanual.npz``.

Unlike ``golden_fk.npz`` / ``golden_gr1_fk.npz`` (frozen from an independent
LEGACY source repo via ``tools/gen_golden.py``, so a porting bug shows up as a
numeric mismatch against code that never touched ``src/kinescore``), ALOHA
has no such legacy source: it is new code written directly against
``kinescore.core.robot.RobotSpec``, not a port. This fixture is therefore
**self-produced** -- generated once from :class:`AlohaSpec` itself (seed 0,
the same random-in-limits-batch + handwritten-poses pattern
``tools/gen_golden.py::golden_fk`` uses for the Panda) and frozen so a FUTURE
refactor of ``AlohaFK``/``AlohaSpec`` is checked against today's numbers, the
same regression-catching role a legacy-diffed fixture plays for Franka/GR-1,
just without an independent "ground truth" to diff against on day one.

Regenerate (only if the URDF or the FK arithmetic changes intentionally) with
the snippet in this file's own module docstring history / commit message --
there is no ``tools/gen_golden.py`` generator for this fixture (that tool is
scoped to legacy-source diffing, which does not apply here).

Requires ``pytorch_kinematics`` + a resolvable ALOHA URDF; skipped entirely
when either is unavailable, or when the golden file itself is missing.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pytorch_kinematics")

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_fk_aloha_bimanual.npz"


def _aloha_spec():
    from kinescore.robots.aloha.spec import AlohaSpec
    try:
        return AlohaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ALOHA URDF / KINESCORE_ASSETS unavailable: {exc}")


@pytest.mark.skipif(
    not GOLDEN_PATH.exists(),
    reason=f"{GOLDEN_PATH} not present; nothing to compare AlohaSpec against.")
def test_aloha_fk_matches_golden(assert_close_dict):
    data = np.load(GOLDEN_PATH)
    spec = _aloha_spec()

    # URDF identity: if this drifts, every numeric comparison below is
    # comparing FK from a DIFFERENT URDF and the failure would be
    # misleading -- check it first, with its own clear message.
    got_sha = spec.urdf_sha256
    want_sha = str(data["urdf_sha256"])
    assert got_sha == want_sha, (
        f"aloha_bimanual.urdf changed (sha256 {want_sha} -> {got_sha}) -- "
        f"regenerate golden_fk_aloha_bimanual.npz deliberately, don't just "
        f"silence this.")

    q_random = torch.as_tensor(data["q_random"], dtype=torch.float32)
    gripper_values = torch.as_tensor(data["gripper_values"], dtype=torch.float32)

    got = {}
    P_random, R_random = [], []
    for g in gripper_values:
        grip = g.expand(*q_random.shape[:2], 2)
        p, r = spec.forward_transforms(q_random, grip)
        P_random.append(p)
        R_random.append(r)
    got["P_random"] = torch.stack(P_random).numpy()
    got["R_random"] = torch.stack(R_random).numpy()
    got["bone_pairs"] = spec.bone_pairs.numpy()
    got["bone_lengths"] = spec.bone_lengths.numpy()
    got["q_lo"] = spec.q_lo.numpy()
    got["q_hi"] = spec.q_hi.numpy()

    want = {k: data[k] for k in
           ("P_random", "R_random", "bone_pairs", "bone_lengths", "q_lo", "q_hi")}
    assert_close_dict(got, want)
