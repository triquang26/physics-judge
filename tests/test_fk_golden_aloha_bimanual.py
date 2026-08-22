"""ALOHA bimanual FK regression freeze against ``golden_fk_aloha_bimanual.npz``.

The fixture was generated from :class:`AlohaSpec` itself (seed 0, random
in-limits batch plus handwritten poses) and frozen, so a later change to
``AlohaFK``/``AlohaSpec`` geometry shows up as a numeric mismatch instead of
passing silently.

Requires ``pytorch_kinematics`` and a resolvable ALOHA URDF; skipped when
either is unavailable, or when the golden file is missing.
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
