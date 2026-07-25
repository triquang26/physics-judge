"""Legacy checkpoint compatibility (defect D5 / the ``n_cams`` fallback).

Synthesizes the *exact* key set the real ``judge_v3l`` checkpoint's cfg has
(verified by loading the actual file with
``.venv-marionette/bin/python`` -- 12 keys, no ``n_cams``, no ``hidden``, no
``dropout``) and checks ``readers.checkpoint.load`` still loads it, inferring
``n_cams=1`` and ``hidden=512`` from the weights rather than failing or
guessing wrong.
"""
from __future__ import annotations

import torch

from kinescore.readers import checkpoint

IN_DIM = 1024  # judge_v3l's real embed_dim (DINOv3-L)
HIDDEN = 512   # judge_v3l's real hidden (verified: mlp.0.bias.shape[0]==512)
N_HEADS = 4
N_JOINTS = 7   # judge_v3l's real N_JOINTS (Franka, 7 DOF + gripper => out=8)


def _legacy_head_state_dict() -> dict:
    from kinescore.heads.attentive import AttentivePoseHead

    head = AttentivePoseHead(in_dim=IN_DIM, hidden=HIDDEN, n_joints=N_JOINTS,
                             n_heads=N_HEADS, n_cams=1)  # no cam_emb created
    return head.state_dict()


def _legacy_cfg() -> dict:
    # Exact key set from the real judge_v3l/judge.pt cfg dict (12 keys, no
    # n_cams, no hidden, no dropout, no view_layout_key/robot_name/
    # limit_semantics -- those are new fields this package adds).
    return {
        "dino_model": "dinov3_vitl16",
        "dino_repo_dir": "",
        "embed_dim": IN_DIM,
        "n_kp": 8,
        "fk_ee_link": "panda_link8",
        "pool": "attn",
        "n_heads": N_HEADS,
        "dino_input": 224,
        "patch_pool": 2,
        "hf_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "patch_size": 16,
        "n_register": 4,
    }


def test_legacy_cfg_loads(tmp_path):
    path = str(tmp_path / "judge_v3l_synthetic.pt")
    torch.save({"head": _legacy_head_state_dict(), "cfg": _legacy_cfg(), "meta": {}},
              path)

    loaded = checkpoint.load(path)
    assert loaded.n_cams == 1, "legacy cfg has no n_cams key; must default to 1"
    assert loaded.hidden == HIDDEN, "hidden must be inferred from mlp.0.bias, not defaulted to the ctor default"
    assert loaded.n_heads == N_HEADS
    assert loaded.embed_dim == IN_DIM
    assert loaded.n_joints == N_JOINTS
    assert not hasattr(loaded.head, "cam_emb")


def test_legacy_cfg_missing_semantics_fields_get_documented_defaults(tmp_path):
    path = str(tmp_path / "judge_v3l_synthetic.pt")
    torch.save({"head": _legacy_head_state_dict(), "cfg": _legacy_cfg(), "meta": {}},
              path)
    loaded = checkpoint.load(path)
    # Every real checkpoint at this cfg vintage used the sigmoid squash.
    assert loaded.limit_semantics == "squashed"
    assert loaded.robot_name is None
    assert loaded.view_layout_key is None


def test_legacy_cfg_with_hidden_256_still_infers_correctly(tmp_path):
    """A hypothetical legacy checkpoint trained with a non-default hidden
    width -- the exact case the source could never reload at all."""
    from kinescore.heads.attentive import AttentivePoseHead

    head = AttentivePoseHead(in_dim=128, hidden=256, n_joints=7, n_heads=4, n_cams=1)
    cfg = {
        "dino_model": "dinov2_vitb14", "dino_repo_dir": "", "embed_dim": 128,
        "n_kp": 8, "fk_ee_link": "panda_link8", "pool": "attn", "n_heads": 4,
        "dino_input": 224, "patch_pool": 1, "hf_model_id": "", "patch_size": 14,
        "n_register": 0,
    }
    path = str(tmp_path / "legacy_hidden256.pt")
    torch.save({"head": head.state_dict(), "cfg": cfg, "meta": {}}, path)

    loaded = checkpoint.load(path)
    assert loaded.hidden == 256
    assert torch.equal(loaded.head.state_dict()["mlp.0.weight"],
                       head.state_dict()["mlp.0.weight"])


def test_legacy_multiview_cfg_with_n_cams(tmp_path):
    """judge_v3l_mv-shaped cfg (13 keys, n_cams=3, cam_emb present)."""
    from kinescore.heads.attentive import AttentivePoseHead

    head = AttentivePoseHead(in_dim=IN_DIM, hidden=HIDDEN, n_joints=N_JOINTS,
                             n_heads=N_HEADS, n_cams=3)
    assert hasattr(head, "cam_emb")
    cfg = dict(_legacy_cfg())
    cfg["n_cams"] = 3
    path = str(tmp_path / "judge_v3l_mv_synthetic.pt")
    torch.save({"head": head.state_dict(), "cfg": cfg, "meta": {}}, path)

    loaded = checkpoint.load(path)
    assert loaded.n_cams == 3
    assert hasattr(loaded.head, "cam_emb")
    assert tuple(loaded.head.cam_emb.shape) == (3, IN_DIM)
