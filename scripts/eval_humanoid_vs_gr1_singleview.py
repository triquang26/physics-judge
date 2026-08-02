#!/usr/bin/env python3
# What:   Evaluate the externally-trained `humanoid.pt` (GR1T1 GR00T-Teleop,
#         reported val 19.19mm on ITS OWN domain) against the held-out val
#         split of the new GR1T2 singleview domain
#         ($KINESCORE_CACHE_DIR/fourier_gr1_singleview_tokens/val), and
#         compare it to the freshly trained fourier_gr1_singleview_rawrad.pt
#         on the SAME val split -- the direct comparison
#         scripts/train_singleview_rawrad.sh's report depends on.
# Why:    humanoid.pt currently scores 1168 GR-1 dreamgen/dreamdojo clips, but
#         its 19.19mm was measured on GR1T1 GROOT-Teleop, never on this
#         GR1T2 singleview domain -- whether that number transfers decides
#         whether those already-scored cells rest on a valid reader or need
#         rescoring. humanoid.pt's head predicts n_out=29 (17 FK arm/waist
#         joints + 12 hand DoF); this script slices its mu/logvar to the
#         first 17 dims (kinescore.readers.checkpoint_v2.ReadoutV2PoseReader's
#         own n_fk split) before handing it to
#         kinescore.training.trainer_rawrad.RawRadTrainer.eval_keypoint_mm,
#         which otherwise assumes head.n_out == robot.n_joints.
# Input:  $KINESCORE_CACHE_DIR/fourier_gr1_singleview_tokens/val
#         $KINESCORE_DATA_ROOT/train/fourier_gr1_singleview/annotation/val
#         $KINESCORE_CKPT_DIR/humanoid.pt
#         $KINESCORE_CKPT_DIR/fourier_gr1_singleview_rawrad.pt
# Output: stdout only -- both val keypoint_mm numbers, side by side.
from __future__ import annotations

import os
import sys

import torch


def _fk_slice_wrapper(head: torch.nn.Module, n_fk: int) -> torch.nn.Module:
    """Wrap ``head`` so ``__call__`` returns only the first ``n_fk`` output
    dims of mu/logvar -- matches
    kinescore.readers.checkpoint_v2.ReadoutV2PoseReader's own aux/FK split,
    done here by hand because RawRadTrainer.eval_keypoint_mm calls the head
    directly (not through ReadoutV2PoseReader) and assumes head.n_out ==
    robot.n_joints.
    """

    class _Sliced(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, n: int) -> None:
            super().__init__()
            self.inner = inner
            self.n = n

        def eval(self):
            self.inner.eval()
            return self

        def forward(self, feat: torch.Tensor, use_context: bool = False):
            out = self.inner(feat, use_context=use_context)
            return {"mu": out["mu"][..., : self.n], "logvar": out["logvar"][..., : self.n]}

    return _Sliced(head, n_fk)


def main() -> int:
    from kinescore.core.clip import ViewLayout
    from kinescore.readers import checkpoint_v2
    from kinescore.robots import get_robot
    from kinescore.training.datasets import load_split
    from kinescore.training.trainer_rawrad import RawRadTrainer

    ckpt_dir = os.environ["KINESCORE_CKPT_DIR"]
    cache_root = os.path.join(os.environ["KINESCORE_CACHE_DIR"], "fourier_gr1_singleview_tokens")
    ann_root = os.path.join(os.environ["KINESCORE_DATA_ROOT"], "train", "fourier_gr1_singleview", "annotation")
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"

    robot = get_robot("fourier_gr1", device=device)
    view_layout = ViewLayout(n_views=1, order=())

    print(f"[eval] loading val split from {cache_root!r} ...")
    val_split = load_split(cache_root, ann_root, "val", down_sample=1,
                           expected_view_layout=view_layout, progress=print)
    print(f"[eval] val: {val_split.n_episodes} episodes, {val_split.n_frames} frames, "
         f"n_joints={val_split.n_joints}")

    q_lo, q_hi = robot.q_lo.to(device), robot.q_hi.to(device)

    # ---- humanoid.pt (GR1T1 GROOT-Teleop, external, n_out=29) ----
    humanoid_path = os.path.join(ckpt_dir, "humanoid.pt")
    loaded = checkpoint_v2.load_head(humanoid_path, device=device)
    print(f"[eval] humanoid.pt: n_out={loaded.head.n_out}  robot_name={loaded.robot_name!r}  "
         f"use_context={loaded.use_context}  meta.val_metrics={loaded.meta.get('val_metrics')}")
    wrapped = _fk_slice_wrapper(loaded.head, robot.n_joints).to(device).eval()
    humanoid_eval = RawRadTrainer.eval_keypoint_mm(
        wrapped, robot, val_split.feats, val_split.q, q_lo=q_lo, q_hi=q_hi, device=device)
    print(f"[RESULT] humanoid.pt on fourier_gr1_singleview val: "
         f"keypoint_mm={humanoid_eval['keypoint_mm']:.2f}  rmse_rad={humanoid_eval['rmse_rad']:.4f}")

    # ---- fourier_gr1_singleview_rawrad.pt (this run's own reader, n_out=17) ----
    new_path = os.path.join(ckpt_dir, "fourier_gr1_singleview_rawrad.pt")
    if os.path.exists(new_path):
        loaded_new = checkpoint_v2.load_head(new_path, device=device)
        print(f"[eval] fourier_gr1_singleview_rawrad.pt: n_out={loaded_new.head.n_out}  "
             f"meta.val_keypoint_mm(reported at train time)={loaded_new.meta.get('val_keypoint_mm')}")
        new_eval = RawRadTrainer.eval_keypoint_mm(
            loaded_new.head.to(device).eval(), robot, val_split.feats, val_split.q,
            q_lo=q_lo, q_hi=q_hi, device=device)
        print(f"[RESULT] fourier_gr1_singleview_rawrad.pt on fourier_gr1_singleview val: "
             f"keypoint_mm={new_eval['keypoint_mm']:.2f}  rmse_rad={new_eval['rmse_rad']:.4f}")
    else:
        print(f"[eval] {new_path!r} not found yet -- skipping the new-reader comparison")

    return 0


if __name__ == "__main__":
    sys.exit(main())
