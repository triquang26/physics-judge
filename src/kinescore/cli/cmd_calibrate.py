"""``kinescore calibrate``: post-hoc sigma temperature for a heteroscedastic head.

Wires :mod:`kinescore.training.calibrate` up to a validation cache: runs a
trained :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` over cached
tokens, compares its ``mu``/``logvar`` against the logged real joints, and
fits :func:`~kinescore.training.calibrate.fit_sigma_temperature`.

``ReadoutV2Head`` ships no ``save``/``load`` in this package (see its module
docstring -- only the inference-time forward pass is ``kinescore.heads``'
concern); ``--checkpoint`` here is expected to be a ``{"state_dict", "cfg"}``
file, the same shape ``models.posendf.readout_v2.ReadoutV2Head.save`` wrote
in the source, loaded by the small local helper below.
"""
from __future__ import annotations

import argparse

NAME = "calibrate"
HELP = "fit a post-hoc sigma temperature for a heteroscedastic head"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True,
                        help="ReadoutV2Head checkpoint: torch.save({'state_dict', 'cfg'})")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--down-sample", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-context", action="store_true",
                        help="evaluate with use_context=False (per-frame mode)")
    parser.add_argument("--per-joint", action="store_true",
                        help="fit one temperature per joint instead of one scalar")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True,
                        help="output JSON path for the fitted temperature")


def _load_readout_v2(path: str, device: str):
    import torch

    from kinescore.heads.heteroscedastic import ReadoutV2Head

    ck = torch.load(path, map_location="cpu")
    if "state_dict" not in ck:
        raise ValueError(
            f"{path!r} does not look like a ReadoutV2Head checkpoint "
            f"(missing 'state_dict'); see this module's docstring for the "
            f"expected {{'state_dict', 'cfg'}} shape")
    cfg = dict(ck.get("cfg", {}))
    head = ReadoutV2Head(**cfg)
    head.load_state_dict(ck["state_dict"])
    return head.to(device).eval()


def run(args: argparse.Namespace) -> int:
    import torch

    from kinescore.cli._provenance import provenance_block, write_json
    from kinescore.training.calibrate import calibrate_sigma
    from kinescore.training.datasets import load_split

    head = _load_readout_v2(args.checkpoint, args.device)
    split = load_split(args.cache_root, args.annotation_root, args.split,
                       args.down_sample, limit=args.limit, progress=print)

    if split.n_joints != head.n_out:
        raise ValueError(
            f"validation split's joint width {split.n_joints} != head.n_out="
            f"{head.n_out}; ReadoutV2Head predicts every value in ``mu`` as "
            f"one flat vector (see its module docstring's N_Q17/N_HAND split "
            f"for how a caller may divide it further) so the cached labels "
            f"must match its full output width.")

    mus, logvars, targets = [], [], []
    with torch.no_grad():
        for i in range(0, split.n_frames, args.batch_size):
            feat = split.feats[i:i + args.batch_size].to(args.device).float()
            out = head(feat[:, None], use_context=not args.no_context)
            mus.append(out["mu"][:, 0].cpu())
            logvars.append(out["logvar"][:, 0].cpu())
            targets.append(split.q[i:i + args.batch_size])

    mu = torch.cat(mus, dim=0)
    logvar = torch.cat(logvars, dim=0)
    target = torch.cat(targets, dim=0)

    result = calibrate_sigma(mu, logvar, target, per_joint=args.per_joint)
    temperature = result.temperature.tolist() if result.temperature.ndim else float(result.temperature)

    print(f"[calibrate] fitted temperature (per_joint={args.per_joint}, "
         f"n_samples={result.n_samples}): {temperature}")

    payload = {
        "temperature": temperature, "per_joint": result.per_joint,
        "n_samples": result.n_samples, "checkpoint": args.checkpoint,
        "split": args.split,
        **provenance_block(n_frames=split.n_frames, n_episodes=split.n_episodes),
    }
    write_json(args.out, payload)
    print(f"[calibrate] wrote {args.out}")
    return 0
