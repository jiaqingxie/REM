"""Corrected-chain check for the short non-oracle 2D warp fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.synthetic.run_geometry_stress import evaluate_chain
from experiments.synthetic.run_warp_minibatch_ot import make_target
from rem.geometry import IdentityMobility
from rem.networks import build_mobility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs/warp_minibatch_ot_20260923/main")
    parser.add_argument("--curvature", type=float, default=1.2)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--variants", default="identity,diagonal,full")
    parser.add_argument("--step-size", type=float, default=0.002)
    parser.add_argument("--physical-burn", type=float, default=4.0)
    parser.add_argument("--physical-draw", type=float, default=12.0)
    parser.add_argument("--save-interval", type=float, default=0.01)
    parser.add_argument("--chains", type=int, default=64)
    parser.add_argument("--core-radius", type=float, default=0.65)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--log-bound", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="outputs/warp_minibatch_ot_20260923/sampling.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.checkpoint_dir)
    target = make_target(args.curvature)
    records = []
    for seed in (int(item) for item in args.seeds.split(",")):
        for variant in args.variants.split(","):
            kind = variant.removesuffix("-no-div")
            if kind == "identity":
                model = IdentityMobility()
                checkpoint_hash = None
            else:
                checkpoint = root / f"c{args.curvature:g}_{kind}_seed{seed}.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = build_mobility(
                    kind, (2,), hidden_dim=args.hidden_dim,
                    depth=args.depth, log_bound=args.log_bound,
                )
                model.load_state_dict(saved["state_dict"])
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metrics, _ = evaluate_chain(
                args, target, model, corrected=not variant.endswith("-no-div"),
                dt=args.step_size, seed=seed
            )
            record = {"seed": seed, "variant": variant, "metrics": metrics,
                      "checkpoint_sha256": checkpoint_hash}
            records.append(record)
            print(json.dumps({"seed": seed, "variant": variant,
                              "latent_mmd": metrics["latent_mmd"],
                              "median_first_passage_steps": metrics[
                                  "median_first_passage_steps"],
                              "first_passage_completion_rate": metrics[
                                  "first_passage_completion_rate"]}), flush=True)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"config": vars(args), "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
