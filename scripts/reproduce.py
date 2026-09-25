"""Run paper protocols from source; no stored results or fitted weights are needed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = {
    "geometry": "run_geometry_stress",
    "highdim": "run_highdim_structured_ot",
    "curved": "run_nonoracle_curved_ot",
    "corrected-highdim": "run_highdim_corrected_sampling",
    "controls": "run_inductive_bias_comparison",
    "pem-control": "run_pem_helmholtz_control",
}


def command(module: str, options: dict) -> list[str]:
    result = [sys.executable, "-m", module]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            result.extend([flag, *map(str, value)])
        else:
            result.extend([flag, str(value)])
    return result


def warp_plan(args, protocol: dict) -> list[list[str]]:
    root = args.output / "warp_ot"
    fit = dict(protocol["warp_ot_train"], device="cpu", threads=args.threads)
    sample = dict(protocol["warp_ot_sample"], threads=args.threads)
    if args.smoke:
        fit.update(train_steps=2, batch_size=16, validation_batches=2,
                   test_batches=2, hidden_dim=8, depth=2, eval_every=1)
        sample.update(hidden_dim=8, depth=2, chains=8, physical_burn=.01,
                      physical_draw=.06, save_interval=.002)
    jobs = []
    for folder, seeds, curvatures, suffix in (
        ("main", "0,1,2", "0,1.2", "main"),
        ("main_extra", "3,4", "1.2", "extra"),
    ):
        if args.stage in ("all", "train"):
            jobs.append(command("experiments.synthetic.run_warp_minibatch_ot",
                                dict(fit, seeds=seeds, curvatures=curvatures,
                                     output=root / folder)))
        if args.stage in ("all", "evaluate"):
            for variants, label in (("identity,diagonal,full", ""), ("full-no-div", "nodiv_")):
                jobs.append(command("experiments.synthetic.evaluate_warp_minibatch_ot",
                                    dict(sample, seeds=seeds, checkpoint_dir=root / folder,
                                         variants=variants, output=root / f"sampling_{label}{suffix}.json")))
    if args.stage in ("all", "evaluate"):
        jobs.append(command("experiments.synthetic.summarize_warp_minibatch_ot", {"root": root}))
    return jobs


def image_plan(args, protocol: dict) -> list[list[str]]:
    dataset = args.suite
    root = args.output / dataset
    train = dict(protocol["image_train"], dataset=dataset,
                 data_root=args.data_root, energy_checkpoint=args.energy_checkpoint,
                 output=root / "train")
    evaluate = dict(protocol["image_evaluate"], **protocol[dataset], dataset=dataset,
                    data_root=args.data_root, output=root / "eval", device=args.device)
    streams = evaluate.pop("sampling_seeds")
    jobs = []
    checkpoints = []
    for fit in range(3):
        name = f"rem_diagonal_seed{fit}"
        filename = "latest.pt" if dataset == "imagenet32" else "checkpoint_150000.pt"
        checkpoints.append(root / "train" / name / "checkpoints" / filename)
        if args.stage in ("all", "train"):
            jobs.append(command("experiments.cifar10.train_rem",
                                dict(train, seed=fit, experiment_id=name)))
    if args.stage in ("all", "evaluate"):
        if dataset == "imagenet32":
            stats = args.fid_real_stats or root / "imagenet32_train_1281167_floor_fid_stats.pt"
            evaluate.update(fid_real_stats=stats, expected_fid_real_samples=1281167)
            if not stats.is_file():
                jobs.append(command("experiments.imagenet.cache_imagenet32_fid_stats",
                                    dict(data_root=args.data_root, output=stats,
                                         image_quantization="floor", expected_samples=1281167,
                                         device=args.device)))
        for stream in streams:
            jobs.append(command("experiments.cifar10.evaluate_rem",
                                dict(evaluate, checkpoint=checkpoints[0], seed=stream,
                                     mobility_training_seed=0, mobility_active_until=0.0,
                                     experiment_id=f"identity_seed{stream}")))
        for fit, checkpoint in enumerate(checkpoints):
            for stream in streams:
                jobs.append(command("experiments.cifar10.evaluate_rem",
                                    dict(evaluate, checkpoint=checkpoint, seed=stream,
                                         mobility_training_seed=fit, mobility_active_until=1.0,
                                         experiment_id=f"rem_fit{fit}_seed{stream}")))
        jobs.append(command("experiments.cifar10.aggregate_reviewer_seed_expansion",
                            dict(roots=[root / "eval"], output=root / "aggregate.json")))
    if args.stage in ("all", "diagnose"):
        jobs.append(command("experiments.cifar10.diagnose_image_mobility",
                            dict(dataset=dataset, checkpoints=checkpoints,
                                 data_root=args.data_root, output=root,
                                 experiment_id="heldout_diagnostics", device=args.device,
                                 seeds="0,1,2", batches_per_seed=64, batch_size=128)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["warp-ot", *SYNTHETIC, "cifar10", "imagenet32"])
    parser.add_argument("--stage", choices=["all", "train", "evaluate", "diagnose"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--energy-checkpoint", type=Path)
    parser.add_argument("--fid-real-stats", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Reduced 2D integration check, not paper results")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.smoke and args.suite != "warp-ot":
        parser.error("--smoke is supported for warp-ot only")
    if args.suite == "warp-ot" and args.stage == "diagnose":
        parser.error("warp-ot supports all, train or evaluate")
    if args.suite in SYNTHETIC and args.stage != "all":
        parser.error("this synthetic entry point runs its full protocol; use --stage all")
    if args.suite in ("cifar10", "imagenet32"):
        if args.data_root is None:
            parser.error("images require --data-root")
        if args.stage in ("all", "train") and args.energy_checkpoint is None:
            parser.error("image training requires --energy-checkpoint")
    # Resolve user paths before moving to the repository root for module imports.
    for key in ("output", "data_root", "energy_checkpoint", "fid_real_stats"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if args.smoke:
        args.output = args.output / "smoke"
    protocol = json.loads((ROOT / "configs/paper_protocol.json").read_text())
    if args.suite == "warp-ot":
        jobs = warp_plan(args, protocol)
    elif args.suite in SYNTHETIC:
        jobs = [command("experiments.synthetic." + SYNTHETIC[args.suite],
                        dict(protocol[args.suite], device=args.device,
                             output=args.output, experiment_id=args.suite))]
    else:
        jobs = image_plan(args, protocol)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.threads)
    for job in jobs:
        print(shlex.join(job), flush=True)
        if not args.dry_run:
            subprocess.run(job, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
