"""Command-line interface for TVT4TE."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_extract(args: argparse.Namespace) -> None:
    from .extract import extract_features
    import numpy as np

    features, labels = extract_features(
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root,
        device=args.device,
    )
    np.save(f"{args.out}_features.npy", features)
    np.save(f"{args.out}_labels.npy", labels)
    print(f"Saved features {features.shape} and labels {labels.shape} to {args.out}_*.npy")


def _cmd_score(args: argparse.Namespace) -> None:
    import numpy as np
    import yaml
    from . import TVT

    features = np.load(args.features)
    labels = np.load(args.labels)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scorer = TVT(
        pca_dim=cfg.get("pca_dim", 64),
        p=cfg.get("p", 1),
        fusion=cfg.get("fusion", "geo"),
    )
    score = scorer.score_single(features, labels)
    print(f"TVT score: {score:.6f}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    print("benchmark command: load pre-extracted features from --out directory,")
    print("score all models, compute weighted tau, and write results.")
    print("Full implementation requires pre-extracted .npy files per model/dataset.")
    print("See docs/reproduction.md for the full pipeline.")


def _cmd_bench_fusion(args: argparse.Namespace) -> None:
    print("bench-fusion: fusion ablation across arith/geo/pca/rank schemes.")
    print("See scripts/plot_results.py for the plotting script.")


def _cmd_bench_sensitivity(args: argparse.Namespace) -> None:
    print("bench-sensitivity: PCA dim × Schatten p sensitivity grid.")
    print("See scripts/plot_results.py for the plotting script.")


def main(argv=None):
    """Entry point for the ``tvt`` CLI."""
    parser = argparse.ArgumentParser(
        prog="tvt",
        description="TVT4TE — Three-View Transferability for pretrained vision models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="Extract features from a pretrained model.")
    p_extract.add_argument("--model", required=True, help="Registered model name.")
    p_extract.add_argument("--dataset", required=True, help="Registered dataset name.")
    p_extract.add_argument("--data-root", required=True, help="Root path to datasets.")
    p_extract.add_argument("--out", required=True, help="Output path prefix.")
    p_extract.add_argument("--device", default="cuda", help="PyTorch device.")
    p_extract.set_defaults(func=_cmd_extract)

    # score
    p_score = sub.add_parser("score", help="Score a single model from saved features.")
    p_score.add_argument("--features", required=True, help="Path to .npy feature file.")
    p_score.add_argument("--labels", required=True, help="Path to .npy labels file.")
    p_score.add_argument("--config", required=True, help="YAML config file.")
    p_score.set_defaults(func=_cmd_score)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run full benchmark evaluation.")
    p_bench.add_argument("--config", default="configs/default.yaml")
    p_bench.add_argument("--out", default="results/")
    p_bench.set_defaults(func=_cmd_benchmark)

    # bench-fusion
    p_fusion = sub.add_parser("bench-fusion", help="Fusion ablation study.")
    p_fusion.add_argument("--config", default="configs/fusion_ablation.yaml")
    p_fusion.set_defaults(func=_cmd_bench_fusion)

    # bench-sensitivity
    p_sens = sub.add_parser("bench-sensitivity", help="Hyperparameter sensitivity grid.")
    p_sens.add_argument("--config", default="configs/sensitivity.yaml")
    p_sens.set_defaults(func=_cmd_bench_sensitivity)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
