#!/usr/bin/env python3
"""Run the maintained MUSA GR00T Rust fixed-input benchmark and optional parity gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from compare_outputs import compare_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp8", "int8"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--execution",
        choices=("eager", "prefer-graph", "require-graph"),
        default="require-graph",
        help="prepared execution policy (default: require-graph)",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing GEMM tactics and persist them at --tactics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("devlocal/gr00t-n1d7/results/gr00t-musa-bench.json"),
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="prebuilt gr00t_bench_musa binary; otherwise cargo run --release is used",
    )
    parser.add_argument(
        "--reference", type=Path, help="reference .npy, NVIDIA .npz, or JSON output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("--warmup must be non-negative and --iterations must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    executable = (
        [str(args.binary)]
        if args.binary is not None
        else [
            "cargo",
            "run",
            "--release",
            "-p",
            "apxinf-model",
            "--features",
            "musa",
            "--example",
            "gr00t_bench_musa",
            "--",
        ]
    )
    command = [
        *executable,
        str(args.checkpoint),
        str(args.backbone),
        str(args.fixture),
        args.precision,
        str(args.device),
        str(args.warmup),
        str(args.iterations),
        str(args.calibration) if args.calibration is not None else "-",
        str(args.tactics) if args.tactics is not None else "-",
        str(args.output),
        "--execution",
        args.execution,
    ]
    if args.autotune:
        if args.tactics is None:
            raise SystemExit("--autotune requires an explicit --tactics output path")
        command.append("--autotune")
    subprocess.run(command, check=True)
    report = json.loads(args.output.read_text())
    if args.reference is not None:
        actual = np.asarray(report["output"]["values"], dtype=np.float64).reshape(
            report["output"]["shape"]
        )
        reference = load_reference(args.reference).astype(np.float64)
        metrics = compare_array(actual, reference)
        cosine = metrics["cosine"]
        relative_l2 = metrics["relative_l2"]
        parity = {
            "reference": str(args.reference),
            **metrics,
        }
        report["parity"] = parity
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        minimum_cosine, maximum_relative_l2 = {
            "bf16": (0.999, 0.05),
            "fp8": (0.997, 0.10),
            "int8": (0.995, 0.10),
        }[args.precision]
        max_abs = parity["max_abs"]
        maximum_abs = 0.125 if args.precision == "int8" else None
        if (
            cosine < minimum_cosine
            or relative_l2 > maximum_relative_l2
            or (maximum_abs is not None and max_abs > maximum_abs)
        ):
            raise SystemExit(
                "parity gate failed: "
                f"cosine={cosine:.9f} (min {minimum_cosine}), "
                f"relative_l2={relative_l2:.9f} (max {maximum_relative_l2}), "
                f"max_abs={max_abs:.9f}"
                + (f" (max {maximum_abs})" if maximum_abs is not None else "")
            )
    print(json.dumps(report, indent=2))


def load_reference(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            for key in ("normalized_actions", "final_action", "output", "actions"):
                if key in archive:
                    return np.asarray(archive[key])
            raise ValueError(
                f"cannot find normalized_actions, final_action, output, or actions in reference {path}; "
                f"available keys: {archive.files}"
            )
    document = json.loads(path.read_text())
    if isinstance(document, list):
        return np.asarray(document)
    if isinstance(document.get("output"), dict) and "shape" in document["output"]:
        return np.asarray(document["output"]["values"]).reshape(document["output"]["shape"])
    for keys in (("output", "values"), ("output",), ("actions",)):
        value = document
        try:
            for key in keys:
                value = value[key]
        except (KeyError, TypeError):
            continue
        return np.asarray(value)
    raise ValueError(f"cannot find output values in reference {path}")


if __name__ == "__main__":
    main()
