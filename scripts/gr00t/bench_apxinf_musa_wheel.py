#!/usr/bin/env python3
"""Benchmark GR00T through the installed MUSA Python wheel, without a subprocess."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from compare_outputs import GATES, compare_array, load_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--execution", choices=("eager", "prefer-graph", "require-graph"),
                        default="require-graph")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--reference", type=Path, help="normalized-action NPZ or benchmark JSON")
    parser.add_argument("--output", type=Path,
                        default=Path("devlocal/gr00t-n1d7/results/gr00t-musa-wheel.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if args.autotune and args.tactics is None:
        parser.error("--autotune requires --tactics")
    return args


def load_fixture(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "apxinf.gr00t-n1.7.preprocessed-fixture.v1":
        raise ValueError("unsupported fixture schema")
    arrays = {}
    for name, dtype in (
        ("pixel_values", "bfloat16"), ("image_grid_thw", "uint32"),
        ("token_ids", "uint32"), ("attention_mask", "uint8"),
        ("state", "bfloat16"), ("noise", "bfloat16"),
    ):
        entry = manifest["tensors"][name]
        if entry["dtype"] != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        storage = {"bfloat16": "<u2", "uint32": "<u4", "uint8": "u1"}[dtype]
        values = np.fromfile(root / entry["file"], dtype=storage).reshape(entry["shape"])
        if dtype == "bfloat16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        arrays[name] = np.ascontiguousarray(values)
    return manifest, arrays


def main() -> None:
    args = parse_args()
    import apxinf_py

    manifest, arrays = load_fixture(args.fixture)
    started = time.perf_counter()
    model = apxinf_py.ModelRunner.load(
        "gr00t", str(args.checkpoint), device=f"musa:{args.device}",
        precision=args.precision, assets={"backbone": str(args.backbone)},
        calibration=str(args.calibration) if args.calibration else None,
        tactics=str(args.tactics) if args.tactics else None,
        autotune=args.autotune, execution=args.execution,
    )
    load_ms = (time.perf_counter() - started) * 1000
    inputs = (
        arrays["pixel_values"], arrays["image_grid_thw"], arrays["token_ids"],
        arrays["attention_mask"], arrays["state"], manifest["embodiment_id"],
        arrays["noise"],
    )
    for _ in range(args.warmup):
        model._infer_preprocessed(*inputs)

    samples = []
    first_output = None
    repeat_max_abs = 0.0
    deterministic = True
    for iteration in range(args.iterations):
        started = time.perf_counter()
        output = model._infer_preprocessed(*inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not np.all(np.isfinite(output)):
            raise ValueError("model output contains non-finite values")
        if first_output is None:
            first_output = output.copy()
        else:
            deterministic = deterministic and np.array_equal(output, first_output)
            repeat_max_abs = max(repeat_max_abs, float(np.max(np.abs(output - first_output))))
        samples.append({"iteration": iteration, "fixture": manifest["fixture"],
                        "token_count": arrays["token_ids"].size, "latency_ms": elapsed_ms})
    execution = model.execution_mode
    allowed = {"eager": {"no_graph"}, "prefer-graph": {"no_graph", "musa-graph"},
               "require-graph": {"musa-graph"}}
    if execution not in allowed[args.execution]:
        raise RuntimeError(f"requested {args.execution}, runtime reported {execution}")
    latencies = np.asarray([sample["latency_ms"] for sample in samples])
    report = {
        "schema": "apxinf.gr00t-n1.7.benchmark.v2",
        "frontend": "python-wheel",
        "extension": apxinf_py.__file__,
        "checkpoint": str(args.checkpoint), "backbone": str(args.backbone),
        "fixture": manifest["fixture"], "device": args.device, "precision": args.precision,
        "requested_execution": args.execution, "execution": execution,
        "timing_boundary": "NumPy/PyO3 input conversion through synchronized model core, action D2H and NumPy output",
        "warmup": args.warmup, "iterations": args.iterations, "load_ms": load_ms,
        "latency_ms": {
            "min": float(latencies.min()), "mean": float(latencies.mean()),
            "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
        "samples_detail": samples, "deterministic": deterministic,
        "repeat_normalized_max_abs": repeat_max_abs,
        "input": {
            "pixel_values": list(arrays["pixel_values"].shape),
            "image_grid_thw": arrays["image_grid_thw"].tolist(),
            "token_count": arrays["token_ids"].size,
            "state": list(arrays["state"].shape),
            "embodiment_id": manifest["embodiment_id"],
        },
        "output": {"shape": list(first_output.shape),
                   "values": first_output.reshape(-1).tolist()},
    }
    passed = deterministic
    if args.reference:
        reference = load_archive(args.reference)
        metrics = compare_array(first_output, reference["normalized_actions"])
        gate = GATES[args.precision]
        parity_passed = (metrics["cosine"] >= gate["minimum_cosine"]
                         and metrics["relative_l2"] <= gate["maximum_relative_l2"])
        if "noise" in reference and not np.array_equal(arrays["noise"], reference["noise"]):
            raise ValueError("fixture and reference noise differ")
        report["parity"] = {"reference": str(args.reference), **metrics, "passed": parity_passed}
        passed = passed and parity_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"latency_ms": report["latency_ms"], "execution": execution,
                      "parity": report.get("parity"), "deterministic": deterministic}, indent=2))
    if not passed:
        raise SystemExit("correctness or repeatability gate failed")


if __name__ == "__main__":
    main()
