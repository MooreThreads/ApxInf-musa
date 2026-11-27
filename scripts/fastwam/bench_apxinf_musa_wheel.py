#!/usr/bin/env python3
"""Benchmark fixed-profile Fast-WAM through an installed ApxInf MUSA wheel."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from compare_outputs import (
    ACTION_GATE,
    compare_float,
    float_gate_failures,
    repeat_max_abs,
)
from formats import exact_input_mismatches, load_fixture, load_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config-json",
        type=Path,
        help="optional ApxInf Fast-WAM architecture JSON; mode is encoded here",
    )
    parser.add_argument("--mode", choices=("fast", "idm", "joint"), required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="inputs.npz or a complete Fast-WAM eager-reference directory",
    )
    parser.add_argument("--reference", type=Path, help="artifact directory, reference JSON, or NPZ")
    parser.add_argument(
        "--allow-unverified-reference-inputs",
        action="store_true",
        help="allow a reference that cannot prove exact fixture/noise identity",
    )
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--execution", choices=("eager", "prefer-graph", "require-graph"), default="eager"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("devlocal/fastwam-musa/results/fastwam-musa-wheel.json"),
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations <= 0:
        parser.error("--warmup must be at least 1 and --iterations must be positive")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    return args


def normalize_result(result: Any, mode: str) -> dict[str, np.ndarray]:
    """Normalize the planned wheel binding's ndarray/tuple/mapping return forms."""
    if isinstance(result, np.ndarray):
        values: dict[str, Any] = {"normalized_actions": result}
    elif isinstance(result, tuple):
        if not result:
            raise ValueError("Fast-WAM tuple output must contain actions")
        # Auxiliary video values are intentionally ignored: the official
        # correctness gate is the complete request's final action only.
        values = {"normalized_actions": result[0]}
    elif isinstance(result, dict):
        values = dict(result)
        for alias in ("actions", "output"):
            if "normalized_actions" not in values and alias in values:
                values["normalized_actions"] = values.pop(alias)
    else:
        raise ValueError(f"unsupported Fast-WAM wheel output type {type(result).__name__}")

    if "normalized_actions" not in values:
        raise ValueError("Fast-WAM wheel output has no normalized_actions")
    normalized = {"normalized_actions": np.ascontiguousarray(values["normalized_actions"])}
    actions = normalized["normalized_actions"]
    if actions.shape == (1, 32, 7):
        actions = actions[0]
    if actions.shape != (32, 7):
        raise ValueError(f"normalized_actions must have shape [32,7], got {actions.shape}")
    if actions.dtype.kind != "f" or not np.isfinite(actions).all():
        raise ValueError("normalized_actions must be finite floating values")
    normalized["normalized_actions"] = actions.astype(np.float32, copy=False)
    return normalized


def _parity_report(
    output: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    action_metrics = compare_float(output["normalized_actions"], reference["normalized_actions"])
    action_failures = float_gate_failures(action_metrics)
    failures.extend(f"actions: {failure}" for failure in action_failures)
    report: dict[str, Any] = {
        "reference": str(args.reference),
        "action_gate": ACTION_GATE,
        "actions": {**action_metrics, "failures": action_failures},
    }
    report["passed"] = not failures
    report["failures"] = failures
    return report, failures


def main() -> None:
    args = parse_args()
    fixture_metadata, arrays = load_fixture(args.fixture, args.mode)
    reference = None
    reference_metadata: dict[str, Any] = {}
    if args.reference is not None:
        reference_metadata, reference = load_reference(args.reference, args.mode)
        if args.reference.is_dir():
            _, reference_inputs = load_fixture(args.reference, args.mode)
            mismatches = exact_input_mismatches(arrays, reference_inputs)
            if mismatches:
                raise ValueError(f"fixture/reference inputs differ: {', '.join(mismatches)}")
        elif not args.allow_unverified_reference_inputs:
            raise ValueError(
                "a JSON/NPZ reference cannot prove image/context/noise identity; use the full "
                "artifact directory or pass --allow-unverified-reference-inputs"
            )

    config_json = args.config_json.read_text() if args.config_json is not None else None
    import apxinf_py

    started = time.perf_counter()
    model = apxinf_py.ModelRunner.load(
        "fastwam",
        str(args.checkpoint),
        device=f"musa:{args.device}",
        precision=args.precision,
        model_variant=args.mode,
        config_json=config_json,
        execution=args.execution,
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    reported_mode = getattr(model, "fastwam_mode", args.mode)
    if reported_mode != args.mode:
        raise RuntimeError(f"requested Fast-WAM {args.mode}, runtime reported {reported_mode}")

    prepare = getattr(model, "prepare_fastwam_preprocessed", None)
    if prepare is None:
        raise RuntimeError(
            "installed apxinf_py wheel has no prepare_fastwam_preprocessed binding; "
            "rebuild the MUSA wheel"
        )
    # One-time host validation, BF16 conversion, and H2D upload. These are
    # deliberately outside both warmup and measured request intervals.
    prepare(
        arrays["input_image"],
        arrays["context"],
        arrays["context_mask"],
        arrays["proprio"],
        arrays["action_noise"],
        video_noise=arrays.get("video_noise"),
    )

    def infer_timed():
        raw_output = model._infer_fastwam_prepared()
        return normalize_result(raw_output, args.mode), float(raw_output["_core_latency_ms"])

    warmup_ms = [infer_timed()[1] for _ in range(args.warmup)]

    samples = []
    first_output: dict[str, np.ndarray] | None = None
    repeat_difference = 0.0
    for iteration in range(args.iterations):
        output, elapsed_ms = infer_timed()
        if first_output is None:
            first_output = {name: value.copy() for name, value in output.items()}
        else:
            repeat_difference = max(repeat_difference, repeat_max_abs(first_output, output))
        samples.append({"iteration": iteration, "latency_ms": elapsed_ms})
    assert first_output is not None

    execution = getattr(model, "execution_mode", "unknown")
    allowed = {
        "eager": {"no_graph"},
        "prefer-graph": {"no_graph", "musa-graph"},
        "require-graph": {"musa-graph"},
    }
    if execution not in allowed[args.execution]:
        raise RuntimeError(f"requested {args.execution}, runtime reported {execution}")
    latencies = np.asarray([sample["latency_ms"] for sample in samples])
    report: dict[str, Any] = {
        "schema": "apxinf.fastwam.benchmark.v2",
        "frontend": "python-wheel",
        "extension": apxinf_py.__file__,
        "model": "fastwam",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "config_json": str(args.config_json) if args.config_json else None,
        "fixture": str(args.fixture),
        "fixture_schema": fixture_metadata.get("schema"),
        "reference_schema": reference_metadata.get("schema"),
        "device": f"musa:{args.device}",
        "visible_devices": os.environ.get("MUSA_VISIBLE_DEVICES"),
        "musa_path": os.environ.get("MUSA_PATH") or os.environ.get("MUSA_HOME"),
        "device_mapping": "device is process-local; physical identity comes from MUSA_VISIBLE_DEVICES",
        "precision": args.precision,
        "requested_execution": args.execution,
        "execution": execution,
        "timing_kind": "rust-reported-synchronized-action-latency",
        "timing_boundary": (
            "device-resident inputs prepared once before warmup; each interval is measured "
            "inside Rust from action-only Fast-WAM eager dispatch through synchronized "
            "device completion and action D2H, excluding the Python/PyO3 call boundary, NumPy materialization, "
            "input validation, F32-to-BF16 conversion, and H2D uploads"
        ),
        "load_ms": load_ms,
        "warmup": {"iterations": args.warmup, "latency_ms": warmup_ms},
        "iterations": args.iterations,
        "latency_ms": {
            "min": float(latencies.min()),
            "mean": float(latencies.mean()),
            "p50": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
        "samples_detail": samples,
        "deterministic": repeat_difference == 0.0,
        "repeat_max_abs": repeat_difference,
        "input": {name: list(value.shape) for name, value in arrays.items()},
        "output": {name: list(value.shape) for name, value in first_output.items()},
    }
    if reference_metadata:
        provenance = reference_metadata.get("provenance", {})
        runtime = provenance.get("runtime", {})
        checkpoint = provenance.get("checkpoint", {})
        config = provenance.get("config", {})
        report["reference_identity"] = {
            "model": reference_metadata.get("model"),
            "mode": reference_metadata.get("mode"),
            "device": runtime.get("device"),
            "device_name": runtime.get("device_name"),
            "precision": runtime.get("precision"),
            "checkpoint_sha256": checkpoint.get("sha256"),
            "config_sha256": config.get("sha256"),
        }
    failures = []
    if repeat_difference != 0.0:
        failures.append(f"fixed-input repeat_max_abs={repeat_difference} is nonzero")
    if reference is not None:
        parity, parity_failures = _parity_report(first_output, reference, args)
        report["parity"] = parity
        failures.extend(parity_failures)
    report["passed"] = not failures
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "execution": execution,
                "latency_ms": report["latency_ms"],
                "deterministic": report["deterministic"],
                "parity": report.get("parity"),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit("correctness or repeatability gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
