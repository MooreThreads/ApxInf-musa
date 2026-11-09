#!/usr/bin/env python3
"""Compare fixed-input ApxInf GR00T outputs against an explicit reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


GATES = {
    "bf16": {"minimum_cosine": 0.999, "maximum_relative_l2": 0.05},
    "fp8": {"minimum_cosine": 0.997, "maximum_relative_l2": 0.10},
    "int8": {"minimum_cosine": 0.995, "maximum_relative_l2": 0.10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apxinf", type=Path, required=True)
    parser.add_argument("--reference", "--torch-eager", dest="reference", type=Path, required=True)
    parser.add_argument("--precision", choices=tuple(GATES), default="bf16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_archive(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"output archive does not exist: {path}")
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        try:
            values = np.asarray(document["output"]["values"], dtype=np.float32)
            shape = tuple(int(value) for value in document["output"]["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} is missing Rust benchmark output values") from error
        if values.size != int(np.prod(shape)):
            raise ValueError(
                f"{path} output values have {values.size} elements, expected {shape}"
            )
        return {"normalized_actions": values.reshape(shape)}
    with np.load(path) as archive:
        required = ("normalized_actions",)
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        return {
            key: np.asarray(archive[key])
            for key in ("normalized_actions", "actions", "noise")
            if key in archive
        }


def compare_array(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: ApxInf {actual.shape}, reference {reference.shape}"
        )
    if actual.size == 0:
        raise ValueError("comparison inputs must not be empty")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError("comparison inputs must contain only finite values")

    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    difference = actual64 - reference64
    denominator = np.linalg.norm(actual64) * np.linalg.norm(reference64)
    reference_norm = np.linalg.norm(reference64)
    if reference_norm == 0 and np.any(difference):
        raise ValueError("nonzero output does not match a zero reference")
    if not np.isfinite(denominator) or not np.isfinite(reference_norm):
        raise ValueError("comparison norms must be finite")
    return {
        "shape": list(actual.shape),
        "cosine": float(np.dot(actual64, reference64) / denominator)
        if denominator
        else (1.0 if not np.any(difference) else 0.0),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm)
        if reference_norm
        else 0.0,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
    }


def main() -> int:
    args = parse_args()
    apxinf = load_archive(args.apxinf)
    reference = load_archive(args.reference)
    if "noise" in apxinf and "noise" in reference and not np.array_equal(
        apxinf["noise"], reference["noise"]
    ):
        raise SystemExit("correctness gate failed: ApxInf and reference noise differ")

    gate = GATES[args.precision]
    keys = ["normalized_actions"]
    if "actions" in apxinf and "actions" in reference:
        keys.append("actions")
    comparisons = {key: compare_array(apxinf[key], reference[key]) for key in keys}
    failures = []
    for key, result in comparisons.items():
        if result["cosine"] < gate["minimum_cosine"]:
            failures.append(
                f"{key} cosine={result['cosine']:.9f} < {gate['minimum_cosine']}"
            )
        if result["relative_l2"] > gate["maximum_relative_l2"]:
            failures.append(
                f"{key} relative_l2={result['relative_l2']:.9f} "
                f"> {gate['maximum_relative_l2']}"
            )

    report = {
        "passed": not failures,
        "precision": args.precision,
        "apxinf": str(args.apxinf.expanduser().resolve()),
        "reference": str(args.reference.expanduser().resolve()),
        "noise_checked": "noise" in apxinf and "noise" in reference,
        "noise_exact_match": (
            np.array_equal(apxinf["noise"], reference["noise"])
            if "noise" in apxinf and "noise" in reference
            else None
        ),
        "gate": gate,
        "comparisons": comparisons,
        "failures": failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures:
        raise SystemExit("correctness gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
