"""Numerical comparison helpers for Fast-WAM wheel benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np


ACTION_GATE = {"minimum_cosine": 0.999, "maximum_relative_l2": 0.05, "maximum_abs": 0.125}


def compare_float(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: ApxInf {actual.shape}, reference {reference.shape}")
    if actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("comparison inputs must be nonempty and finite")
    actual64 = actual.astype(np.float64).reshape(-1)
    reference64 = reference.astype(np.float64).reshape(-1)
    difference = actual64 - reference64
    reference_norm = np.linalg.norm(reference64)
    denominator = np.linalg.norm(actual64) * reference_norm
    if reference_norm == 0 and np.any(difference):
        raise ValueError("nonzero output does not match a zero reference")
    return {
        "shape": list(actual.shape),
        "cosine": float(np.dot(actual64, reference64) / denominator)
        if denominator
        else (1.0 if not np.any(difference) else 0.0),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm)
        if reference_norm
        else 0.0,
        "max_abs": float(np.max(np.abs(difference))),
    }


def float_gate_failures(metrics: dict[str, Any], gate: dict[str, float] = ACTION_GATE) -> list[str]:
    failures = []
    if metrics["cosine"] < gate["minimum_cosine"]:
        failures.append(f"cosine={metrics['cosine']:.9f} < {gate['minimum_cosine']}")
    if metrics["relative_l2"] > gate["maximum_relative_l2"]:
        failures.append(
            f"relative_l2={metrics['relative_l2']:.9f} > {gate['maximum_relative_l2']}"
        )
    if metrics["max_abs"] > gate["maximum_abs"]:
        failures.append(f"max_abs={metrics['max_abs']:.9f} > {gate['maximum_abs']}")
    return failures


def repeat_max_abs(first: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> float:
    if first.keys() != current.keys():
        raise ValueError("repeated inference returned a different output set")
    maximum = 0.0
    for name in first:
        if first[name].shape != current[name].shape:
            raise ValueError(f"repeated {name} shape changed")
        if not np.isfinite(first[name]).all() or not np.isfinite(current[name]).all():
            raise ValueError(f"{name} contains non-finite values")
        maximum = max(
            maximum,
            float(np.max(np.abs(first[name].astype(np.float64) - current[name].astype(np.float64)))),
        )
    return maximum
