#!/usr/bin/env python3
"""Export repository fixtures or compare MUSA runner logits against them."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if not args.export and not args.result:
        parser.error("provide --export or --result")
    with np.load(args.reference, allow_pickle=False) as ref:
        if args.export:
            fixture = {key: ref[key].tolist() for key in ("tokens", "greedy_tokens")}
            if "image_pixel_values" in ref:
                pixels = ref["image_pixel_values"]
                fixture.update(pixel_shape=list(pixels.shape), pixels=pixels.flatten().tolist(),
                               grids=ref["image_grid_thw"].tolist())
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(fixture, allow_nan=False))
        if args.result:
            result = json.loads(args.result.read_text())
            actual = np.asarray(result["logits_last"], dtype=np.float64)
            expected = np.asarray(ref["logits_last"], dtype=np.float64)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise SystemExit("FAIL: logits shape or finite-value check")
            # Match Rust: descending logits, then ascending token ID.
            actual_top = np.argsort(-actual, kind="stable")[:5]
            expected_top = np.argsort(-expected, kind="stable")[:5]
            overlap = len(set(actual_top) & set(expected_top))
            metrics = {
                "cosine": float(np.dot(actual, expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))),
                "max_abs": float(np.max(np.abs(actual-expected))),
                "mean_abs": float(np.mean(np.abs(actual-expected))),
                "argmax_match": bool(actual.argmax() == expected.argmax()),
                "top5_overlap": overlap,
                "greedy_match": result["generated"] == ref["greedy_tokens"].tolist(),
                "prefill_ms": result["prefill_ms"],
                "decode_median_ms": float(np.median(result["decode_ms"])),
            }
            print(json.dumps(metrics, indent=2, allow_nan=False))
            if not (metrics["argmax_match"] and metrics["greedy_match"] and overlap == 5):
                raise SystemExit("FAIL: argmax, top-5 or greedy regression")


if __name__ == "__main__":
    main()
