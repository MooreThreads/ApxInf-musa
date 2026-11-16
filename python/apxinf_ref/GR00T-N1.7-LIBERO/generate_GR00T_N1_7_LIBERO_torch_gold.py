"""Generate NVIDIA GR00T-N1.7-LIBERO BF16 gold from a fixed preprocessed fixture."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="NVIDIA Isaac-GR00T source checkout")
    parser.add_argument("--input", type=Path, required=True, help="native fixture directory or gold inputs.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--capture-runs", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.capture_runs < 2:
        parser.error("warmup must be non-negative; at least two captures are required")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    source = args.source.resolve()
    if not (source / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py").is_file():
        parser.error("--source must contain the NVIDIA GR00T N1.7 implementation")
    sys.path.insert(0, str(source))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from torch_gold.artifact import read_fixture, write_artifact
    from torch_gold.capture import capture

    inputs = read_fixture(args.input)
    debug, provenance = capture(args, inputs)
    write_artifact(args.output, inputs, debug, provenance)
    print(f"Verified GR00T-N1.7-LIBERO artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
