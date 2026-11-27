#!/usr/bin/env python3
"""Generate deterministic official Fast-WAM gold for fast, IDM, or joint inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="official Fast-WAM source checkout")
    parser.add_argument("--config", type=Path, required=True, help="model YAML or composed task YAML")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="two-camera .npz or prior artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "idm", "joint"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--num-video-frames", type=int, default=9)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=5.0,
                        help="inference scheduler shift (original released checkpoint: 5.0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--capture-runs", type=int, default=2)
    return parser.parse_args(argv)


def bootstrap_torch(device_spec: str) -> None:
    import torch  # noqa: F401
    if device_spec.split(":", 1)[0].strip().lower() == "musa":
        try:
            import torch_musa  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("device 'musa' requires a matching torch_musa installation") from error


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.action_horizon <= 0 or args.action_dim <= 0:
        raise SystemExit("action horizon and dimension must be positive")
    if args.num_video_frames <= 0 or args.num_video_frames % 4 != 1:
        raise SystemExit("--num-video-frames must be positive and satisfy T % 4 == 1")
    if args.num_inference_steps <= 0:
        raise SystemExit("--num-inference-steps must be positive")
    if args.warmup_runs < 0 or args.capture_runs < 2:
        raise SystemExit("warmup must be non-negative and at least two captures are required")
    if args.camera_height <= 0 or args.camera_width <= 0:
        raise SystemExit("camera dimensions must be positive")
    if args.camera_height % 16 != 0 or (2 * args.camera_width) % 16 != 0:
        raise SystemExit("the concatenated model image height and width must be multiples of 16")
    if args.output.exists():
        raise SystemExit("output already exists; choose a new directory")
    source = args.source.expanduser().resolve()
    if not (source / "src/fastwam/models/wan22/fastwam.py").is_file():
        raise SystemExit("--source must contain src/fastwam/models/wan22/fastwam.py")
    args.source = source
    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    bootstrap_torch(args.device)
    from torch_gold.capture import run

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
