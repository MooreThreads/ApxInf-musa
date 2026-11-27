#!/usr/bin/env python3
"""Convert official Fast-WAM pickle checkpoints to strict SafeTensors assets.

This is an offline, trusted-input tool. Product Rust code never evaluates the
PyTorch pickle. Converted Linear weights use ApxInf's matmul-ready
``[input, output]`` layout; convolution kernels retain PyTorch layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "apxinf.fastwam.converted-checkpoint.v1"
UPSTREAM_REVISION = "7faa71108368fbb3b6885649f112af607427a2d4"
RELEASE_MODEL_SHA256 = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
EXPECTED_MOT_TENSORS = 1649
EXPECTED_VAE_TENSORS = 196
EXPECTED_LAYERS = 30


@dataclass(frozen=True)
class TensorRecord:
    source_name: str
    output_name: str
    tensor: Any
    transpose_linear: bool
    cast_bf16: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.tensor.shape)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.shape[1], self.shape[0]) if self.transpose_linear else self.shape

    @property
    def nbytes(self) -> int:
        element_size = 2 if self.cast_bf16 else int(self.tensor.element_size())
        return int(self.tensor.numel()) * element_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="official libero_uncond_2cam224.pt")
    parser.add_argument("--vae", type=Path, required=True, help="official Wan2.2_VAE.pth")
    parser.add_argument("--output", type=Path, required=True, help="new, empty output directory")
    parser.add_argument(
        "--max-shard-size-gib",
        type=float,
        default=4.0,
        help="maximum tensor payload per SafeTensors shard (default: 4 GiB)",
    )
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="skip hashing large source files (not suitable for release artifacts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the conversion plan without writing tensors",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_shard_size_gib) or args.max_shard_size_gib <= 0:
        parser.error("--max-shard-size-gib must be finite and positive")
    return args


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_tensor_map(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty tensor mapping")
    bad = [
        key
        for key, tensor in value.items()
        if not isinstance(key, str) or not hasattr(tensor, "shape")
    ]
    if bad:
        raise ValueError(f"{name} contains non-tensor entries: {bad[:8]}")
    return value


def _expect_shape(tensors: dict[str, Any], name: str, shape: tuple[int, ...]) -> None:
    if name not in tensors:
        raise ValueError(f"checkpoint is missing {name}")
    actual = tuple(int(value) for value in tensors[name].shape)
    if actual != shape:
        raise ValueError(f"{name} must have shape {shape}, got {actual}")


def validate_model_payload(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Fast-WAM checkpoint root must be a mapping")
    mot = _require_tensor_map(payload.get("mot"), "mot")
    proprio = _require_tensor_map(payload.get("proprio_encoder"), "proprio_encoder")
    if len(mot) != EXPECTED_MOT_TENSORS:
        raise ValueError(f"mot must contain {EXPECTED_MOT_TENSORS} tensors, got {len(mot)}")
    if set(proprio) != {"weight", "bias"}:
        raise ValueError(f"proprio_encoder keys must be weight/bias, got {sorted(proprio)}")
    if any(
        str(tensor.dtype) != "torch.bfloat16"
        for tensor in [*mot.values(), *proprio.values()]
    ):
        raise ValueError("all released Fast-WAM MoT/proprio tensors must be torch.bfloat16")

    for expert in ("video", "action"):
        prefix = f"mixtures.{expert}.blocks."
        layers = {
            int(name[len(prefix) :].split(".", 1)[0])
            for name in mot
            if name.startswith(prefix)
        }
        if layers != set(range(EXPECTED_LAYERS)):
            raise ValueError(
                f"{expert} block indices must be 0..{EXPECTED_LAYERS - 1}, got {sorted(layers)}"
            )

    _expect_shape(mot, "mixtures.video.patch_embedding.weight", (3072, 48, 1, 2, 2))
    _expect_shape(mot, "mixtures.video.blocks.29.ffn.2.weight", (3072, 14336))
    _expect_shape(mot, "mixtures.video.head.head.weight", (192, 3072))
    _expect_shape(mot, "mixtures.action.action_encoder.weight", (1024, 7))
    _expect_shape(mot, "mixtures.action.blocks.29.self_attn.q.weight", (3072, 1024))
    _expect_shape(mot, "mixtures.action.head.weight", (7, 1024))
    _expect_shape(proprio, "weight", (4096, 8))
    _expect_shape(proprio, "bias", (4096,))
    return mot, proprio


def validate_vae_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    vae = _require_tensor_map(payload, "VAE state_dict")
    if len(vae) != EXPECTED_VAE_TENSORS:
        raise ValueError(f"VAE must contain {EXPECTED_VAE_TENSORS} tensors, got {len(vae)}")
    if any(str(tensor.dtype) != "torch.float32" for tensor in vae.values()):
        raise ValueError("all released Wan2.2 VAE tensors must be torch.float32")
    _expect_shape(vae, "encoder.conv1.weight", (160, 12, 3, 3, 3))
    _expect_shape(vae, "encoder.middle.1.to_qkv.weight", (1920, 640, 1, 1))
    if not any(name.startswith("decoder.") for name in vae):
        raise ValueError("VAE checkpoint contains no decoder tensors")
    return vae


def model_records(mot: dict[str, Any], proprio: dict[str, Any]) -> list[TensorRecord]:
    records = []
    for source_name, tensor in mot.items():
        # Strict topology checks make every rank-2 *.weight a Linear matrix.
        # Norms are rank 1, modulations rank 3 and patch embedding rank 5.
        transpose = source_name.endswith(".weight") and len(tensor.shape) == 2
        records.append(TensorRecord(source_name, f"mot.{source_name}", tensor, transpose))
    for source_name, tensor in proprio.items():
        records.append(
            TensorRecord(
                source_name,
                f"proprio_encoder.{source_name}",
                tensor,
                source_name == "weight",
            )
        )
    return sorted(records, key=lambda item: item.output_name)


def vae_records(vae: dict[str, Any]) -> list[TensorRecord]:
    # Matches official WanVideoVAEStateDictConverter.from_civitai.
    return sorted(
        (TensorRecord(name, f"model.{name}", tensor, False, True) for name, tensor in vae.items()),
        key=lambda item: item.output_name,
    )


def plan_shards(records: Iterable[TensorRecord], maximum_bytes: int) -> list[list[TensorRecord]]:
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    shards: list[list[TensorRecord]] = []
    current: list[TensorRecord] = []
    current_bytes = 0
    for record in records:
        if record.nbytes > maximum_bytes:
            raise ValueError(
                f"tensor {record.output_name} is {record.nbytes} bytes, larger than shard limit {maximum_bytes}"
            )
        if current and current_bytes + record.nbytes > maximum_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += record.nbytes
    if current:
        shards.append(current)
    if not shards:
        raise ValueError("cannot shard an empty tensor set")
    return shards


def write_group(
    root: Path,
    records: list[TensorRecord],
    maximum_bytes: int,
    metadata: dict[str, str],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=False)
    shards = plan_shards(records, maximum_bytes)
    weight_map: dict[str, str] = {}
    shard_manifest = []
    for index, shard in enumerate(shards, 1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        temporary = root / f".{filename}.partial-{os.getpid()}"
        output = root / filename
        materialized = {}
        for record in shard:
            tensor = record.tensor.detach().cpu()
            if record.transpose_linear:
                tensor = tensor.transpose(0, 1)
            if record.cast_bf16:
                tensor = tensor.to(dtype=__import__("torch").bfloat16)
            materialized[record.output_name] = tensor.contiguous()
            weight_map[record.output_name] = filename
        save_file(materialized, temporary, metadata=metadata)
        temporary.replace(output)
        shard_manifest.append(
            {
                "file": filename,
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "tensor_count": len(shard),
            }
        )
        del materialized

    index_document = {
        "metadata": {"total_size": sum(record.nbytes for record in records)},
        "weight_map": weight_map,
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index_document, indent=2, sort_keys=True) + "\n"
    )
    return {"tensor_count": len(records), "shards": shard_manifest}


def tensor_manifest(records: list[TensorRecord]) -> list[dict[str, Any]]:
    return [
        {
            "name": record.output_name,
            "source_name": record.source_name,
            "source_shape": list(record.shape),
            "shape": list(record.output_shape),
            "source_dtype": str(record.tensor.dtype).removeprefix("torch."),
            "dtype": "bfloat16" if record.cast_bf16 else str(record.tensor.dtype).removeprefix("torch."),
            "layout": "linear_input_output" if record.transpose_linear else "source_contiguous",
        }
        for record in records
    ]


def load_torch(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve(strict=True)
    vae_path = args.vae.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir() if output.is_dir() else [output]):
        raise ValueError(f"output must not exist or must be empty: {output}")

    model_payload = load_torch(model_path)
    mot, proprio = validate_model_payload(model_payload)
    model = model_records(mot, proprio)
    vae_payload = load_torch(vae_path)
    vae = vae_records(validate_vae_payload(vae_payload))

    maximum_bytes = int(args.max_shard_size_gib * 1024**3)
    plan = {
        "model_tensors": len(model),
        "model_bytes": sum(record.nbytes for record in model),
        "model_shards": len(plan_shards(model, maximum_bytes)),
        "vae_tensors": len(vae),
        "vae_bytes": sum(record.nbytes for record in vae),
        "vae_shards": len(plan_shards(vae, maximum_bytes)),
        "linear_transposes": sum(record.transpose_linear for record in model),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    model_sha = None if args.skip_source_hash else sha256_file(model_path)
    vae_sha = None if args.skip_source_hash else sha256_file(vae_path)
    if model_sha is not None and model_sha != RELEASE_MODEL_SHA256:
        raise ValueError(
            f"official model SHA-256 mismatch: expected {RELEASE_MODEL_SHA256}, got {model_sha}"
        )
    common_metadata = {
        "format": SCHEMA,
        "upstream_revision": UPSTREAM_REVISION,
        "linear_layout": "input_output",
    }
    model_output = write_group(output / "model", model, maximum_bytes, common_metadata)
    vae_output = write_group(output / "vae", vae, maximum_bytes, common_metadata)
    manifest = {
        "schema": SCHEMA,
        "upstream": {
            "repository": "https://github.com/yuantianyuan01/FastWAM",
            "revision": UPSTREAM_REVISION,
        },
        "source": {
            "model": {"path": str(model_path), "sha256": model_sha, "bytes": model_path.stat().st_size},
            "vae": {"path": str(vae_path), "sha256": vae_sha, "bytes": vae_path.stat().st_size},
        },
        "conversion": {
            "trusted_pickle_offline_only": True,
            "linear_weight_layout": "[input,output]",
            "convolution_weight_layout": "[output,input/groups,...kernel]",
            "vae_dtype": "bfloat16 (official loader casts the source float32 VAE to torch_dtype)",
            "plan": plan,
        },
        "assets": {"weights": model_output, "vae": vae_output},
        "tensors": {"weights": tensor_manifest(model), "vae": tensor_manifest(vae)},
    }
    (output / "fastwam-conversion.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(output), **plan}, indent=2))


if __name__ == "__main__":
    main()
