"""Torch-free, bit-preserving fixture reader and verified artifact writer."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import MODEL_NAME, SCHEMA

TYPES = {"bfloat16": "<u2", "uint32": "<u4", "uint8": "u1"}
NAMES = ("pixel_values", "image_grid_thw", "token_ids", "attention_mask", "state", "noise")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fixture(path):
    path = Path(path).resolve()
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in (*NAMES, "embodiment_id")}
    else:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["schema"] != "apxinf.gr00t-n1.7.preprocessed-fixture.v1":
            raise ValueError("unsupported fixture schema")
        arrays = {}
        for name in NAMES:
            entry = manifest["tensors"][name]
            tensor_path = (path / entry["file"]).resolve()
            if not tensor_path.is_relative_to(path):
                raise ValueError("fixture tensor escapes fixture directory")
            arrays[name] = np.fromfile(tensor_path, dtype=TYPES[entry["dtype"]]).reshape(entry["shape"])
        arrays["embodiment_id"] = np.asarray([manifest["embodiment_id"]], dtype=np.int64)
    for name in ("pixel_values", "state", "noise"):
        if arrays[name].dtype != np.dtype("uint16"):
            raise ValueError(f"{name} must contain raw BF16 uint16 bits")
        decoded = (arrays[name].astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(decoded).all():
            raise ValueError(f"{name} contains non-finite BF16 values")
    if arrays["noise"].shape != (1, 40, 132) or arrays["state"].shape != (1, 1, 132):
        raise ValueError("expected LIBERO batch-one state [1,1,132] and noise [1,40,132]")
    if arrays["token_ids"].ndim != 1 or arrays["attention_mask"].shape != arrays["token_ids"].shape:
        raise ValueError("token ids and mask must be equal-length vectors")
    return arrays


def array_metadata(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def write_artifact(output, inputs, debug, provenance):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    normalized = debug["normalized_actions"]
    if normalized.shape != (40, 132) or not np.isfinite(normalized).all():
        raise ValueError("invalid normalized action output")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        metadata = {}
        for filename, arrays in (("inputs.npz", inputs), ("debug.npz", debug)):
            np.savez_compressed(staging / filename, **arrays)
            metadata[filename] = {name: array_metadata(value) for name, value in arrays.items()}
            with np.load(staging / filename, allow_pickle=False) as archive:
                for name, value in arrays.items():
                    if not np.array_equal(value, archive[name]):
                        raise ValueError(f"artifact roundtrip mismatch: {filename}:{name}")
        reference = {"model": MODEL_NAME, "output": {"shape": list(normalized.shape),
                     "values": normalized.reshape(-1).tolist()}}
        (staging / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
        manifest = {"schema": SCHEMA, "model": MODEL_NAME, "provenance": provenance,
                    "arrays": metadata, "encoding": {name: "bfloat16-bits" for name in
                    ("pixel_values", "state", "noise")},
                    "files": {name: sha256(staging / name) for name in
                    ("inputs.npz", "debug.npz", "reference.json")}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name, digest in manifest["files"].items():
            if sha256(staging / name) != digest:
                raise ValueError(f"artifact hash mismatch: {name}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
