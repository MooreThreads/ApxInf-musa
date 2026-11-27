from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formats import exact_input_mismatches, load_fixture, load_reference


def bf16_bits(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


class FormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def artifact(self, mode: str = "joint") -> Path:
        arrays = {
            "input_image": bf16_bits(np.full((1, 3, 224, 448), 0.5)),
            "context": bf16_bits(np.full((1, 4, 4096), 0.25)),
            "context_mask": np.ones((1, 4), dtype=bool),
            "proprio": bf16_bits(np.zeros((1, 8))),
            "action_noise": bf16_bits(np.full((1, 32, 7), -0.5)),
        }
        if mode in ("idm", "joint"):
            arrays["video_noise"] = bf16_bits(np.full((1, 48, 3, 14, 28), 0.125))
        np.savez(self.root / "inputs.npz", **arrays)
        np.savez(
            self.root / "debug.npz",
            normalized_actions=np.ones((32, 7), dtype=np.float32),
            decoded_video=np.zeros((9, 224, 448, 3), dtype=np.uint8),
        )
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "apxinf.fastwam.torch-eager-gold.v1",
                    "model": "yuantianyuan01/FastWAM",
                    "mode": mode,
                }
            )
        )
        (self.root / "reference.json").write_text(
            json.dumps(
                {
                    "model": "yuantianyuan01/FastWAM",
                    "mode": mode,
                    "output": {"shape": [32, 7], "values": np.ones((32, 7)).reshape(-1).tolist()},
                    "decoded_video": {"shape": [9, 224, 448, 3]},
                }
            )
        )
        return self.root

    def test_artifact_decodes_bf16_and_video(self) -> None:
        root = self.artifact()
        metadata, fixture = load_fixture(root, "joint")
        self.assertEqual(metadata["mode"], "joint")
        self.assertEqual(fixture["input_image"].dtype, np.float32)
        np.testing.assert_array_equal(fixture["input_image"], 0.5)
        _, reference = load_reference(root, "joint")
        self.assertEqual(reference["normalized_actions"].shape, (32, 7))
        self.assertEqual(reference["decoded_video"].shape, (9, 224, 448, 3))

    def test_mode_shape_and_nonfinite_rejections(self) -> None:
        root = self.artifact("fast")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_fixture(root, "joint")
        with np.load(root / "inputs.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["input_image"] = np.zeros((1, 3, 1, 1), dtype=np.float32)
        np.savez(root / "inputs.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "input_image must have shape"):
            load_fixture(root, "fast")

    def test_exact_input_mismatch_names_noise(self) -> None:
        root = self.artifact("fast")
        _, first = load_fixture(root, "fast")
        second = {name: value.copy() for name, value in first.items()}
        second["action_noise"][0, 0, 0] += 1
        self.assertEqual(exact_input_mismatches(first, second), ["action_noise"])

    def test_reference_json_rejects_bad_values(self) -> None:
        path = self.root / "reference.json"
        path.write_text(json.dumps({"output": {"shape": [32, 7], "values": [1]}}))
        with self.assertRaisesRegex(ValueError, "match output.shape"):
            load_reference(path, "fast")


if __name__ == "__main__":
    unittest.main()
