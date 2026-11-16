import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_gold.artifact import read_fixture, sha256, write_artifact


class ArtifactTests(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        arrays = {"pixel_values": np.zeros((512, 1536), dtype=np.uint16),
                  "state": np.zeros((1, 1, 132), dtype=np.uint16),
                  "noise": np.zeros((1, 40, 132), dtype=np.uint16),
                  "token_ids": np.arange(145, dtype=np.uint32),
                  "attention_mask": np.ones(145, dtype=np.uint8),
                  "image_grid_thw": np.array([[1, 16, 16]] * 2, dtype=np.uint32),
                  "embodiment_id": np.array([2], dtype=np.int64)}
        debug = {"normalized_actions": np.ones((40, 132), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gold"
            write_artifact(output, arrays, debug, {"test": True})
            restored = read_fixture(output / "inputs.npz")
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], restored[name])
            manifest = json.loads((output / "manifest.json").read_text())
            for name, digest in manifest["files"].items():
                self.assertEqual(digest, sha256(output / name))
            with self.assertRaises(FileExistsError):
                write_artifact(output, arrays, debug, {})

    def test_nonfinite_output_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gold"
            with self.assertRaises(ValueError):
                write_artifact(output, {}, {"normalized_actions": np.full((40, 132), np.nan)}, {})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
