import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_gold.artifact import read_fixture, sha256, write_artifact


def fixture_arrays():
    return {
        "camera_primary": np.zeros((1, 256, 256, 3), dtype=np.uint8),
        "camera_wrist": np.ones((256, 256, 3), dtype=np.uint8),
        "context": np.zeros((1, 128, 4096), dtype=np.uint16),
        "context_mask": np.ones((1, 128), dtype=np.uint8),
        "proprio": np.zeros((1, 8), dtype=np.float32),
    }


class FastWamArtifactTests(unittest.TestCase):
    def test_fixture_requires_batch_one_two_camera_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez_compressed(path, **fixture_arrays())
            arrays = read_fixture(path)
            self.assertEqual(arrays["camera_primary"].shape, (256, 256, 3))
            self.assertEqual(arrays["camera_wrist"].shape, (256, 256, 3))
            self.assertEqual(arrays["context_mask"].dtype, np.bool_)
            bad = fixture_arrays()
            bad["camera_primary"] = np.zeros((2, 256, 256, 3), dtype=np.uint8)
            np.savez_compressed(path, **bad)
            with self.assertRaisesRegex(ValueError, "batch size one"):
                read_fixture(path)

    def test_atomic_roundtrip_and_no_overwrite(self):
        inputs = fixture_arrays()
        inputs.update(
            input_image=np.zeros((1, 3, 224, 448), dtype=np.uint16),
            action_noise=np.zeros((1, 32, 7), dtype=np.uint16),
        )
        debug = {"normalized_actions": np.ones((32, 7), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gold"
            write_artifact(
                output,
                mode="fast",
                inputs=inputs,
                debug=debug,
                provenance={"test": True},
                action_horizon=32,
                action_dim=7,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["mode"], "fast")
            for name, digest in manifest["files"].items():
                self.assertEqual(digest, sha256(output / name))
            restored = read_fixture(output)
            self.assertEqual(restored["context"].shape, (1, 128, 4096))
            with self.assertRaises(FileExistsError):
                write_artifact(
                    output,
                    mode="fast",
                    inputs=inputs,
                    debug=debug,
                    provenance={},
                    action_horizon=32,
                    action_dim=7,
                )

    def test_mode_specific_outputs_are_checked(self):
        inputs = fixture_arrays()
        actions = np.zeros((32, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "returned_video_latents"):
                write_artifact(
                    Path(directory) / "idm",
                    mode="idm",
                    inputs=inputs,
                    debug={"normalized_actions": actions},
                    provenance={},
                    action_horizon=32,
                    action_dim=7,
                )
            with self.assertRaisesRegex(ValueError, "decoded_video"):
                write_artifact(
                    Path(directory) / "joint",
                    mode="joint",
                    inputs=inputs,
                    debug={"normalized_actions": actions},
                    provenance={},
                    action_horizon=32,
                    action_dim=7,
                )

    def test_nonfinite_or_wrong_shape_actions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, actions in enumerate(
                (np.zeros((31, 7), dtype=np.float32), np.full((32, 7), np.nan, dtype=np.float32))
            ):
                output = Path(directory) / str(index)
                with self.assertRaises(ValueError):
                    write_artifact(
                        output,
                        mode="fast",
                        inputs=fixture_arrays(),
                        debug={"normalized_actions": actions},
                        provenance={},
                        action_horizon=32,
                        action_dim=7,
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
