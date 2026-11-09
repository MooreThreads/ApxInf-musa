import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_apxinf_musa import load_reference
from compare_outputs import compare_array


class ComparisonTests(unittest.TestCase):
    def test_identical_arrays(self):
        result = compare_array(np.ones((2, 3)), np.ones((2, 3)))
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertEqual(result["relative_l2"], 0.0)

    def test_zero_arrays(self):
        result = compare_array(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertEqual(result["cosine"], 1.0)
        self.assertEqual(result["relative_l2"], 0.0)

    def test_zero_output_fails_nonzero_reference(self):
        result = compare_array(np.zeros((2, 3)), np.ones((2, 3)))
        self.assertEqual(result["cosine"], 0.0)
        self.assertEqual(result["relative_l2"], 1.0)

    def test_nonzero_output_rejects_zero_reference(self):
        with self.assertRaises(ValueError):
            compare_array(np.ones((2, 3)), np.zeros((2, 3)))

    def test_rejects_nonfinite_or_empty(self):
        for values in (np.array([np.nan]), np.array([np.inf]), np.array([])):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    compare_array(values, values)

    def test_rejects_shape_mismatch_with_equal_size(self):
        with self.assertRaises(ValueError):
            compare_array(np.ones((2, 3)), np.ones((3, 2)))

    def test_npz_prefers_normalized_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.npz"
            np.savez(path, normalized_actions=np.ones((2, 3)), actions=np.zeros((2, 3)))
            np.testing.assert_array_equal(load_reference(path), np.ones((2, 3)))

    def test_json_preserves_output_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(json.dumps({"output": {"shape": [2, 3], "values": [1] * 6}}))
            self.assertEqual(load_reference(path).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
