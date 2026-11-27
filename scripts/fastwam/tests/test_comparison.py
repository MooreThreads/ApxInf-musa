from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_apxinf_musa_wheel import normalize_result
from compare_outputs import compare_float, float_gate_failures, repeat_max_abs


class ComparisonTests(unittest.TestCase):
    def test_identical_float_output_passes(self) -> None:
        values = np.ones((32, 7), dtype=np.float32)
        metrics = compare_float(values, values)
        self.assertEqual(float_gate_failures(metrics), [])
        self.assertEqual(metrics["cosine"], 1.0)

    def test_float_shape_nonfinite_and_gate_failures(self) -> None:
        with self.assertRaises(ValueError):
            compare_float(np.ones((2, 3)), np.ones((3, 2)))
        with self.assertRaises(ValueError):
            compare_float(np.array([np.nan]), np.array([np.nan]))
        metrics = compare_float(np.zeros((2, 3)), np.ones((2, 3)))
        self.assertTrue(float_gate_failures(metrics))

    def test_normalizes_supported_wheel_returns(self) -> None:
        actions = np.ones((1, 32, 7), dtype=np.float32)
        fast = normalize_result(actions, "fast")
        self.assertEqual(fast["normalized_actions"].shape, (32, 7))
        idm = normalize_result((actions, np.ones((1,), dtype=np.float32)), "idm")
        self.assertEqual(set(idm), {"normalized_actions"})
        joint = normalize_result(
            {"actions": actions, "decoded_video": np.zeros((1,), dtype=np.uint8)}, "joint"
        )
        self.assertEqual(set(joint), {"normalized_actions"})

    def test_repeatability_detects_action_difference(self) -> None:
        first = {"normalized_actions": np.ones((32, 7), dtype=np.float32)}
        self.assertEqual(repeat_max_abs(first, first), 0.0)
        changed = {"normalized_actions": first["normalized_actions"].copy()}
        changed["normalized_actions"][0, 0] += 0.25
        self.assertEqual(repeat_max_abs(first, changed), 0.25)


if __name__ == "__main__":
    unittest.main()
