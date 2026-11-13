# 精度比较函数的 CPU 单元测试：验证 cosine/relative L2 及零数组、
# 非有限值、空数组、形状不匹配等边界情况，不验证模型本身的精度。
# 使用人工 NumPy 数组，不加载模型、不调用 benchmark、不使用 GPU。

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


if __name__ == "__main__":
    unittest.main()
