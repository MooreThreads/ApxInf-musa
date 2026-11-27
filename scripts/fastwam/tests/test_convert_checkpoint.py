from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_checkpoint import TensorRecord, plan_shards


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], element_size: int = 2):
        self.shape = shape
        self._element_size = element_size

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result

    def element_size(self) -> int:
        return self._element_size


class ConvertCheckpointTests(unittest.TestCase):
    def test_linear_output_shape_is_transposed(self) -> None:
        record = TensorRecord("x.weight", "mot.x.weight", FakeTensor((7, 1024)), True)
        self.assertEqual(record.output_shape, (1024, 7))
        self.assertEqual(record.nbytes, 7 * 1024 * 2)

    def test_shards_are_deterministic_and_do_not_exceed_limit(self) -> None:
        records = [
            TensorRecord(str(index), str(index), FakeTensor((3,)), False)
            for index in range(5)
        ]
        shards = plan_shards(records, maximum_bytes=12)
        self.assertEqual(
            [[item.output_name for item in shard] for shard in shards],
            [["0", "1"], ["2", "3"], ["4"]],
        )
        self.assertTrue(all(sum(item.nbytes for item in shard) <= 12 for shard in shards))

    def test_rejects_tensor_larger_than_shard(self) -> None:
        record = TensorRecord("big", "big", FakeTensor((8,)), False)
        with self.assertRaisesRegex(ValueError, "larger than shard limit"):
            plan_shards([record], maximum_bytes=8)


if __name__ == "__main__":
    unittest.main()
