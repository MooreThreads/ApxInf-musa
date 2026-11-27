import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class QwenTopFiveTest(unittest.TestCase):
    def test_cutoff_tie_prefers_lower_token_id(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/check_qwen3vl_musa.py'
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / 'reference.npz'
            result = Path(tmp) / 'result.json'
            np.savez(ref, logits_last=[9., 8., 7., 6., 5., 4.], greedy_tokens=[0])
            # IDs 4 and 5 tie for the fifth place; only ID 4 is in the reference top five.
            result.write_text(json.dumps(dict(logits_last=[9., 8., 7., 6., 5., 5.],
                                              generated=[0], prefill_ms=1, decode_ms=[1])))
            run = subprocess.run([sys.executable, str(script), str(ref), '--result', str(result)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(json.loads(run.stdout)['top5_overlap'], 5)


if __name__ == '__main__':
    unittest.main()
