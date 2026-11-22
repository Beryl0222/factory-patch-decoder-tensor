"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        self.assertEqual(a["kernel"].shape, (2, 1, 3, 3))
        revised = r["transfers"][3]
        self.assertEqual(a[revised["tensor"]].shape[0], revised["read_rows"][1] - revised["read_rows"][0])
        self.assertFalse(np.array_equal(a["a1"], a["a1r"]))
        self.assertEqual(r["transfers"][-2]["missing"], [[6, 7]])
