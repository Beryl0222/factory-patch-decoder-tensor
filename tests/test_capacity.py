"""大尺寸容量口径测试：万级潜变量行下工作区必须有界。

依据 data/capacity.json：
- 输入 16384×107、8 通道、4 组、核 7×5、步长 2、扩张 3，
  max_unsealed_rows=31，peak_scratch_bytes=524288；
- 引擎不得物化整幅输出，任何时刻驻留缓冲只与未封行窗口规模相关。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from streamdeconv import (
    PREVIEW,
    QUANTITATIVE,
    CongestionError,
    DecodeDescriptor,
    DescriptorError,
    ScratchAccount,
    StreamDeconv,
)

DATA = Path(__file__).resolve().parents[1] / "data"


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.cap = json.loads((DATA / "capacity.json").read_text())
        c = self.cap
        # capacity.json 给出算子规模，画幅裁剪框由几何派生：取完整输出范围。
        ih, iw = c["input_hw"]
        kh, kw = c["kernel_hw"]
        sh, sw = c["stride"]
        dh, dw = c["dilation"]
        self.raw_h = (ih - 1) * sh + (kh - 1) * dh + 1
        self.raw_w = (iw - 1) * sw + (kw - 1) * dw + 1
        self.desc = DecodeDescriptor(
            canvas="capacity-coil",
            input_hw=tuple(c["input_hw"]),
            kernel_hw=tuple(c["kernel_hw"]),
            in_channels=c["channels"],
            out_channels=c["channels"],
            groups=c["groups"],
            stride=tuple(c["stride"]),
            dilation=tuple(c["dilation"]),
            offset=(0, 0),
            crop=(0, self.raw_h, 0, self.raw_w),
            expected_owners={"a": (0, ih // 2), "b": (ih // 2, ih)},
            max_unsealed_rows=c["max_unsealed_rows"])
        self.desc.validate()
        rng = np.random.default_rng(c["seed"])
        self.kernel = rng.normal(size=(
            c["channels"], c["channels"] // c["groups"], kh, kw))
        self.rng = rng

    def test_full_length_stream_stays_within_scratch_budget(self):
        """按窗口流喂完整钢卷，峰值工作区不超过 peak_scratch_bytes。"""
        account = ScratchAccount()
        # 零暂存模式：窗口张量是调用方持有的只读输入，不计入接收端工作区，
        # 与 capacity.json 的记账口径一致。
        engine = StreamDeconv(self.desc, self.kernel, account,
                              stage_windows=False)
        ih, iw = self.desc.input_hw
        in_c = self.desc.in_channels
        split = ih // 2
        budget = self.cap["peak_scratch_bytes"]

        # 每个窗口拥 6 行权属、读 8 行；单窗触达输出行
        # (6-1)*2+18+1=29 <= max_unsealed_rows(31)，配合逐窗承诺持续腾槽。
        win = 6
        quant_count = 0

        def drive(source: str, lo: int, hi: int) -> None:
            nonlocal quant_count
            for start in range(lo, hi, win):
                end = min(hi, start + win)
                read_lo = max(0, start - 1)
                read_hi = min(ih, end + 1)
                values = self.rng.normal(
                    size=(read_hi - read_lo, iw, in_c))
                engine.feed_window(
                    {"source": source, "window": f"{source}-{start}",
                     "rev": 1, "owned_rows": [start, end],
                     "read_rows": [read_lo, read_hi]}, values)
                engine.feed_commit(
                    {"source": source, "through": end, "missing": []})
                quant_count += len(engine.drain(QUANTITATIVE))
                engine.drain(PREVIEW)
                self.assertLessEqual(account.peak, budget)

        drive("a", 0, split)
        drive("b", split, ih)

        # 两来源补齐最终承诺，全部输出行完成封行。
        engine.feed_commit({"source": "a", "through": split, "missing": []})
        engine.feed_commit({"source": "b", "through": ih, "missing": []})
        quant_count += len(engine.drain(QUANTITATIVE))
        engine.drain(PREVIEW)
        self.assertEqual(engine.sealed_front, self.raw_h)
        self.assertEqual(quant_count, self.raw_h)
        self.assertLessEqual(account.peak, budget)
        # 已消费封行被压缩：内部列表与未封槽都不随钢卷长度增长。
        self.assertLessEqual(len(engine._sealed),
                             self.cap["max_unsealed_rows"])
        self.assertEqual(engine._slots, {})
        # 窗口元数据只为未交接支持域保留，数量受槽位窗口规模约束。
        active = [w for w in engine._windows if w not in engine._retired]
        self.assertLessEqual(len(active),
                             self.cap["max_unsealed_rows"])
        # 行映射在承诺推进后压缩：全部跑完后只可能剩在途窗口规模的残留，
        # 与 16384 的钢卷总长解耦。
        self.assertLess(len(engine._row_window), 64)
        self.assertGreaterEqual(ih, 10000)
        # 全幅输出若物化需要的字节数远超预算，印证没有整幅驻留。
        full_canvas_bytes = (self.raw_h * self.raw_w
                             * self.desc.out_channels * 8)
        self.assertGreater(full_canvas_bytes, budget * 4)
        self.assertEqual(account.live_bytes, 0)

    def test_congestion_names_blocking_promise_at_scale(self):
        """拥塞诊断在大尺寸下精确指出来源与未决潜变量行。"""
        engine = StreamDeconv(self.desc, self.kernel)
        ih, iw = self.desc.input_hw
        values = self.rng.normal(size=(2, iw, self.desc.in_channels))
        # 用远超槽位上限的大窗口触发拥塞。
        big_owned = (0, self.cap["max_unsealed_rows"] * 4)
        with self.assertRaises(CongestionError) as ctx:
            engine.feed_window(
                {"source": "a", "window": "a-big", "rev": 1,
                 "owned_rows": list(big_owned),
                 "read_rows": [0, big_owned[1]]},
                self.rng.normal(size=(big_owned[1], iw, self.desc.in_channels)))
        blockage = ctx.exception.blockage
        self.assertEqual(blockage["max_unsealed_rows"],
                         self.cap["max_unsealed_rows"])
        self.assertTrue(blockage["blockers"])
        self.assertTrue(all(0 <= b["latent_row"] < ih
                            and b["source"] in ("a", "b")
                            for b in blockage["blockers"]))
        # 被拒窗口不留部分状态。
        self.assertEqual(engine._slots, {})
        self.assertEqual(engine._windows, {})

    def test_descriptor_rejects_owner_gap(self):
        """权属公布不连续的画幅描述在构造期即被拒绝。"""
        from streamdeconv import DescriptorError
        broken = DecodeDescriptor(
            canvas="x",
            input_hw=self.desc.input_hw,
            kernel_hw=self.desc.kernel_hw,
            in_channels=self.desc.in_channels,
            out_channels=self.desc.out_channels,
            groups=self.desc.groups,
            stride=self.desc.stride,
            dilation=self.desc.dilation,
            offset=(0, 0),
            crop=self.desc.crop,
            expected_owners={"a": (0, 4),
                             "b": (5, self.desc.input_hw[0])})
        with self.assertRaises(DescriptorError):
            broken.validate()


if __name__ == "__main__":
    unittest.main()
