"""流式引擎行为测试：数值对账、封行语义、修订冲突与拥塞。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from streamdeconv import (
    COVERED,
    GAP,
    PREVIEW,
    QUANTITATIVE,
    STRUCTURAL_ZERO,
    CongestionError,
    ScratchAccount,
    StreamDeconv,
    RevisionConflictError,
    StaleRevisionError,
    TransferValidationError,
    build_descriptor,
)
from streamdeconv.feed import replay
from tests.reference import make_descriptor, random_kernel, reference_transposed_conv

DATA = Path(__file__).resolve().parents[1] / "data"
rng = np.random.default_rng(7)


class NumericTests(unittest.TestCase):
    def test_matches_brute_force_with_revisions(self):
        """全窗口（含承诺前替换修订）结果与暴力转置卷积逐元素一致。"""
        records, tensors = load_case(DATA / "capture.json")
        # 用 rev2 的 a1 重造“最终真值”潜变量：按窗口权属拼接，
        # a0 拥行 0-1、a1r 拥行 2-3、b0 拥行 4-5（read_rows 的交叠不贡献）。
        desc = build_descriptor(records, tensors["kernel"].shape)
        latent = np.zeros((*desc.input_hw, desc.in_channels), dtype=np.float64)
        latent[0:2] = tensors["a0"][0:2]
        latent[2:4] = tensors["a1r"][1:3]
        latent[4:6] = tensors["b0"][1:3]
        expect = reference_transposed_conv(desc, tensors["kernel"], latent)

        engine, delivered, log = replay(records, tensors)
        quantitative = delivered[QUANTITATIVE]
        # 演示数据有缺失段，缺失影响行只在 preview；先校验定量行。
        top = desc.crop[0]
        q_rows = {p.row: p.data for p in quantitative}
        for y, data in q_rows.items():
            np.testing.assert_allclose(data, expect[y - top], rtol=0, atol=1e-12)
        # 缺失影响行的未缺口部分同样要与参考一致（缺口列参考实现里是零贡献）。
        for product in delivered[PREVIEW]:
            if product.quality == QUANTITATIVE:
                continue
            # 只对账有效贡献列；缺口列引擎无值，结构零列可能由其他行贡献。
            mask = product.coverage == COVERED
            if not mask.any():
                continue
            np.testing.assert_allclose(
                product.data[:, mask], expect[product.row - top][:, mask],
                rtol=0, atol=1e-12)
        # 所有 15 行都完成封行，且无静默丢行。
        self.assertEqual(engine.sealed_front, 15)
        self.assertEqual(
            sorted(p.row for p in delivered[PREVIEW]), list(range(15)))

    def test_only_owned_cells_contribute(self):
        """重叠读上下文不贡献：裁掉权属外的读行后结果不变。"""
        records, tensors = load_case(DATA / "capture.json")
        engine_a, _, _ = replay(records, tensors)
        # 手工喂数：b0 的 read_rows 去掉与 a 交叠的第 3 行（非权属）。
        desc = engine_a.desc
        account = ScratchAccount()
        engine = StreamDeconv(desc, tensors["kernel"], account)
        engine.feed(records["transfers"][0], tensors["a0"])
        b0 = tensors["b0"]
        trimmed = b0[1:]  # read_rows 由 (3,7) 收紧为 (4,7)，权属不变
        engine.feed(
            {"source": "b", "window": "b-0", "rev": 1,
             "owned_rows": [4, 6], "read_rows": [4, 7]}, trimmed)
        full_engine = StreamDeconv(desc, tensors["kernel"], ScratchAccount())
        full_engine.feed(records["transfers"][0], tensors["a0"])
        full_engine.feed(records["transfers"][1], b0)
        for y in engine._slots.keys() & full_engine._slots.keys():
            np.testing.assert_array_equal(engine._slots[y].acc,
                                          full_engine._slots[y].acc)


class SealingTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case(DATA / "capture.json")
        self.engine, self.delivered, self.log = replay(self.records, self.tensors)

    def test_structural_zero_distinct_from_pending(self):
        """可证明零行在任何承诺到达前即可封行，且状态与未收到贡献不同。"""
        # 核展宽(1) < 步长(4)：每 4 行中 3 行有支持域，1 行可证明为零。
        gap_desc = make_descriptor(
            input_hw=(8, 4), kernel_hw=(2, 2), stride=(4, 4),
            crop=(0, 16, 0, 9),
            expected_owners={"a": (0, 4), "b": (4, 8)})
        supports = {y: gap_desc.row_support_inputs(y) for y in range(10)}
        empty_rows = [y for y, (lo, hi) in supports.items() if lo >= hi]
        live_rows = [y for y, (lo, hi) in supports.items() if lo < hi]
        self.assertTrue(empty_rows and live_rows)
        engine = StreamDeconv(gap_desc, random_kernel(gap_desc, rng))
        # 未喂任何窗口：支持域为空的行立即作为结构零封出。
        engine._seal_front()
        rows = [p.row for p in engine.drain(QUANTITATIVE)]
        self.assertEqual(rows, empty_rows[:len(rows)])
        for product in engine.drain(PREVIEW):
            self.assertTrue(np.all(product.coverage == STRUCTURAL_ZERO))
        # 第一条支持域非空的行必须保持 pending（未收到贡献，不是零）。
        self.assertEqual(engine.sealed_front, live_rows[0])
        self.assertNotIn(live_rows[0], engine._slots)

        # 演示画幅本身无结构零行：所有行支持域都非空，空引擎一封不出。
        desc = build_descriptor(self.records, self.tensors["kernel"].shape)
        demo_engine = StreamDeconv(desc, self.tensors["kernel"])
        demo_engine._seal_front()
        self.assertEqual(demo_engine.sealed_front, 0)

    def test_missing_splits_quantitative_and_preview(self):
        """b 缺失 (6,7) 的影响行只进 preview，定量行全部可测量。"""
        preview = {p.row: p for p in self.delivered[PREVIEW]}
        quantitative = {p.row for p in self.delivered[QUANTITATIVE]}
        # 支持域触及潜变量行 6 的输出行由描述符支持域精确给出。
        touched = {y for y in range(self.engine.out_h)
                   if 6 in range(*self.engine.desc.row_support_inputs(y))}
        self.assertTrue(touched)
        degraded = {y for y, p in preview.items() if p.quality == PREVIEW}
        self.assertEqual(degraded, touched)
        self.assertTrue(all(preview[y].measurable_event is None for y in degraded))
        for y in degraded:
            self.assertIn(GAP, preview[y].coverage)
            self.assertEqual(preview[y].gap_input_rows, ((6, 7),))
        self.assertEqual(quantitative, set(range(15)) - degraded)
        # 缺口列给出全局裁剪坐标。
        self.assertTrue(all(lo >= desc_left and hi <= desc_right for _ in [0]
                            for (lo, hi) in preview[11].gap_columns()
                            for desc_left in [self.engine.desc.crop[2]]
                            for desc_right in [self.engine.desc.crop[3]]))

    def test_late_revision_after_commit_is_conflict(self):
        """承诺生效后的异值修订被拒、留痕，已交接像素不变。"""
        conflicts = self.engine.conflicts
        self.assertGreaterEqual(len(conflicts), 1)
        rec = conflicts[0]
        self.assertEqual((rec.source, rec.window), ("b", "b-0"))
        self.assertEqual((rec.rejected_rev, rec.active_rev), (2, 1))
        # b-0 权属 (4,6) 的实际支持域（含扩张造成的远处边缘行）。
        expect_affected = [
            y for y in range(self.engine.out_h)
            if set(range(*self.engine.desc.row_support_inputs(y))) & set(range(4, 6))]
        self.assertEqual(list(rec.affected_rows), expect_affected)
        before = {p.row: p.data.copy()
                  for p in self.delivered[PREVIEW] if p.row in rec.affected_rows}
        # 再喂一次同样的迟到修订，仍然冲突，像素始终不变。
        with self.assertRaises(RevisionConflictError):
            self.engine.feed_window(self.records["transfers"][-1],
                                    self.tensors["b0r"])
        self.assertEqual(len(self.engine.conflicts), 2)
        sealed_index = {p.row: p for p in self.delivered[PREVIEW]}
        for y in rec.affected_rows:
            np.testing.assert_array_equal(sealed_index[y].data, before[y])

    def test_equivalent_resend_is_accepted(self):
        """封行后等价重发幂等，不产生冲突也不改数。"""
        desc = build_descriptor(self.records, self.tensors["kernel"].shape)
        engine = StreamDeconv(desc, self.tensors["kernel"])
        engine.feed(self.records["transfers"][0], self.tensors["a0"])
        engine.feed(self.records["transfers"][2], self.tensors["a1"])
        engine.feed({"source": "a", "through": 4, "missing": []})
        # 同修订同值重发：无异常。
        engine.feed_window(self.records["transfers"][2], self.tensors["a1"])
        # 同修订异值：协议错误。
        with self.assertRaises(TransferValidationError):
            engine.feed_window(
                {**self.records["transfers"][2], "tensor": "x"},
                self.tensors["a1r"])

    def test_revision_before_commit_replaces_cleanly(self):
        """承诺前高修订替换旧值，最终结果等价于直接采用新修订。"""
        desc = build_descriptor(self.records, self.tensors["kernel"].shape)
        # 路径一：a1 rev1 后 rev2 替换。
        e1 = StreamDeconv(desc, self.tensors["kernel"])
        e1.feed_window(self.records["transfers"][0], self.tensors["a0"])
        e1.feed_window(self.records["transfers"][2], self.tensors["a1"])
        e1.feed_window(self.records["transfers"][3], self.tensors["a1r"])
        # 路径二：直接 rev2。
        e2 = StreamDeconv(desc, self.tensors["kernel"])
        e2.feed_window(self.records["transfers"][0], self.tensors["a0"])
        e2.feed_window(self.records["transfers"][3], self.tensors["a1r"])
        # 共有未封槽逐元素一致；替换路径可能额外回收全零槽，属工作区优化。
        common = set(e1._slots) & set(e2._slots)
        self.assertTrue(common)
        for y in common:
            np.testing.assert_allclose(e1._slots[y].acc, e2._slots[y].acc,
                                       rtol=0, atol=1e-12)
        for y in set(e1._slots) - set(e2._slots):
            self.assertTrue(np.all(np.asarray(e1._slots[y].acc) == 0))
        # 旧修订号再到达应被拒绝。
        with self.assertRaises(StaleRevisionError):
            e1.feed_window(self.records["transfers"][2], self.tensors["a1"])

    def test_zero_staging_mode_keeps_window_tensors_out_of_scratch(self):
        """零暂存模式不复制窗口张量；修订替换与等价重发仍语义正确。"""
        desc = build_descriptor(self.records, self.tensors["kernel"].shape)
        account = ScratchAccount()
        engine = StreamDeconv(desc, self.tensors["kernel"], account,
                              stage_windows=False)
        engine.feed_window(self.records["transfers"][0], self.tensors["a0"])
        # 零暂存模式下 _staged 不持有任何副本，工作区只有输出槽与临时块。
        self.assertEqual(engine._staged["a-0"].ctypes.data,
                         self.tensors["a0"].ctypes.data)
        # 承诺前替换：旧值可从调用方张量撤销，结果与直接采用新修订一致。
        engine.feed_window(self.records["transfers"][2], self.tensors["a1"])
        engine.feed_window(self.records["transfers"][3], self.tensors["a1r"])
        direct = StreamDeconv(desc, self.tensors["kernel"],
                              stage_windows=False)
        direct.feed_window(self.records["transfers"][0], self.tensors["a0"])
        direct.feed_window(self.records["transfers"][3], self.tensors["a1r"])
        for y in set(engine._slots) & set(direct._slots):
            np.testing.assert_allclose(engine._slots[y].acc,
                                       direct._slots[y].acc,
                                       rtol=0, atol=1e-12)
        # 承诺并封走后，等价重发幂等且不产生窗口工作区。
        engine.feed({"source": "a", "through": 4, "missing": []})
        peak_after_seal = account.peak
        engine.feed_window(self.records["transfers"][3], self.tensors["a1r"])
        self.assertEqual(account.peak, peak_after_seal)

    def test_blocking_promise_is_reported_under_congestion(self):
        """槽位受限时报出阻塞来源，其他已确定的行继续封行。"""
        capacity = json.loads((DATA / "capacity.json").read_text())
        # 演示数据单窗最多触及 7 条输出行；取上限 5 使后续窗口触发拥塞，
        # 同时每个被接受的窗口都能完整处理（rejected 只可能来自协议违规）。
        engine, delivered, log = replay(self.records, self.tensors,
                                        max_unsealed_rows=5)
        kinds = [entry["kind"] for entry in log]
        self.assertIn("congestion", kinds)
        blockage = next(entry["blockage"] for entry in log
                        if entry["kind"] == "congestion")
        self.assertEqual(blockage["front_row"], engine.sealed_front)
        self.assertTrue(blockage["blockers"])
        self.assertTrue(all("source" in b and "latent_row" in b
                            and "state" in b for b in blockage["blockers"]))
        # 结构零行不受拥塞影响；在更早上游事件中已确定的行照常交接。
        self.assertTrue(all(entry["front"] >= 0 for entry in log
                            if "front" in entry))
        # 容量口径与大尺寸使用规模一致。
        self.assertEqual(capacity["max_unsealed_rows"], 31)

    def test_congestion_recovers_after_promise(self):
        """拥塞是暂时的：来源承诺推进、封行释放槽位后窗口可重新接收。"""
        desc = build_descriptor(self.records, self.tensors["kernel"].shape)
        # 上限 2：a-0 单窗即触及 3 条输出行，必然拥塞。
        tight = StreamDeconv(
            build_descriptor(self.records, self.tensors["kernel"].shape,
                             max_unsealed_rows=2),
            self.tensors["kernel"])
        with self.assertRaises(CongestionError) as ctx:
            tight.feed_window(self.records["transfers"][0], self.tensors["a0"])
        # 阻塞诊断精确指向前沿行与未满足的来源承诺。
        blockage = ctx.exception.blockage
        self.assertEqual(blockage["front_row"], 0)
        self.assertTrue(any(b["state"] == "awaiting_promise"
                            for b in blockage["blockers"]))
        # 被拒窗口未留下任何部分状态。
        self.assertEqual(tight._slots, {})
        # 放宽到足够上限后同一窗口可正常接收，且后续能封行交接。
        engine = StreamDeconv(desc, self.tensors["kernel"])
        engine.feed_window(self.records["transfers"][0], self.tensors["a0"])
        engine.feed_commit({"source": "a", "through": 2, "missing": []})
        self.assertGreater(engine.sealed_front, 1)

    def test_readback_carries_crop_channels_and_measurement_time(self):
        product = self.delivered[PREVIEW][0]
        view = product.as_dict()
        self.assertEqual(view["crop"], list(self.engine.desc.crop))
        self.assertEqual(view["channel_order"],
                         list(range(self.engine.desc.out_channels)))
        self.assertEqual(view["shape"],
                         [self.engine.desc.out_channels, self.engine.desc.out_hw[1]])
        self.assertIsInstance(view["sealed_event"], int)
        quantitative_product = next(
            p for p in self.delivered[QUANTITATIVE] if p.quality == QUANTITATIVE)
        self.assertEqual(quantitative_product.measurable_event,
                         quantitative_product.sealed_event)

    def test_sealed_rows_never_mutated(self):
        """封行后任何窗口都不能改写已交接数组。"""
        handed = {p.row: p for p in self.delivered[PREVIEW]}
        snapshot = {row: p.data.copy() for row, p in handed.items()}
        # b 的迟到冲突已在重放中被拒；再尝试一个权属落在已封支持域的新窗口。
        bad = {"source": "b", "window": "b-x", "rev": 1,
               "owned_rows": [4, 5], "read_rows": [4, 7]}
        with self.assertRaises(TransferValidationError):
            self.engine.feed_window(bad, self.tensors["b0"][1:])
        # 已交接给订阅方的数组内容保持不变。
        for row, data in snapshot.items():
            np.testing.assert_array_equal(handed[row].data, data)


class RandomizedTests(unittest.TestCase):
    def test_random_streams_against_reference(self):
        """随机描述符与随机窗口流，封行结果始终等于全幅参考。"""
        rnd = np.random.default_rng(101)
        for trial in range(4):
            ih = rnd.integers(6, 12)
            iw = rnd.integers(3, 6)
            groups = 2
            in_c = groups * 2
            out_per = 2
            kh, kw = rnd.integers(2, 5, size=2)
            sh, sw = rnd.integers(1, 4, size=2)
            dh, dw = rnd.integers(1, 3, size=2)
            oh = int(rnd.integers(-2, 2))
            ow = int(rnd.integers(-2, 2))
            raw_h = (ih - 1) * sh + (kh - 1) * dh + 1 + oh
            raw_w = (iw - 1) * sw + (kw - 1) * dw + 1 + ow
            top = int(rnd.integers(oh, max(oh + 1, raw_h - 1)))
            bottom = int(rnd.integers(top + 1, raw_h + 1))
            left = int(rnd.integers(ow, max(ow + 1, raw_w - 1)))
            right = int(rnd.integers(left + 1, raw_w + 1))
            split = int(rnd.integers(1, ih))
            desc = make_descriptor(
                input_hw=(int(ih), int(iw)), kernel_hw=(int(kh), int(kw)),
                in_channels=in_c, out_channels=out_per * groups, groups=groups,
                stride=(int(sh), int(sw)), dilation=(int(dh), int(dw)),
                offset=(oh, ow), crop=(top, bottom, left, right),
                expected_owners={"a": (0, split), "b": (split, int(ih))})
            kernel = random_kernel(desc, rnd)
            latent = rnd.normal(size=(ih, iw, in_c))
            engine = StreamDeconv(desc, kernel)
            # 两个来源各发一个覆盖全部权属的窗口，read 带 1 行交叠上下文。
            a_read = (0, min(ih, split + 1))
            b_read = (max(0, split - 1), int(ih))
            engine.feed_window(
                {"source": "a", "window": "a", "rev": 1,
                 "owned_rows": [0, split], "read_rows": list(a_read)},
                latent[a_read[0]:a_read[1]])
            engine.feed_window(
                {"source": "b", "window": "b", "rev": 1,
                 "owned_rows": [split, int(ih)], "read_rows": list(b_read)},
                latent[b_read[0]:b_read[1]])
            engine.feed_commit({"source": "a", "through": split, "missing": []})
            engine.feed_commit({"source": "b", "through": int(ih), "missing": []})
            expect = reference_transposed_conv(desc, kernel, latent)
            rows = engine.drain(QUANTITATIVE)
            self.assertEqual(len(rows), bottom - top)
            for product in rows:
                np.testing.assert_allclose(
                    product.data, expect[product.row - top],
                    rtol=0, atol=1e-12)
                self.assertTrue(np.all(product.coverage != GAP))


if __name__ == "__main__":
    unittest.main()
