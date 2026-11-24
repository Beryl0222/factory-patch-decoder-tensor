"""窗口流转置卷积引擎：散射累加、承诺封行、修订与拥塞控制。

状态机要点
----------
* 每个窗口只按 ``owned_rows`` 贡献；重叠的读上下文（read_rows 超出部分）
  不产生输出。
* 来源 ``through`` 是不含该行的承诺界：承诺界内的潜变量行必须已有有效
  窗口贡献或被该来源显式宣告 missing，否则封行等待。
* 封行严格自前向后；一条输出行的实际支持域由核尺寸、扩张、步长、偏移
  与裁剪共同决定（见 :class:`DecodeDescriptor.row_support_inputs`），
  远处边缘行同样按支持域判定。
* 承诺生效后只接受等价重发；异值修订生成 :class:`ConflictRecord` 并抛出
  :class:`RevisionConflictError`，已交接像素不变，画幅保留可追查状态。
* 未封行槽位受 ``max_unsealed_rows`` 约束；超限时报出阻塞来源承诺，
  已经能确定的行照常封行交接。
"""
from __future__ import annotations

import bisect
import heapq

import numpy as np

from .accounting import ScratchAccount
from .errors import (
    CongestionError,
    RevisionConflictError,
    StaleRevisionError,
    TransferValidationError,
)
from .geometry import DecodeDescriptor
from .ops import scatter_window
from .products import (
    COVERED,
    GAP,
    PREVIEW,
    QUANTITATIVE,
    STRUCTURAL_ZERO,
    ConflictRecord,
    RowProduct,
    WindowState,
)


class _Slot:
    """一条未封输出行的归约槽。"""

    __slots__ = ("acc", "windows")

    def __init__(self, acc: np.ndarray):
        self.acc = acc
        self.windows: set[str] = set()


class StreamDeconv:
    def __init__(self, desc: DecodeDescriptor, kernel: np.ndarray,
                 account: ScratchAccount | None = None,
                 stage_windows: bool = True):
        """流式引擎。

        stage_windows=True 时接收端复制窗口张量做版本暂存（计入工作区），
        以支持承诺前的高修订精确替换；False 时零拷贝引用调用方的只读输入，
        不产生窗口副本（符合容量口径中“调用方已有只读输入不计入工作区”），
        此时承诺前替换要求调用方保证旧张量在其支持行封走前仍可读。
        """
        desc.validate()
        desc.precompute()
        if kernel.shape != (desc.in_channels,
                            desc.out_channels // desc.groups,
                            *desc.kernel_hw):
            raise TransferValidationError("核形状与解码单描述不一致")
        self.desc = desc
        self.kernel = kernel if kernel.dtype == np.float64 else kernel.astype(np.float64)
        self.account = account or ScratchAccount()
        self.stage_windows = stage_windows
        self.out_h, self.out_w = desc.out_hw

        self._slots: dict[int, _Slot] = {}
        self._windows: dict[str, WindowState] = {}      # 长期保留，供修订判定
        self._staged: dict[str, np.ndarray] = {}          # 版本暂存（计入工作区）
        self._retired: set[str] = set()                   # 张量已释放的窗口
        # 最小堆 (窗口支持的最大输出行, 版本, window_id)；版本随修订递增，
        # 替换路径留下的旧条目靠版本判废，把暂存回收降为摊还 O(log n)。
        self._retire_heap: list[tuple[int, int, str]] = []
        self._window_versions: dict[str, int] = {}
        self._row_window: dict[int, str] = {}             # 潜变量行 -> 生效窗口
        # 每来源已被窗口权属覆盖的潜变量区间（有序、互不交叠），
        # 供承诺校验使用，与逐行字典解耦以便后者压缩。
        self._covered: dict[str, list[list[int]]] = {
            s: [] for s in desc.expected_owners
        }
        self._through: dict[str, int] = {s: 0 for s in desc.expected_owners}
        self._missing: dict[str, list[tuple[int, int]]] = {
            s: [] for s in desc.expected_owners
        }
        self._front = self.desc.crop[0]                  # 下一条待封全局输出行
        self._sealed: list[RowProduct] = []
        self._subscriptions = frozenset({QUANTITATIVE, PREVIEW})
        self._cursors = {QUANTITATIVE: 0, PREVIEW: 0}
        self._pending_consumers: dict[int, int] = {}
        self.conflicts: list[ConflictRecord] = []
        self._event = 0

    # ------------------------------------------------------------------ 校验
    def _validate_window(self, rec: dict, values: np.ndarray) -> None:
        source = rec["source"]
        if source not in self.desc.expected_owners:
            raise TransferValidationError(f"未知来源 {source}")
        owned_lo, owned_hi = rec["owned_rows"]
        read_lo, read_hi = rec["read_rows"]
        owner_lo, owner_hi = self.desc.expected_owners[source]
        if not (owner_lo <= owned_lo < owned_hi <= owner_hi):
            raise TransferValidationError("权属区间超出该来源公布范围")
        if not (read_lo <= owned_lo and owned_hi <= read_hi):
            raise TransferValidationError("read_rows 必须覆盖 owned_rows")
        expect_shape = (read_hi - read_lo, self.desc.input_hw[1], self.desc.in_channels)
        if tuple(values.shape) != expect_shape:
            raise TransferValidationError(
                f"窗口张量形状 {tuple(values.shape)} 与约定 {expect_shape} 不符")
        if values.dtype != np.float64:
            raise TransferValidationError("窗口值必须为 float64")
        # 权属不得与同来源其他已生效窗口交叠，也不得落在已宣告缺失段内。
        for wid, state in self._windows.items():
            if wid == rec["window"]:
                continue
            if state.source == source and not (owned_hi <= state.owned_rows[0]
                                               or state.owned_rows[1] <= owned_lo):
                raise TransferValidationError(f"与已生效窗口 {wid} 权属交叠")
        for m_lo, m_hi in self._missing[source]:
            if not (owned_hi <= m_lo or m_hi <= owned_lo):
                raise TransferValidationError("权属区间与来源已宣告缺失段冲突")

    @staticmethod
    def _merge_interval(intervals: list[list[int]], lo: int, hi: int) -> None:
        """把 [lo, hi) 并入有序区间表，合并相邻或交叠段。"""
        merged: list[list[int]] = []
        placed = False
        for a, b in intervals:
            if b < lo:
                merged.append([a, b])
            elif a > hi:
                if not placed:
                    merged.append([lo, hi])
                    placed = True
                merged.append([a, b])
            else:
                lo = min(lo, a)
                hi = max(hi, b)
        if not placed:
            merged.append([lo, hi])
        intervals[:] = merged

    def _validate_commit(self, rec: dict) -> None:
        source = rec["source"]
        if source not in self.desc.expected_owners:
            raise TransferValidationError(f"未知来源 {source}")
        through = int(rec["through"])
        owner_lo, owner_hi = self.desc.expected_owners[source]
        if not (self._through[source] <= through <= owner_hi):
            raise TransferValidationError("承诺界必须单调且不超过权属范围")
        missing = [tuple(x) for x in rec.get("missing", [])]
        cursor = 0
        for lo, hi in missing:
            if lo < cursor or hi <= lo:
                raise TransferValidationError("missing 区间必须有序且互不交叠")
            if lo < owner_lo or hi > through:
                raise TransferValidationError("missing 只能落在本次承诺界内")
            cursor = hi
        # 承诺界内每一行要么落在某窗口权属覆盖区间，要么被宣告缺失。
        missing_rows: set[int] = set()
        for lo, hi in missing:
            missing_rows.update(range(lo, hi))
        covered = self._covered[source]
        starts = [a for a, _ in covered]
        for r in range(owner_lo, through):
            if r in missing_rows:
                continue
            idx = bisect.bisect_right(starts, r) - 1
            if idx < 0 or covered[idx][1] <= r:
                raise TransferValidationError(
                    f"来源 {source} 承诺到 {through}，但潜变量行 {r} 既无窗口也未宣告缺失")

    # ------------------------------------------------------------- 槽位与拥塞
    def _hold(self, values: np.ndarray) -> np.ndarray:
        """按暂存策略持有窗口张量：暂存模式复制并记账，否则零拷贝引用。"""
        if self.stage_windows:
            return self.account.note(values.copy())
        return values

    def _discard(self, wid: str, tensor: np.ndarray) -> None:
        if self.stage_windows:
            self.account.release(tensor)

    def _slot_for(self, window_id: str, create: bool = True):
        def resolve(y: int):
            slot = self._slots.get(y)
            if slot is None:
                if not create:
                    return None  # 撤销旧值时不得实例化新槽
                slot = _Slot(self.account.note(np.zeros(
                    (self.desc.out_channels, self.out_w), dtype=np.float64)))
                self._slots[y] = slot
            slot.windows.add(window_id)
            return slot.acc
        return resolve

    def _cropped_support(self, owned_rows: tuple[int, int]) -> list[int]:
        first, last = self.desc.window_support_rows(owned_rows)
        top = self.desc.crop[0]
        return [y for y in range(max(first, top),
                                 min(last, self.desc.crop[1]))]

    def _congestion_blockage(self) -> dict:
        """说出当前阻塞封行进度的来源承诺与未决格点。"""
        front = self._front
        blockers: list[dict] = []
        if front < self.desc.crop[1]:
            lo, hi = self.desc.row_support_inputs(front)
            for r in range(lo, hi):
                source = self.desc.owner_lookup[r]
                if source is None:
                    continue
                if r in self._row_window:
                    continue
                if any(m_lo <= r < m_hi for m_lo, m_hi in self._missing[source]):
                    state = "declared_missing"
                elif self._through[source] > r:
                    state = "promised_but_window_absent"
                else:
                    state = "awaiting_promise"
                blockers.append({
                    "source": source,
                    "latent_row": r,
                    "state": state,
                    "promised_through": self._through[source],
                })
        return {
            "message": (f"未封行槽位不足，前沿输出行 {front} 等待来源承诺推进，"
                        f"阻塞点：" + "；".join(
                            f"{b['source']} 行{b['latent_row']}({b['state']},"
                            f"through={b['promised_through']}" for b in blockers)
                        if blockers else "未封行槽位不足"),
            "front_row": front,
            "blockers": blockers,
            "through": dict(self._through),
            "unsealed_rows": len(self._slots),
            "max_unsealed_rows": self.desc.max_unsealed_rows,
        }

    # ------------------------------------------------------------------ 窗口
    def feed_window(self, rec: dict, values: np.ndarray) -> None:
        self._event += 1
        values = np.asarray(values, dtype=np.float64)
        self._validate_window(rec, values)
        source, wid, rev = rec["source"], rec["window"], int(rec["rev"])
        owned = tuple(rec["owned_rows"])
        read_rows = tuple(rec["read_rows"])
        existing = self._windows.get(wid)

        if existing is not None:
            if rev < existing.rev:
                raise StaleRevisionError(
                    f"窗口 {wid} 修订 {rev} 已被 {existing.rev} 取代")
            if rev == existing.rev:
                if wid in self._retired or (
                        wid in self._staged
                        and np.array_equal(values, self._staged[wid])):
                    return  # 等价重发（含封行后的重放），幂等接收
                raise TransferValidationError(f"窗口 {wid} 同修订号内容不一致")
            committed = [r for r in range(*owned) if self._through[source] > r]
            if committed:
                # 权属的一部分已进入承诺界：异值修订只能留痕，
                # 不能改写已交接像素。retired 窗口的全部权属必然已承诺。
                committed_rows = committed
                affected = sorted(y for y in self._cropped_support(
                    (min(committed_rows), max(committed_rows) + 1))
                    if y < self._front)
                record = ConflictRecord(
                    source=source, window=wid, rejected_rev=rev,
                    active_rev=existing.rev,
                    reason=("承诺已生效，异值修订被拒绝；"
                            f"已交接受影响行 {affected}，像素保持原值"),
                    affected_rows=tuple(affected), event_index=self._event)
                self.conflicts.append(record)
                raise RevisionConflictError(record)
            # 承诺界外：高修订替换本窗。
            if wid in self._retired:
                # 旧张量已随封行离开工作区，已交接像素不受影响；
                # 仅把新值散射到仍未封的行。
                self._retired.discard(wid)
                for r in range(*existing.owned_rows):
                    self._row_window.pop(r, None)
            else:
                old = self._staged.pop(wid)
                scatter_window(self.desc, self.kernel, old, existing.owned_rows,
                               existing.read_rows[0],
                               self._slot_for(wid, create=False),
                               self.account, sign=-1.0)
                self._discard(wid, old)
                # 撤销后全零的槽没有任何有效贡献，回收其工作区。
                dead_slots = [(y, s) for y, s in self._slots.items()
                              if not np.any(s.acc)]
                for dead_y, dead_slot in dead_slots:
                    self.account.release(dead_slot.acc)
                    del self._slots[dead_y]
                for r in range(*existing.owned_rows):
                    self._row_window.pop(r, None)
            staged = self._hold(values)
            scatter_window(self.desc, self.kernel, staged, owned,
                           read_rows[0], self._slot_for(wid), self.account)
            self._staged[wid] = staged
            self._windows[wid] = WindowState(source, wid, rev, owned, read_rows)
            self._push_retire(wid, owned)
            self._merge_interval(self._covered[source], *owned)
            for r in range(*owned):
                self._row_window[r] = wid
            self._seal_front()
            return

        # 全新窗口：先把已经能确定的行封走腾出槽位，再做拥塞预检查。
        self._seal_front()
        if self.desc.max_unsealed_rows is not None:
            touched = {y for y in self._cropped_support(owned) if y >= self._front}
            growth = sum(1 for y in touched if y not in self._slots)
            if len(self._slots) + growth > self.desc.max_unsealed_rows:
                raise CongestionError(self._congestion_blockage())

        staged = self._hold(values)
        scatter_window(self.desc, self.kernel, staged, owned,
                       read_rows[0], self._slot_for(wid), self.account)
        self._staged[wid] = staged
        self._windows[wid] = WindowState(source, wid, rev, owned, read_rows)
        self._push_retire(wid, owned)
        self._merge_interval(self._covered[source], *owned)
        for r in range(*owned):
            self._row_window[r] = wid
        self._seal_front()

    # ------------------------------------------------------------------ 承诺
    def feed_commit(self, rec: dict) -> None:
        self._event += 1
        self._validate_commit(rec)
        source = rec["source"]
        self._through[source] = int(rec["through"])
        missing = [tuple(x) for x in rec.get("missing", [])]
        # 同一来源多次承诺：保留落在最新承诺界内的全部缺失宣告（区间互不交叠）。
        merged = sorted(set(self._missing[source]) | set(missing))
        self._missing[source] = merged
        self._seal_front()

    def feed(self, rec: dict, values: np.ndarray | None = None) -> None:
        """按交换记录类型分发；冲突与拥塞以异常上抛，调用方可继续后续记录。"""
        if "window" in rec:
            if values is None:
                raise TransferValidationError("窗口记录缺少张量")
            self.feed_window(rec, values)
        else:
            self.feed_commit(rec)

    # ------------------------------------------------------------------ 封行
    def _is_structural_zero(self, y: int) -> bool:
        lo, hi = self.desc.row_support_inputs(y)
        return lo >= hi

    def _build_coverage(self, y: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
        coverage = np.full(self.out_w, STRUCTURAL_ZERO, dtype=object)
        lo, hi = self.desc.row_support_inputs(y)
        rows = np.arange(lo, hi)
        owners = self.desc.owner_lookup
        # 逐潜变量行判定缺失/有效，全部为 O(1) 查表。
        missing_flags = np.zeros(len(rows), dtype=bool)
        covered_flags = np.zeros(len(rows), dtype=bool)
        gap_inputs: list[tuple[int, int]] = []
        for idx, r in enumerate(rows):
            source = owners[r]
            if source is None:
                continue
            if any(m_lo <= r < m_hi for m_lo, m_hi in self._missing[source]):
                missing_flags[idx] = True
                gap_inputs.append((int(r), int(r) + 1))
            elif r in self._row_window:
                covered_flags[idx] = True
        has_gap = bool(missing_flags.any())
        has_cover = bool(covered_flags.any())
        live = self.desc.live_col_mask
        # 缺口优先：该单元有缺失行本应给出的贡献，即使其他行已有有效贡献
        # 也不能用于定量判断，只能作为预览。
        if has_gap:
            coverage[live] = GAP
        elif has_cover:
            coverage[live] = COVERED
        return coverage, gap_inputs

    def _seal_front(self) -> None:
        top, bottom, left, right = self.desc.crop
        while self._front < bottom:
            y = self._front
            if self._is_structural_zero(y):
                coverage = np.full(self.out_w, STRUCTURAL_ZERO, dtype=object)
                product = RowProduct(
                    row=y,
                    data=self.account.note(np.zeros(
                        (self.desc.out_channels, self.out_w), dtype=np.float64)),
                    coverage=coverage, quality=QUANTITATIVE,
                    gap_input_rows=(),
                    channel_order=tuple(range(self.desc.out_channels)),
                    crop=self.desc.crop, sealed_event=self._event,
                    measurable_event=self._event)
                self._pending_consumers[id(product)] = len(self._subscriptions)
                self._sealed.append(product)
                self._front += 1
                continue
            # 封行判定只看实际支持域。对支持域内每个有权属来源的潜变量行：
            # 承诺界未越过该行 -> 晚到窗口仍可改图，必须等待；
            # 承诺界已越过 -> 行内必须已有有效窗口贡献或被显式宣告缺失
            # （后者由 feed_commit 的协议校验保证）。
            lo, hi = self.desc.row_support_inputs(y)
            owners = self.desc.owner_lookup
            resolved = True
            for r in range(lo, hi):
                source = owners[r]
                if source is None:
                    continue
                if self._through[source] <= r:
                    resolved = False
                    break
            if not resolved:
                break
            slot = self._slots.get(y)
            coverage, gap_inputs = self._build_coverage(y)
            quality = PREVIEW if gap_inputs else QUANTITATIVE
            if slot is None:
                # 支持域非空但承诺证明不会再有贡献：零值是“已证实为零”，
                # 与“尚未收到任何贡献”的 pending 含义不同。
                acc = self.account.note(np.zeros(
                    (self.desc.out_channels, self.out_w), dtype=np.float64))
                provenance: tuple[str, ...] = ()
            else:
                acc = slot.acc
                provenance = tuple(sorted(slot.windows))
            product = RowProduct(
                row=y,
                data=acc,
                coverage=coverage,
                quality=quality,
                gap_input_rows=tuple(gap_inputs),
                channel_order=tuple(range(self.desc.out_channels)),
                crop=self.desc.crop,
                sealed_event=self._event,
                measurable_event=self._event if quality == QUANTITATIVE else None,
                provenance=provenance)
            self._pending_consumers[id(product)] = (
                len(self._subscriptions) if quality == QUANTITATIVE else 1)
            self._sealed.append(product)
            if slot is not None:
                del self._slots[y]
            self._front += 1
            self._release_staged()
            self._compact_row_window()

    def _compact_row_window(self) -> None:
        """压缩逐行窗口映射，使驻留规模与钢卷总长无关。

        承诺校验已改由来源级覆盖区间承担。已承诺到权属终点的来源不会再
        产生新窗口，其行映射整体失效；其余来源取承诺界最小值作为保留下界。
        """
        if not self._row_window:
            return
        active_through = [
            t for source, t in self._through.items()
            if t < self.desc.expected_owners[source][1]]
        keep_from = min(active_through) if active_through \
            else self.desc.input_hw[0]
        for r in [r for r in self._row_window if r < keep_from]:
            del self._row_window[r]

    def _push_retire(self, wid: str, owned_rows: tuple[int, int]) -> None:
        support = self._cropped_support(owned_rows)
        max_support = max(support) if support else self.desc.crop[0] - 1
        version = self._window_versions.get(wid, 0) + 1
        self._window_versions[wid] = version
        heapq.heappush(self._retire_heap, (max_support, version, wid))

    def _release_staged(self) -> None:
        """版本暂存张量在窗口支持的输出行全部交接后离开工作区。

        窗口元数据保留（标记 retired），使封行后的迟到修订仍可被识别并留痕。
        用最小堆按“窗口支持的最大输出行”排序，替换路径的旧条目按版本判废。
        """
        while self._retire_heap and self._retire_heap[0][0] < self._front:
            _, version, wid = heapq.heappop(self._retire_heap)
            if self._window_versions.get(wid) != version:
                continue  # 已被更高修订取代的过期条目
            if wid in self._retired or wid not in self._staged:
                continue
            self._discard(wid, self._staged.pop(wid))
            self._retired.add(wid)

    # ------------------------------------------------------------------ 回读
    def drain(self, subscription: str) -> list[RowProduct]:
        """取走该订阅可见的新封行；quantitative 只给无缺失影响的行。

        数据数组的工作区记账按订阅引用计数：最后一个有权消费的订阅取走后
        才释放，避免同一缓冲被重复归还。
        """
        if subscription not in (QUANTITATIVE, PREVIEW):
            raise TransferValidationError(f"未知订阅 {subscription}")
        cursor = self._cursors[subscription]
        items: list[RowProduct] = []
        while cursor < len(self._sealed):
            product = self._sealed[cursor]
            if subscription == PREVIEW or product.quality == QUANTITATIVE:
                items.append(product)
                key = id(product)
                self._pending_consumers[key] -= 1
                if self._pending_consumers[key] == 0:
                    self.account.release(product.data)
                    del self._pending_consumers[key]
            cursor += 1
        self._cursors[subscription] = cursor
        # 两个订阅都越过的封行前缀不再需要，压缩列表使驻留与画幅长度无关。
        consumed_to = min(self._cursors.values())
        if consumed_to > 0:
            del self._sealed[:consumed_to]
            for sub in self._cursors:
                self._cursors[sub] -= consumed_to
        return items

    def blockage(self) -> dict:
        """当前封行前沿的阻塞诊断（不改变状态）。"""
        return self._congestion_blockage()

    @property
    def sealed_front(self) -> int:
        """已成功封行的行界（不含该行）。"""
        return self._front
