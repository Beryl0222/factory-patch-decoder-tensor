"""封行交接产物与冲突、阻塞留痕的数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 单元覆盖状态：已收到有效贡献 / 来源宣告缺失形成的明确缺口 / 可证明为零
COVERED = "covered"
GAP = "gap"
STRUCTURAL_ZERO = "structural_zero"

# 行级质量：可用于定量判断 / 仅允许预览
QUANTITATIVE = "quantitative"
PREVIEW = "preview"


@dataclass(frozen=True)
class RowProduct:
    """一条成功封行的分行产物。

    data 轴为 (输出通道, 列)，float64 累加结果；
    coverage 轴为 (列,)，逐列区分有效贡献、缺口与可证明零；
    gap_input_rows 为导致缺口的潜变量行（全局坐标，左闭右开区间列表）；
    sealed_event 为封行时的事件序号，measurable_event 是该行可用于测量的
    最早事件（定量行等于封行事件，预览行永远不可测量故为 None）。
    """

    row: int
    data: np.ndarray
    coverage: np.ndarray
    quality: str
    gap_input_rows: tuple[tuple[int, int], ...]
    channel_order: tuple[int, ...]
    crop: tuple[int, int, int, int]
    sealed_event: int
    measurable_event: int | None
    provenance: tuple[str, ...] = field(default_factory=tuple)

    @property
    def measurable(self) -> bool:
        return self.quality == QUANTITATIVE

    def gap_columns(self) -> tuple[tuple[int, int], ...]:
        """缺口腔在裁剪坐标中的左闭右开区间（游程编码）。"""
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for x, state in enumerate(self.coverage):
            if state == GAP and start is None:
                start = x
            elif state != GAP and start is not None:
                runs.append((start + self.crop[2], x + self.crop[2]))
                start = None
        if start is not None:
            runs.append((start + self.crop[2], len(self.coverage) + self.crop[2]))
        return tuple(runs)

    def as_dict(self) -> dict:
        """回读通道顺序、裁剪坐标及测量可用性的结构化视图。"""
        return {
            "row": self.row,
            "quality": self.quality,
            "measurable": self.measurable,
            "sealed_event": self.sealed_event,
            "measurable_event": self.measurable_event,
            "channel_order": list(self.channel_order),
            "crop": list(self.crop),
            "gap_columns": [list(run) for run in self.gap_columns()],
            "gap_input_rows": [list(run) for run in self.gap_input_rows],
            "provenance": list(self.provenance),
            "shape": list(self.data.shape),
        }


@dataclass(frozen=True)
class ConflictRecord:
    """承诺生效后到达的异值修订，画幅留在可追查状态。"""

    source: str
    window: str
    rejected_rev: int
    active_rev: int
    reason: str
    affected_rows: tuple[int, ...]
    event_index: int

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "window": self.window,
            "rejected_rev": self.rejected_rev,
            "active_rev": self.active_rev,
            "reason": self.reason,
            "affected_rows": list(self.affected_rows),
            "event_index": self.event_index,
        }


@dataclass(frozen=True)
class WindowState:
    """已生效窗口的版本暂存记录（权属区间按全局潜变量坐标）。"""

    source: str
    window: str
    rev: int
    owned_rows: tuple[int, int]
    read_rows: tuple[int, int]
