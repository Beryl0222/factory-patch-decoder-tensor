"""工作区记账：只统计接收端新分配的解包、转换、归约、版本暂存及交接缓冲。

调用方持有的只读输入（窗口张量本身）与已消费（已封行交走）的输出不计入；
额外物化的每个副本都计入。容量口径与 ``capacity.json`` 的
``peak_scratch_bytes`` 对齐，用于在大尺寸（万级潜变量行）下自证有界。
"""
from __future__ import annotations

import numpy as np


class ScratchAccount:
    """跟踪当前工作区字节与峰值。"""

    def __init__(self) -> None:
        self._live = 0
        self.peak = 0

    @property
    def live_bytes(self) -> int:
        return self._live

    def note(self, array: np.ndarray) -> np.ndarray:
        """登记一块由接收端分配的 ndarray，返回原数组便于链式使用。"""
        self._live += array.nbytes
        self.peak = max(self.peak, self._live)
        return array

    def release(self, array: np.ndarray) -> None:
        """交还一块已消费的缓冲。"""
        self._live -= array.nbytes
        if self._live < 0:  # 防御性检查，记账错误应当显眼
            raise RuntimeError("工作区记账出现负余额")

    def snapshot(self) -> dict[str, int]:
        return {"live_bytes": self._live, "peak_bytes": self.peak}
