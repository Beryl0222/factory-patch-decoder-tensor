"""分组转置卷积的流式散射累加。

只对窗口声明权属（owned）的潜变量格点做转置卷积；read_rows 中超出
owned_rows 的重叠读上下文仅用于定位，不产生贡献。结果直接散射进
按行持有的输出槽，不物化整张画幅。

核轴：(输入通道, 组内输出通道, 核行, 核列)。
输出坐标：``y = i*stride_h + k*dilation_h + offset_h``，列同理。
同一输出行可由不同 (i, k) 对多次命中（大步长/扩张下支持域重叠），
全部以 float64 累加；``sign`` 取负即用于修订替换时的旧值撤销。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .accounting import ScratchAccount
from .geometry import DecodeDescriptor


def scatter_window(
    desc: DecodeDescriptor,
    kernel: np.ndarray,
    values: np.ndarray,
    owned_rows: tuple[int, int],
    read_start: int,
    slot_for: Callable[[int], np.ndarray | None],
    account: ScratchAccount,
    sign: float = 1.0,
) -> None:
    """把一个窗口的权属格点散射累加进输出槽。

    slot_for(y) 返回全局输出行 y 的累加数组（形状 输出通道×裁剪宽），
    行落在裁剪框外时返回 None。按 (ki, kj) 分块、块内矩阵乘向量化；
    同一 ki 下输出行互不重复（stride>0），重复命中只发生在不同 ki 之间，
    由各自分块自然累加，无需排序去重。
    """
    owned_lo, owned_hi = owned_rows
    win_h, win_w, in_c = values.shape
    groups = desc.groups
    cg = in_c // groups
    og = desc.out_channels // groups
    kh, kw = desc.kernel_hw
    sh, sw = desc.stride
    dh, dw = desc.dilation
    oh, ow = desc.offset
    top, bottom, left, right = desc.crop

    i_global = np.arange(owned_lo, owned_hi)
    local_row = i_global - read_start
    row_in_tensor = (local_row >= 0) & (local_row < win_h)
    j_global = np.arange(win_w)

    for g in range(groups):
        xg = values[:, :, g * cg:(g + 1) * cg]
        kg = kernel[g * cg:(g + 1) * cg]  # (cg, og, kh, kw)
        for ki in range(kh):
            y_global = i_global * sh + ki * dh + oh
            row_ok = row_in_tensor & (y_global >= top) & (y_global < bottom)
            if not row_ok.any():
                continue
            ys = y_global[row_ok]
            local_kept = local_row[row_ok]
            for kj in range(kw):
                x_global = j_global * sw + kj * dw + ow
                col_ok = (x_global >= left) & (x_global < right)
                if not col_ok.any():
                    continue
                xr = xg[local_kept][:, col_ok, :]       # (nR, nC, cg)
                kk = kg[:, :, ki, kj]                  # (cg, og)
                tile = account.note(xr @ kk)           # (nR, nC, og)
                x_local = x_global[col_ok] - left
                for ridx, y in enumerate(ys):
                    acc = slot_for(int(y))
                    if acc is None:
                        continue
                    acc[g * og:(g + 1) * og][:, x_local] += sign * tile[ridx].T
                account.release(tile)
