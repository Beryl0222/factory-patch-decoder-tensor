"""画幅绑定的解码单描述与支持域几何。

坐标约定（与采集资料一致）：
- 潜变量格点 ``i`` 的输出位置为 ``i * stride + k * dilation + offset``，
  其中 ``k`` 取核内坐标区间 ``[0, kernel_size)``；
- 裁剪框为左闭右开的全局坐标区间；
- 输入通道按 ``groups`` 连续分组，输出组保持同序。

转置卷积沿行、列可分离，本模块只依赖区间算术，不物化整张画幅，
因此钢卷（输入行方向）可以任意长而不改变工作区上界。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import DescriptorError


@dataclass(frozen=True)
class DecodeDescriptor:
    """一个画幅绑定的唯一解码单描述。"""

    canvas: str
    input_hw: tuple[int, int]          # 潜变量全幅 (行数, 列数)
    kernel_hw: tuple[int, int]         # (核行, 核列)
    in_channels: int
    out_channels: int
    groups: int
    stride: tuple[int, int]
    dilation: tuple[int, int]
    offset: tuple[int, int]
    crop: tuple[int, int, int, int]    # (top, bottom, left, right)，左闭右开
    expected_owners: dict[str, tuple[int, int]]
    max_unsealed_rows: int | None = None

    # ---- 派生几何 ----
    @property
    def out_hw(self) -> tuple[int, int]:
        top, bottom, left, right = self.crop
        return (bottom - top, right - left)

    def latent_support_rows(self, i: int) -> tuple[int, int]:
        """潜变量行 ``i`` 可贡献到的全局输出行区间（左闭右开）。"""
        kh, dh, sh, oh = self.kernel_hw[0], self.dilation[0], self.stride[0], self.offset[0]
        first = i * sh + oh
        return first, first + (kh - 1) * dh + 1

    def row_support_inputs(self, y: int) -> tuple[int, int]:
        """可能影响全局输出行 ``y`` 的潜变量行区间（左闭右开）。

        即所有满足 ``i*sh+oh <= y <= i*sh+oh+(kh-1)*dh`` 的 ``i``。
        封行依据必须与该实际支持域一致，不能只按步长邻行判断。
        """
        kh, dh, sh, oh = self.kernel_hw[0], self.dilation[0], self.stride[0], self.offset[0]
        span = (kh - 1) * dh
        # 由 i*sh+oh+span >= y 得 i >= (y-oh-span)/sh（下界向上取整）；
        # 由 i*sh+oh <= y 得 i <= (y-oh)/sh（上界向下取整）。
        i_min = -(-(y - oh - span) // sh)
        i_max = (y - oh) // sh
        return max(0, i_min), min(self.input_hw[0], i_max + 1)

    def window_support_rows(self, owned_rows: tuple[int, int]) -> tuple[int, int]:
        """窗口权属潜变量区间影响到的全局输出行区间（裁剪前）。"""
        lo, hi = owned_rows
        if lo >= hi:
            raise DescriptorError("权属区间为空")
        first, _ = self.latent_support_rows(lo)
        _, last = self.latent_support_rows(hi - 1)
        return first, last

    def latent_support_cols(self, j: int) -> tuple[int, int]:
        kw, dw, sw, ow = self.kernel_hw[1], self.dilation[1], self.stride[1], self.offset[1]
        first = j * sw + ow
        return first, first + (kw - 1) * dw + 1

    def col_support_inputs(self, x: int) -> tuple[int, int]:
        kw, dw, sw, ow = self.kernel_hw[1], self.dilation[1], self.stride[1], self.offset[1]
        span = (kw - 1) * dw
        j_min = -(-(x - ow - span) // sw)
        j_max = (x - ow) // sw
        return max(0, j_min), min(self.input_hw[1], j_max + 1)

    def owner_of(self, latent_row: int) -> str | None:
        """公布的权属来源；expected_owners 之外的格点没有任何来源负责。"""
        if 0 <= latent_row < self.input_hw[0]:
            owner = self._owner_lookup[latent_row]
            return owner if owner is not None else None
        return None

    def precompute(self) -> None:
        """派生一次性查表，供引擎在长卷上以 O(1) 访问，避免逐行线性扫描。"""
        if getattr(self, "_ready", False):
            return
        ih, _ = self.input_hw
        lookup: list[str | None] = [None] * ih
        for source, (lo, hi) in self.expected_owners.items():
            for r in range(lo, hi):
                lookup[r] = source
        object.__setattr__(self, "_owner_lookup", lookup)
        top, bottom, left, right = self.crop
        lo_arr = np.empty(bottom - top, dtype=np.int64)
        hi_arr = np.empty(bottom - top, dtype=np.int64)
        for idx, y in enumerate(range(top, bottom)):
            lo_arr[idx], hi_arr[idx] = self.row_support_inputs(y)
        object.__setattr__(self, "_support_lo", lo_arr)
        object.__setattr__(self, "_support_hi", hi_arr)
        # 裁剪列是否能由任意潜变量列产生（扩张>1 时存在空洞列）。
        live = np.zeros(right - left, dtype=bool)
        for idx, x in enumerate(range(left, right)):
            jlo, jhi = self.col_support_inputs(x)
            live[idx] = jlo < jhi
        object.__setattr__(self, "_live_col_mask", live)
        object.__setattr__(self, "_ready", True)

    @property
    def owner_lookup(self) -> list[str | None]:
        self.precompute()
        return self._owner_lookup

    @property
    def support_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """每条裁剪输出行的支持潜变量行 (下界数组, 上界数组)。"""
        self.precompute()
        return self._support_lo, self._support_hi

    @property
    def live_col_mask(self) -> np.ndarray:
        self.precompute()
        return self._live_col_mask

    def validate(self) -> None:
        """构造期一致性校验，错误一律抛 DescriptorError。"""
        ih, iw = self.input_hw
        kh, kw = self.kernel_hw
        sh, sw = self.stride
        dh, dw = self.dilation
        if ih <= 0 or iw <= 0 or kh <= 0 or kw <= 0:
            raise DescriptorError("尺寸必须为正")
        if sh <= 0 or sw <= 0 or dh <= 0 or dw <= 0:
            raise DescriptorError("步长与扩张必须为正")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise DescriptorError("通道数必须为正")
        if self.groups <= 0 or self.in_channels % self.groups:
            raise DescriptorError("输入通道不能被组数整除")
        if self.out_channels % self.groups:
            raise DescriptorError("输出通道不能被组数整除")
        top, bottom, left, right = self.crop
        if bottom <= top or right <= left:
            raise DescriptorError("裁剪框必须非空")
        # 可寻址输出坐标范围 [min(0, offset), offset+(n-1)*stride+(k-1)*dilation]。
        raw_bottom = (ih - 1) * sh + (kh - 1) * dh + 1 + self.offset[0]
        raw_right = (iw - 1) * sw + (kw - 1) * dw + 1 + self.offset[1]
        if not (min(0, self.offset[0]) <= top < bottom <= raw_bottom):
            raise DescriptorError("行裁剪框越界")
        if not (min(0, self.offset[1]) <= left < right <= raw_right):
            raise DescriptorError("列裁剪框越界")
        # 权属公布必须恰好覆盖连续的潜变量行轴。
        covered = sorted((lo, hi, s) for s, (lo, hi) in self.expected_owners.items())
        if not covered or covered[0][0] != 0:
            raise DescriptorError("expected_owners 必须从第 0 行开始")
        cursor = 0
        for lo, hi, source in covered:
            if lo != cursor or hi <= lo:
                raise DescriptorError(f"来源 {source} 的权属区间不连续")
            cursor = hi
        if cursor != ih:
            raise DescriptorError("expected_owners 未覆盖全部潜变量行")
        if self.max_unsealed_rows is not None and self.max_unsealed_rows <= 0:
            raise DescriptorError("max_unsealed_rows 必须为正")


def build_descriptor(
    records: dict,
    kernel_shape: tuple[int, ...],
    max_unsealed_rows: int | None = None,
) -> DecodeDescriptor:
    """从采集交换记录与核张量形状构造解码单描述。

    核轴为（输入通道, 组内输出通道, 核行, 核列），
    故总输出通道 = 核形状[1] * groups。
    """
    in_c, out_per_group, kh, kw = kernel_shape
    desc = DecodeDescriptor(
        canvas=str(records["canvas"]),
        input_hw=tuple(records["input_hw"]),
        kernel_hw=(kh, kw),
        in_channels=in_c,
        out_channels=out_per_group * records["groups"],
        groups=records["groups"],
        stride=tuple(records["stride"]),
        dilation=tuple(records["dilation"]),
        offset=tuple(records.get("offset", (0, 0))),
        crop=tuple(records["crop"]),
        expected_owners={
            source: tuple(bounds) for source, bounds in records["expected_owners"].items()
        },
        max_unsealed_rows=max_unsealed_rows,
    )
    desc.validate()
    return desc
