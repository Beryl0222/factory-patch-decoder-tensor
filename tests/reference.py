"""测试辅助：暴力全幅参考实现与小尺寸构造器。"""
from __future__ import annotations

import numpy as np

from streamdeconv.geometry import DecodeDescriptor


def reference_transposed_conv(desc: DecodeDescriptor, kernel: np.ndarray,
                              latent: np.ndarray) -> np.ndarray:
    """全幅分组转置卷积，裁剪后返回 (输出行, 输出通道, 裁剪宽)。

    与流式引擎使用同一坐标公式，但一次性物化，仅供测试对账。
    """
    ih, iw, in_c = latent.shape
    groups = desc.groups
    cg = in_c // groups
    og = desc.out_channels // groups
    kh, kw = desc.kernel_hw
    sh, sw = desc.stride
    dh, dw = desc.dilation
    oh, ow = desc.offset
    top, bottom, left, right = desc.crop
    raw_h = (ih - 1) * sh + (kh - 1) * dh + 1 + oh
    raw_w = (iw - 1) * sw + (kw - 1) * dw + 1 + ow
    # 可寻址坐标范围 [min(0, offset), raw)；数组带偏移分配，
    # 保证负 offset 下负坐标不发生环绕，正 offset 下不压缩有效宽度。
    y0, x0 = min(0, oh), min(0, ow)
    out = np.zeros((raw_h - y0, raw_w - x0, desc.out_channels),
                   dtype=np.float64)
    for g in range(groups):
        for i in range(ih):
            for j in range(iw):
                for ki in range(kh):
                    for kj in range(kw):
                        y = i * sh + ki * dh + oh
                        x = j * sw + kj * dw + ow
                        out[y - y0, x - x0, g * og:(g + 1) * og] += (
                            latent[i, j, g * cg:(g + 1) * cg]
                            @ kernel[g * cg:(g + 1) * cg, :, ki, kj])
    cropped = out[top - y0:bottom - y0, left - x0:right - x0]
    return cropped.transpose(0, 2, 1).copy()


def make_descriptor(**overrides) -> DecodeDescriptor:
    base = dict(
        canvas="unit",
        input_hw=(7, 4),
        kernel_hw=(3, 3),
        in_channels=2,
        out_channels=2,
        groups=2,
        stride=(2, 2),
        dilation=(2, 1),
        offset=(-1, 1),
        crop=(0, 15, 0, 9),
        expected_owners={"a": (0, 4), "b": (4, 7)},
        max_unsealed_rows=None,
    )
    base.update(overrides)
    desc = DecodeDescriptor(**base)
    desc.validate()
    return desc


def random_kernel(desc: DecodeDescriptor, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(desc.in_channels,
                            desc.out_channels // desc.groups,
                            *desc.kernel_hw))
