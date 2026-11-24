"""分窗转置卷积的流式封行交接能力。"""
from __future__ import annotations

from .accounting import ScratchAccount
from .engine import StreamDeconv
from .errors import (
    CongestionError,
    DescriptorError,
    RevisionConflictError,
    StaleRevisionError,
    StreamDeconvError,
    TransferValidationError,
)
from .geometry import DecodeDescriptor, build_descriptor
from .products import (
    COVERED,
    GAP,
    STRUCTURAL_ZERO,
    PREVIEW,
    QUANTITATIVE,
    ConflictRecord,
    RowProduct,
    WindowState,
)

__all__ = [
    "StreamDeconv", "DecodeDescriptor", "build_descriptor", "ScratchAccount",
    "RowProduct", "ConflictRecord", "WindowState",
    "COVERED", "GAP", "STRUCTURAL_ZERO", "PREVIEW", "QUANTITATIVE",
    "StreamDeconvError", "DescriptorError", "TransferValidationError",
    "StaleRevisionError", "RevisionConflictError", "CongestionError",
]
