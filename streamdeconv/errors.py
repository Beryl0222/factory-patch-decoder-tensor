"""流式转置卷积的异常类型。"""
from __future__ import annotations


class StreamDeconvError(Exception):
    """流式转置卷积引擎的基础异常。"""


class DescriptorError(StreamDeconvError):
    """画幅解码单描述自身不合法。"""


class TransferValidationError(StreamDeconvError):
    """窗口或承诺记录与解码单描述不一致。"""


class StaleRevisionError(StreamDeconvError):
    """收到的修订号不高于已生效修订且内容不一致。"""


class RevisionConflictError(StreamDeconvError):
    """承诺生效后到达的异值修订，承载冲突留痕记录。"""

    def __init__(self, record):
        self.record = record
        super().__init__(record.reason)


class CongestionError(StreamDeconvError):
    """未封行槽位不足，承载阻塞来源承诺的诊断信息。"""

    def __init__(self, blockage):
        self.blockage = blockage
        super().__init__(blockage["message"])
