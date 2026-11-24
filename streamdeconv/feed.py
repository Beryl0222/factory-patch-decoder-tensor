"""把采集交换记录按顺序回放进引擎。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .accounting import ScratchAccount
from .engine import StreamDeconv
from .errors import (
    CongestionError,
    RevisionConflictError,
    StreamDeconvError,
)
from .geometry import build_descriptor
from .products import PREVIEW, QUANTITATIVE, RowProduct


def build_engine(records: dict, tensors: dict[str, np.ndarray],
                 max_unsealed_rows: int | None = None,
                 account: ScratchAccount | None = None) -> StreamDeconv:
    """按记录中的解码单描述与核张量构造引擎。"""
    kernel = tensors["kernel"]
    desc = build_descriptor(records, tuple(kernel.shape), max_unsealed_rows)
    return StreamDeconv(desc, kernel, account)


def replay(records: dict, tensors: dict[str, np.ndarray],
           max_unsealed_rows: int | None = None
           ) -> tuple[StreamDeconv, dict[str, list[RowProduct]], list[dict]]:
    """按 transfers 顺序回放；返回引擎、两个订阅的分行产物与事件日志。

    迟到异值修订抛冲突但不中断回放（引擎已留痕）；拥塞同样记录后继续，
    已能确定的其他行在每次事件后照常封行交接。
    """
    account = ScratchAccount()
    engine = build_engine(records, tensors, max_unsealed_rows, account)
    delivered = {QUANTITATIVE: [], PREVIEW: []}
    log: list[dict] = []
    for index, rec in enumerate(records["transfers"]):
        try:
            if "window" in rec:
                engine.feed(rec, tensors[rec["tensor"]])
            else:
                engine.feed(rec)
        except RevisionConflictError as exc:
            log.append({"event": index + 1, "kind": "conflict",
                        "record": exc.record.as_dict()})
        except CongestionError as exc:
            log.append({"event": index + 1, "kind": "congestion",
                        "blockage": exc.blockage})
        except StreamDeconvError as exc:
            # 校验类异常同样留痕，调用方可决定中止或继续；引擎状态保持在
            # 异常前一刻，已封行产物不受影响。
            log.append({"event": index + 1, "kind": "rejected",
                        "error": type(exc).__name__, "message": str(exc)})
        for subscription in records.get("subscriptions",
                                        (QUANTITATIVE, PREVIEW)):
            delivered[subscription].extend(engine.drain(subscription))
        if log and log[-1]["event"] == index + 1:
            log[-1]["front"] = engine.sealed_front
    return engine, delivered, log


def replay_files(capture_path: str | Path,
                 max_unsealed_rows: int | None = None):
    """便捷入口：从 caseio 兼容的采集文件回放（需工作区在 sys.path 上）。"""
    from caseio import load_case

    records, tensors = load_case(Path(capture_path))
    return replay(records, tensors, max_unsealed_rows)
