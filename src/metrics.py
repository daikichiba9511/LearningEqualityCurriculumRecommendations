from __future__ import annotations

from collections.abc import Sequence


def iou(a: Sequence[str], b: Sequence[str]) -> float:
    return len(set(a).intersection(set(b))) / len(set(a + b))
