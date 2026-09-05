from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DialogRegion:
    """検出した会話ウィンドウ。座標は元画像上のピクセル。"""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    layout: str

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    variant: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
