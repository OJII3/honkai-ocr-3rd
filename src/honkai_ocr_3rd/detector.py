from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import DialogRegion


@dataclass(frozen=True)
class DetectorConfig:
    """ゲーム画面に対する弱い事前知識。

    解像度に依存しないよう、すべて画面サイズに対する比率で持つ。
    """

    search_top: float = 0.52
    search_bottom: float = 0.99
    expected_height: float = 0.275
    min_height: float = 0.16
    max_height: float = 0.38
    min_width: float = 0.28


def dialog_mask(frame: np.ndarray) -> np.ndarray:
    """明るい半透明パネルを拾うためのデバッグ可能なマスクを作る。"""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # パネルは白〜淡色。彩度を広めに許容して、背景の明るい部分だけで
    # 検出が落ちないようにする。
    mask = cv2.inRange(hsv, (0, 0, 145), (180, 185, 255))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _candidate_score(
    mask: np.ndarray,
    gray: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    config: DetectorConfig,
) -> float:
    image_height, image_width = gray.shape
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)
    inner_x1 = min(x2 - 1, x + max(4, int(width * 0.08)))
    inner_x2 = max(inner_x1 + 1, x2 - max(4, int(width * 0.02)))
    inner_y1 = min(y2 - 1, y + max(3, int(height * 0.14)))
    inner_y2 = max(inner_y1 + 1, y2 - max(3, int(height * 0.14)))

    interior_mask = mask[inner_y1:inner_y2, inner_x1:inner_x2]
    fill_score = float(np.mean(interior_mask > 0))

    # 上下の境界は、周辺の画素との差分として見る。背景の大きな輪郭よりも
    # ウィンドウ全体に続く線を評価するため、横方向の中央値を使う。
    top = gray[max(0, y - 2):min(image_height, y + 3), inner_x1:inner_x2]
    bottom = gray[max(0, y2 - 3):min(image_height, y2 + 2), inner_x1:inner_x2]
    top_edge = float(np.mean(np.abs(np.diff(top.astype(np.float32), axis=0)))) / 255.0
    bottom_edge = float(np.mean(np.abs(np.diff(bottom.astype(np.float32), axis=0)))) / 255.0
    border_score = _clamp((top_edge + bottom_edge) * 2.5)

    height_ratio = height / image_height
    height_score = _clamp(1.0 - abs(height_ratio - config.expected_height) / 0.14)
    right_score = _clamp((x + width) / image_width)
    bottom_score = _clamp((y2 / image_height - 0.72) / 0.25)

    # 明るい面積を主軸にしつつ、固定に近い高さと画面右端まで伸びる特徴を
    # 併用する。confidence は確率ではなく候補順位用のスコア。
    return (
        fill_score * 0.48
        + border_score * 0.17
        + height_score * 0.20
        + right_score * 0.10
        + bottom_score * 0.05
    )


def _horizontal_transition(gray: np.ndarray, y: int) -> np.ndarray:
    """行 y の前後にある横長の境界を列ごとに測る。"""

    if y < 2 or y + 2 > gray.shape[0]:
        return np.zeros(gray.shape[1], dtype=np.float32)
    before = np.mean(gray[y - 2:y].astype(np.float32), axis=0)
    after = np.mean(gray[y:y + 2].astype(np.float32), axis=0)
    return np.abs(after - before)


def _edge_candidates(frame: np.ndarray, config: DetectorConfig) -> list[tuple[float, int, int, int, int]]:
    """パネル上端・下端の横線から矩形候補を作る。"""

    image_height, image_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    expected_height = int(image_height * config.expected_height)
    y_start = max(2, int(image_height * config.search_top) - int(image_height * 0.08))
    y_end = min(image_height - expected_height - 2, int(image_height * 0.80))
    candidates: list[tuple[float, int, int, int, int]] = []

    for y in range(y_start, y_end + 1):
        profile = _horizontal_transition(gray, y)
        # 文字や背景の短い輪郭を捨て、横長の境界だけを残す。
        edge_binary = (profile >= 24).astype(np.uint8)[None, :] * 255
        edge_binary = cv2.morphologyEx(
            edge_binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
        )
        components, labels, stats, _ = cv2.connectedComponentsWithStats(edge_binary)
        for component in range(1, components):
            start = int(stats[component, cv2.CC_STAT_LEFT])
            span = int(stats[component, cv2.CC_STAT_WIDTH])
            if span / image_width < config.min_width:
                continue

            margin = max(8, int(expected_height * 0.25))
            x = max(0, start - margin)
            right = min(image_width, start + span + margin)
            width = right - x
            bottom = min(image_height - 1, y + expected_height)
            if bottom / image_height < 0.86:
                continue
            bottom_profile = _horizontal_transition(gray, bottom)
            top_strength = float(np.mean(profile[start:start + span])) / 255.0
            bottom_strength = float(np.mean(bottom_profile[x:right])) / 255.0

            inner_x1 = x + max(4, int(width * 0.08))
            inner_x2 = max(inner_x1 + 1, right - max(4, int(width * 0.03)))
            inner_y1 = y + max(4, int(expected_height * 0.14))
            inner_y2 = max(inner_y1 + 1, bottom - max(4, int(expected_height * 0.14)))
            interior = gray[inner_y1:inner_y2, inner_x1:inner_x2]
            bright_fraction = float(np.mean(interior > 150))
            mean_brightness = float(np.mean(interior))
            if bright_fraction < 0.35 or mean_brightness < 150.0:
                continue
            brightness = _clamp((mean_brightness - 95.0) / 150.0)
            span_score = _clamp(span / (image_width * 0.65))

            score = (
                _clamp(top_strength * 2.0) * 0.25
                + _clamp(bottom_strength * 2.0) * 0.22
                + bright_fraction * 0.26
                + brightness * 0.17
                + span_score * 0.10
            )
            candidates.append((score, x, y, width, bottom - y))
    return candidates


def detect_dialog(frame: np.ndarray, config: DetectorConfig | None = None) -> DialogRegion | None:
    """フレームから会話ウィンドウを1つ検出する。

    現段階では、対象ゲームの「画面下部・ほぼ一定の高さ・右端まで伸びる
    淡色パネル」という特徴を利用する。検出に失敗したフレームは無理に
    OCRせず、後段で要確認として扱う。
    """

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("BGRのカラー画像を指定してください")

    config = config or DetectorConfig()
    image_height, image_width = frame.shape[:2]
    mask = dialog_mask(frame)
    search_y1 = int(image_height * config.search_top)
    search_y2 = int(image_height * config.search_bottom)

    edge_candidates = _edge_candidates(frame, config)
    if edge_candidates:
        score, x, y, width, height = max(edge_candidates, key=lambda candidate: candidate[0])
        if score >= 0.40:
            width = min(width, image_width - x)
            height = min(height, image_height - y)
            layout = "full" if x / image_width <= 0.08 else "right"
            return DialogRegion(
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=round(float(_clamp(score)), 4),
                layout=layout,
            )

    search_mask = np.zeros_like(mask)
    search_mask[search_y1:search_y2, :] = mask[search_y1:search_y2, :]
    contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, int, int, int, int]] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        width_ratio = width / image_width
        height_ratio = height / image_height
        if width_ratio < config.min_width:
            continue
        if not config.min_height <= height_ratio <= config.max_height:
            continue
        if y < int(image_height * 0.48):
            continue
        if (y + height) / image_height < 0.86:
            continue

        inner_x1 = x + max(4, int(width * 0.08))
        inner_x2 = max(inner_x1 + 1, min(image_width, x + width) - max(4, int(width * 0.03)))
        inner_y1 = y + max(4, int(height * 0.14))
        inner_y2 = max(inner_y1 + 1, min(image_height, y + height) - max(4, int(height * 0.14)))
        interior = gray[inner_y1:inner_y2, inner_x1:inner_x2]
        if float(np.mean(interior > 150)) < 0.35 or float(np.mean(interior)) < 150.0:
            continue

        # パネルの右側は画面外まで続くことが多く、実況者や文字によって
        # マスクが途切れても右端までの領域として評価する。
        region_width = max(width, image_width - x - 2) if x + width > image_width * 0.72 else width
        score = _candidate_score(mask, gray, x, y, region_width, height, config)
        candidates.append((score, x, y, region_width, height))

    if not candidates:
        return None

    score, x, y, width, height = max(candidates, key=lambda candidate: candidate[0])
    if score < 0.30:
        return None

    # 切り出しが画面端を越えないようにする。
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    layout = "full" if x / image_width <= 0.08 else "right"
    return DialogRegion(
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=round(float(_clamp(score)), 4),
        layout=layout,
    )


def text_regions(frame: np.ndarray, dialog: DialogRegion) -> dict[str, np.ndarray]:
    """ダイアログ矩形を話者名と本文に分ける。"""

    x1 = _estimate_text_start(frame, dialog)
    x2 = dialog.right - int(dialog.width * 0.06)
    name_y1 = dialog.y + int(dialog.height * 0.10)
    name_y2 = dialog.y + int(dialog.height * 0.34)
    body_y1 = dialog.y + int(dialog.height * 0.35)
    body_y2 = dialog.y + int(dialog.height * 0.84)

    return {
        "name": frame[name_y1:name_y2, x1:x2],
        "body": frame[body_y1:body_y2, x1:x2],
    }


def _estimate_text_start(frame: np.ndarray, dialog: DialogRegion) -> int:
    """キャラクターで左側が覆われるレイアウトの本文開始位置を推定する。"""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x_min = dialog.x + int(dialog.width * 0.045)
    x_max = dialog.right - int(dialog.width * 0.08)
    y_min = dialog.y + int(dialog.height * 0.14)
    y_max = dialog.y + int(dialog.height * 0.78)
    if x_max <= x_min or y_max <= y_min:
        return x_min

    # 明るいパネル上の濃い文字と、パネルに重なったキャラクターを列方向に
    # まとめる。キャラクターは広い連続領域、文字はその後ろのまとまった
    # 領域になるため、最初の大きな塊を飛ばせる。
    ink = (gray[y_min:y_max, x_min:x_max] < 185).astype(np.float32)
    projection = cv2.blur(np.mean(ink, axis=0)[None, :], (9, 1))[0]
    active = (projection > 0.025).astype(np.uint8)[None, :] * 255
    active = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
    )[0]
    components, labels, stats, _ = cv2.connectedComponentsWithStats(active)
    runs = sorted(
        (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            float(np.max(projection[int(stats[index, cv2.CC_STAT_LEFT]):int(stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH])])),
            float(np.mean(projection[int(stats[index, cv2.CC_STAT_LEFT]):int(stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH])])),
        )
        for index in range(1, components)
        if stats[index, cv2.CC_STAT_WIDTH] >= 6
    )
    if not runs:
        return x_min

    first_left, first_width, _, first_mean = runs[0]
    # 左側の人物や立ち絵は、文字より列占有率が高い。
    if first_mean > 0.25 and first_width >= max(18, int(dialog.width * 0.02)) and len(runs) >= 2:
        return max(x_min, x_min + runs[1][0] - 5)
    return max(x_min, x_min + first_left - 5)


def annotate_frame(frame: np.ndarray, dialog: DialogRegion | None) -> np.ndarray:
    """検出結果を目視確認できる画像を返す。"""

    annotated = frame.copy()
    if dialog is None:
        cv2.putText(
            annotated,
            "dialog: not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    cv2.rectangle(annotated, (dialog.x, dialog.y), (dialog.right, dialog.bottom), (0, 180, 0), 3)
    regions = text_regions(frame, dialog)
    name_x = _estimate_text_start(frame, dialog)
    name_y1 = dialog.y + int(dialog.height * 0.10)
    body_y1 = dialog.y + int(dialog.height * 0.35)
    body_x = name_x
    body_y2 = dialog.y + int(dialog.height * 0.84)
    cv2.rectangle(
        annotated,
        (name_x, name_y1),
        (name_x + regions["name"].shape[1], name_y1 + regions["name"].shape[0]),
        (255, 120, 0),
        2,
    )
    cv2.rectangle(
        annotated,
        (body_x, body_y1),
        (body_x + regions["body"].shape[1], body_y2),
        (255, 0, 200),
        2,
    )
    label = f"dialog: {dialog.layout} {dialog.confidence:.2f}"
    cv2.putText(annotated, label, (dialog.x, max(30, dialog.y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 0), 2, cv2.LINE_AA)
    return annotated
