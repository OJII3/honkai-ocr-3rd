from __future__ import annotations

import csv
import io
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from .models import OCRResult


@dataclass(frozen=True)
class OCRConfig:
    language: str = "jpn"
    speaker_psm: int = 7
    body_psm: int = 6
    upscale: int = 3


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    japanese = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef"
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"^[|_`~]+|[|_`~]+$", "", line).strip()
        # Tesseractは日本語を1文字ずつ空白で区切ることがある。
        line = re.sub(rf"(?<=[{japanese}]) +(?=[{japanese}])", "", line)
        line = re.sub(r"\s+([、。！？…」』）】〉》,.!?])", r"\1", line)
        line = re.sub(r"([「『（【〈《])\s+", r"\1", line)
        lines.append(line)
    return "\n".join(line for line in lines if line)


def _variants(image: np.ndarray, upscale: int) -> list[tuple[str, np.ndarray]]:
    if image.size == 0:
        return []

    enlarged = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    # 青い文字を黒、背景を白にした補助画像。色が変わった場合でも、
    # グレースケール系の候補は残る。
    hsv = cv2.cvtColor(enlarged, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (80, 35, 15), (145, 255, 240))
    color_mask = np.full_like(blue, 255)
    color_mask[blue > 0] = 0

    return [
        ("gray", gray),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("blue", color_mask),
    ]


def _body_line_regions(image: np.ndarray) -> list[np.ndarray]:
    if image.size == 0 or image.ndim != 3:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_fraction = np.mean(gray < 160, axis=1)
    height = image.shape[0]
    kernel_size = max(5, int(round(height * 0.10)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.blur(dark_fraction.reshape(-1, 1), (1, kernel_size)).ravel()
    radius = max(2, kernel_size // 2)
    peak_floor = max(0.02, float(smoothed.max()) * 0.12)
    peaks = [
        (float(smoothed[row]), row)
        for row in range(radius, height - radius)
        if smoothed[row] >= float(np.max(smoothed[row - radius:row + radius + 1]))
        and smoothed[row] >= peak_floor
    ]
    if not peaks:
        return []

    minimum_distance = max(10, int(round(height * 0.19)))
    selected: list[tuple[float, int]] = []
    for peak in sorted(peaks, key=lambda value: (value[0], -value[1]), reverse=True):
        if all(abs(peak[1] - other[1]) >= minimum_distance for other in selected):
            selected.append(peak)
    centers = sorted(row for _, row in selected[:4])
    if len(centers) < 2:
        return []

    boundaries = [0]
    boundaries.extend((left + right) // 2 for left, right in zip(centers, centers[1:]))
    active_threshold = max(0.02, float(smoothed.max()) * 0.08)
    active_rows = np.flatnonzero(dark_fraction >= active_threshold)
    last_text_row = int(active_rows[-1]) if len(active_rows) else height - 1
    boundaries.append(min(height, last_text_row + 1))
    padding = max(4, int(round(height * 0.05)))
    regions: list[np.ndarray] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        y1 = max(0, start if index == 0 else start + padding // 2)
        y2 = min(height, end + padding)
        if y2 > y1:
            regions.append(image[y1:y2])
    return regions


def _run_tesseract(image: np.ndarray, psm: int, language: str) -> tuple[str, float]:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return "", 0.0

    command = [
        "tesseract",
        "stdin",
        "stdout",
        "-l",
        language,
        "--psm",
        str(psm),
        "-c",
        "preserve_interword_spaces=1",
        "-c",
        "tessedit_create_tsv=1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=os.environ.copy(),
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "", 0.0

    if completed.returncode != 0:
        return "", 0.0

    output = completed.stdout.decode("utf-8", errors="replace")
    rows = csv.DictReader(io.StringIO(output), delimiter="\t")
    words: list[tuple[int, str]] = []
    confidences: list[float] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if text and confidence >= 0:
            words.append((int(row.get("line_num") or 0), text))
            confidences.append(confidence)

    if not words:
        return "", 0.0

    grouped: list[str] = []
    current_line = words[0][0]
    current_words: list[str] = []
    for line_number, word in words:
        if line_number != current_line:
            grouped.append(" ".join(current_words))
            current_line = line_number
            current_words = []
        current_words.append(word)
    grouped.append(" ".join(current_words))
    return _normalize_text("\n".join(grouped)), float(np.mean(confidences))


def recognize(image: np.ndarray, kind: str, config: OCRConfig | None = None) -> OCRResult:
    config = config or OCRConfig()
    if kind == "name":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray < 185)) < 0.008:
            return OCRResult(text="", confidence=0.0, variant="empty")
    psm = config.speaker_psm if kind == "name" else config.body_psm
    candidates: list[OCRResult] = []
    variants = _variants(image, config.upscale)
    if kind == "name":
        # 話者名は短く単純なため、最も情報を残す候補だけで十分。
        variants = [(variant, prepared) for variant, prepared in variants if variant == "gray"]

    if kind == "body":
        line_results: list[OCRResult] = []
        for line in _body_line_regions(image):
            line_candidates: list[OCRResult] = []
            for variant, prepared in _variants(line, config.upscale):
                if variant not in {"gray", "otsu"}:
                    continue
                text, confidence = _run_tesseract(prepared, 13, config.language)
                if text:
                    line_candidates.append(OCRResult(text=text, confidence=confidence, variant=variant))
            if not line_candidates:
                line_results = []
                break
            best = max(line_candidates, key=lambda result: result.confidence)
            otsu = next((result for result in line_candidates if result.variant == "otsu"), None)
            if otsu is not None and otsu.confidence + 1.5 >= best.confidence:
                best = otsu
            line_results.append(best)
        if len(line_results) >= 2:
            return OCRResult(
                text="\n".join(result.text for result in line_results),
                confidence=float(np.mean([result.confidence for result in line_results])),
                variant="lines",
            )

    for variant, prepared in variants:
        text, confidence = _run_tesseract(prepared, psm, config.language)
        if text:
            candidates.append(OCRResult(text=text, confidence=confidence, variant=variant))

    if not candidates:
        return OCRResult(text="", confidence=0.0, variant="none")

    # 認識率を第一にしつつ、空に近い誤結果を避ける。
    length_bonus = 0.10 if kind == "body" else 0.0
    best = max(
        candidates,
        key=lambda result: (
            result.confidence + length_bonus * min(len(result.text), 120),
            result.confidence,
        ),
    )
    if kind == "name" and len(re.sub(r"\s", "", best.text)) < 2:
        return OCRResult(text="", confidence=0.0, variant=best.variant)
    return best
