from __future__ import annotations

import json
import html
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2

from .detector import annotate_frame, detect_dialog, text_regions
from .models import DialogRegion, OCRResult
from .ocr import OCRConfig, recognize


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _compact_text(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _is_weak_text(text: str) -> bool:
    compact = _compact_text(text)
    japanese = sum("\u3040" <= character <= "\u9fff" for character in compact)
    noise = sum(
        character.isascii() and (character.isalnum() or character in "_`~|\\")
        for character in compact
    )
    return len(compact) < 3 or japanese < 4 or (noise >= 3 and noise > japanese * 0.25)


def _is_continuation(left: str, right: str) -> bool:
    if not left or not right or _is_weak_text(left) or _is_weak_text(right):
        return True
    left_compact = _compact_text(left)
    right_compact = _compact_text(right)
    if left_compact in right_compact or right_compact in left_compact:
        return True
    return _similar(left_compact, right_compact) >= 0.44


def _event_quality(event: dict[str, Any]) -> float:
    text = str(event.get("text", ""))
    compact = _compact_text(text)
    japanese = sum("\u3040" <= character <= "\u9fff" for character in compact)
    non_japanese = sum(character.isascii() and not character.isspace() for character in compact)
    confidence = float(event.get("confidence", 0.0))
    return japanese * 1.5 + min(len(compact), 120) * 0.25 + confidence * 0.05 - non_japanese * 0.4


def _needs_review(event: dict[str, Any]) -> bool:
    text = str(event.get("text", ""))
    compact = _compact_text(text)
    noise = sum(
        character.isascii() and (character.isalnum() or character in "_`~|\\")
        for character in compact
    )
    return not text or float(event.get("confidence", 0.0)) < 60 or noise >= 3


def merge_events(events: list[dict[str, Any]], max_gap: float = 4.5) -> list[dict[str, Any]]:
    """文字送り途中の候補を、同一台詞の代表イベントへまとめる。"""

    merged: list[dict[str, Any]] = []
    for event in events:
        if not merged:
            event["samples"] = 1
            merged.append(event)
            continue

        previous = merged[-1]
        gap = float(event["start"]) - float(previous["end"])
        previous_speaker = str(previous.get("speaker", ""))
        current_speaker = str(event.get("speaker", ""))
        speaker_matches = (
            not previous_speaker
            or not current_speaker
            or previous_speaker == current_speaker
            or _similar(_compact_text(previous_speaker), _compact_text(current_speaker)) >= 0.70
        )
        if gap <= max_gap and speaker_matches and _is_continuation(str(previous.get("text", "")), str(event.get("text", ""))):
            previous["end"] = event["end"]
            previous["samples"] = int(previous.get("samples", 1)) + 1
            previous["needs_review"] = bool(previous.get("needs_review")) or bool(event.get("needs_review"))
            if _event_quality(event) > _event_quality(previous):
                for key in ("speaker", "text", "confidence", "dialog", "layout", "source_frame", "event_files"):
                    if key in event:
                        previous[key] = event[key]
        else:
            event["samples"] = 1
            merged.append(event)
    return merged


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _relative_output_path(path: str, output_dir: Path) -> str:
    try:
        return Path(path).relative_to(output_dir).as_posix()
    except ValueError:
        return Path(path).name


def _write_review_html(output_dir: Path, events: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for event in events:
        files = event["event_files"]
        frame = html.escape(_relative_output_path(files["frame"], output_dir), quote=True)
        name = html.escape(_relative_output_path(files["name"], output_dir), quote=True)
        body = html.escape(_relative_output_path(files["body"], output_dir), quote=True)
        speaker = html.escape(str(event.get("speaker", "")))
        text = html.escape(str(event.get("text", "")))
        review_class = " review" if event.get("needs_review") else ""
        cards.append(
            f"""<article class=\"card{review_class}\">
  <img class=\"frame\" src=\"{frame}\" alt=\"検出フレーム\">
  <div class=\"meta\">{event["start"]:.3f}s – {event["end"]:.3f}s / {event["layout"]} / confidence {event["confidence"]}</div>
  <div class=\"crops\"><img src=\"{name}\" alt=\"話者名\"><img src=\"{body}\" alt=\"本文\"></div>
  <div class=\"speaker\">{speaker}</div>
  <div class=\"text\">{text}</div>
</article>"""
        )
    document = f"""<!doctype html>
<meta charset=\"utf-8\">
<title>honkai-ocr review</title>
<style>
body {{ background:#202124; color:#eee; font-family:sans-serif; margin:1rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:1rem; }}
.card {{ background:#303134; border:2px solid #555; border-radius:8px; padding:.6rem; }}
.card.review {{ border-color:#e6a23c; }}
.frame {{ width:100%; display:block; }}
.meta {{ color:#aaa; font-size:.8rem; margin:.4rem 0; }}
.crops {{ display:grid; gap:.3rem; grid-template-columns:1fr 1fr; }}
.crops img {{ background:#fff; width:100%; max-height:150px; object-fit:contain; }}
.speaker {{ color:#ffb6e6; font-weight:bold; margin-top:.4rem; }}
.text {{ white-space:pre-wrap; line-height:1.5; }}
</style>
<main class=\"grid\">{''.join(cards)}</main>
"""
    (output_dir / "review.html").write_text(document, encoding="utf-8")


def process_image(image_path: Path, output_dir: Path, run_ocr: bool = True) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dialog = detect_dialog(image)
    annotated_path = output_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), annotate_frame(image, dialog))
    result: dict[str, Any] = {
        "source": str(image_path),
        "annotated": str(annotated_path),
        "dialog": dialog.as_dict() if dialog else None,
    }
    if dialog is None:
        _write_json(output_dir / "result.json", result)
        return result

    regions = text_regions(image, dialog)
    crops: dict[str, str] = {}
    ocr_results: dict[str, dict[str, Any]] = {}
    for kind, crop in regions.items():
        crop_path = output_dir / f"{kind}.png"
        cv2.imwrite(str(crop_path), crop)
        crops[kind] = str(crop_path)
        if run_ocr:
            ocr_results[kind] = recognize(crop, kind).as_dict()
    result["crops"] = crops
    if run_ocr:
        result["ocr"] = ocr_results
    _write_json(output_dir / "result.json", result)
    return result


def _save_event_frame(
    frame: Any,
    output_dir: Path,
    event_number: int,
    dialog: DialogRegion,
) -> dict[str, str]:
    frame_dir = output_dir / "events"
    frame_dir.mkdir(parents=True, exist_ok=True)
    prefix = frame_dir / f"{event_number:04d}"
    annotated_path = prefix.with_name(prefix.name + "_annotated.jpg")
    cv2.imwrite(str(annotated_path), annotate_frame(frame, dialog))
    paths = {"frame": str(annotated_path)}
    for kind, crop in text_regions(frame, dialog).items():
        crop_path = prefix.with_name(prefix.name + f"_{kind}.png")
        cv2.imwrite(str(crop_path), crop)
        paths[kind] = str(crop_path)
    return paths


def process_video(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 2.0,
    max_seconds: float | None = 300.0,
    run_ocr: bool = True,
) -> list[dict[str, Any]]:
    """動画を間引きながら処理し、台詞候補を時系列にまとめる。"""

    if sample_fps <= 0:
        raise ValueError("sample_fps は正数にしてください")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"動画を開けません: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if frame_count else None
    interval = max(1, round(source_fps / sample_fps))
    max_frame = int(max_seconds * source_fps) if max_seconds is not None else None

    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_regions: dict[str, Any] | None = None
    previous_results: tuple[OCRResult, OCRResult] | None = None
    frame_index = 0
    sampled = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_frame is not None and frame_index >= max_frame:
            break
        if frame_index % interval != 0:
            frame_index += 1
            continue

        timestamp = frame_index / source_fps
        sampled += 1
        dialog = detect_dialog(frame)
        if dialog is None:
            previous_regions = None
            previous_results = None
            frame_index += 1
            continue

        regions = text_regions(frame, dialog)
        can_reuse = (
            run_ocr
            and previous_regions is not None
            and previous_results is not None
            and all(
                region.shape == previous_regions[kind].shape
                and float(cv2.absdiff(region, previous_regions[kind]).mean()) < 1.5
                for kind, region in regions.items()
            )
        )
        if can_reuse:
            name_result, body_result = previous_results
        else:
            name_result = recognize(regions["name"], "name") if run_ocr else OCRResult("", 0.0, "disabled")
            body_result = recognize(regions["body"], "body") if run_ocr else OCRResult("", 0.0, "disabled")
            if run_ocr:
                previous_regions = {kind: region.copy() for kind, region in regions.items()}
                previous_results = (name_result, body_result)
        text = body_result.text
        if run_ocr and _is_weak_text(text):
            body_result = OCRResult("", 0.0, "weak")
            text = ""
        speaker = name_result.text
        is_same_event = (
            current is not None
            and timestamp - float(current["last_seen"]) <= max(1.5, 3.0 / sample_fps)
            and (_similar(current["text"], text) >= 0.62 or not current["text"] or not text)
        )

        if not is_same_event:
            if current is not None:
                events.append(current)
            event_number = len(events) + 1
            paths = _save_event_frame(frame, output_dir, event_number, dialog)
            current = {
                "start": round(timestamp, 3),
                "end": round(timestamp, 3),
                "last_seen": timestamp,
                "speaker": speaker,
                "text": text,
                "speaker_confidence": round(name_result.confidence, 2),
                "confidence": round(min(dialog.confidence * 100, body_result.confidence), 2),
                "layout": dialog.layout,
                "dialog": dialog.as_dict(),
                "source_frame": frame_index,
                "needs_review": dialog.confidence < 0.55 or body_result.confidence < 55 or not text,
                "event_files": paths,
            }
        else:
            current["end"] = round(timestamp, 3)
            current["last_seen"] = timestamp
            # 同じ台詞の中では、OCR信頼度が高いフレームを代表値にする。
            if body_result.confidence > float(current["confidence"]):
                current["speaker"] = speaker or current["speaker"]
                current["text"] = text or current["text"]
                current["confidence"] = round(body_result.confidence, 2)
                current["source_frame"] = frame_index

        frame_index += 1

    if current is not None:
        events.append(current)
    capture.release()

    for event in events:
        event.pop("last_seen", None)
    metadata = {
        "video": str(video_path),
        "source_fps": source_fps,
        "duration": duration,
        "sample_fps": sample_fps,
        "sampled_frames": sampled,
        "max_seconds": max_seconds,
    }
    events = merge_events(events)
    for event in events:
        event["needs_review"] = bool(event.get("needs_review")) or _needs_review(event)
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "events.json", events)
    _write_review_html(output_dir, events)
    return events
