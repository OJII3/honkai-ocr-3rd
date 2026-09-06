import cv2
import numpy as np

from honkai_ocr_3rd.detector import _remove_advance_arrow, detect_dialog, text_regions


def synthetic_dialog() -> np.ndarray:
    frame = np.full((720, 1280, 3), (65, 55, 75), dtype=np.uint8)
    # 実画面に近い、右端まで続く淡色の会話パネル。
    points = np.array([[300, 515], [350, 500], [1278, 500], [1278, 705], [320, 705]], np.int32)
    cv2.fillPoly(frame, [points], (225, 225, 225))
    cv2.polylines(frame, [points], True, (245, 245, 245), 5)
    cv2.putText(frame, "speaker", (360, 555), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 110, 170), 2)
    cv2.putText(frame, "body text", (360, 610), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 110, 170), 2)
    return frame


def test_detect_dialog_finds_bottom_panel() -> None:
    dialog = detect_dialog(synthetic_dialog())

    assert dialog is not None
    assert dialog.x < 380
    assert dialog.y < 540
    assert dialog.right > 1200
    assert dialog.height > 100


def test_text_regions_follow_detected_panel() -> None:
    frame = synthetic_dialog()
    dialog = detect_dialog(frame)

    assert dialog is not None
    regions = text_regions(frame, dialog)
    assert regions["name"].shape[0] > 0
    assert regions["body"].shape[0] > regions["name"].shape[0]


def test_bright_panel_above_bottom_is_not_a_dialog() -> None:
    frame = np.full((720, 1280, 3), (65, 55, 75), dtype=np.uint8)
    cv2.rectangle(frame, (60, 390), (520, 585), (225, 225, 225), -1)

    assert detect_dialog(frame) is None


def test_remove_advance_arrow_masks_only_bottom_right_indicator() -> None:
    body = np.full((100, 500, 3), 220, dtype=np.uint8)
    cv2.fillPoly(body, [np.array([[470, 65], [490, 65], [480, 83]], np.int32)], (20, 20, 20))
    cv2.rectangle(body, (30, 25), (120, 35), (20, 20, 20), -1)

    cleaned = _remove_advance_arrow(body)

    assert int(cleaned[74, 480, 0]) > 200
    assert int(cleaned[30, 60, 0]) < 50
