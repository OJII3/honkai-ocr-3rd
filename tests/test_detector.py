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


def test_detect_dialog_normalizes_high_resolution_coordinates() -> None:
    high_resolution = synthetic_dialog()
    low_resolution = cv2.resize(high_resolution, (640, 360), interpolation=cv2.INTER_AREA)

    low_dialog = detect_dialog(low_resolution)
    high_dialog = detect_dialog(high_resolution)

    assert low_dialog is not None
    assert high_dialog is not None
    assert abs(high_dialog.x - low_dialog.x * 2) <= 2
    assert abs(high_dialog.y - low_dialog.y * 2) <= 2
    assert abs(high_dialog.width - low_dialog.width * 2) <= 2
    assert abs(high_dialog.height - low_dialog.height * 2) <= 2


def test_detect_dialog_rejects_narrow_bright_decoy() -> None:
    frame = np.full((360, 640, 3), (45, 70, 80), dtype=np.uint8)
    panel = np.array([[70, 264], [82, 255], [639, 255], [639, 350], [80, 350]], np.int32)
    cv2.fillPoly(frame, [panel], (225, 225, 225))
    cv2.polylines(frame, [panel], True, (245, 245, 245), 3)
    cv2.rectangle(frame, (220, 235), (420, 310), (160, 220, 240), -1)

    dialog = detect_dialog(frame)

    assert dialog is not None
    assert dialog.x < 120
    assert dialog.right > 600
    assert dialog.y > 220


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
