import cv2
import numpy as np

import honkai_ocr_3rd.ocr as ocr
from honkai_ocr_3rd.ocr import (
    OCRConfig,
    _body_line_content_region,
    _body_line_regions,
    _name_content_region,
    _normalize_text,
    recognize,
)


def test_normalize_text_joins_spaces_between_japanese_characters() -> None:
    assert _normalize_text("エ リ シ ア\nこ の 色") == "エリシア\nこの色"


def test_body_line_regions_separates_multiple_text_lines() -> None:
    image = np.full((90, 240, 3), 235, dtype=np.uint8)
    for top, bottom in ((5, 25), (31, 51), (57, 77)):
        cv2.rectangle(image, (8, top), (220, bottom), (40, 40, 40), -1)

    regions = _body_line_regions(image)

    assert len(regions) == 3
    assert all(region.shape[0] >= 20 for region in regions)


def test_name_content_region_removes_wide_empty_margin() -> None:
    image = np.full((45, 240, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (12, 8), (78, 28), (70, 70, 70), -1)

    region = _name_content_region(image)

    assert region.shape[1] < image.shape[1] / 2


def test_body_line_content_region_removes_background_side() -> None:
    image = np.full((30, 240, 3), 245, dtype=np.uint8)
    image[:, :45] = 60
    cv2.rectangle(image, (80, 6), (210, 22), (70, 70, 70), -1)

    region = _body_line_content_region(image)

    assert region.shape[1] < image.shape[1]
    assert region.shape[1] > 100


def test_paddle_result_is_used_when_configured(monkeypatch) -> None:
    class Prediction:
        json = {"res": {"rec_text": "だけど今は……", "rec_score": 0.98}}

    class Model:
        def predict(self, images, batch_size):
            yield Prediction()

    monkeypatch.setattr(ocr, "_PADDLE_MODEL", Model())
    monkeypatch.setattr(ocr, "_PADDLE_UNAVAILABLE", False)
    image = np.full((40, 160, 3), 245, dtype=np.uint8)

    result = recognize(image, "body", OCRConfig(engine="paddle"))

    assert result.text == "だけど今は......"
    assert result.variant == "paddle"
