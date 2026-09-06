import cv2
import numpy as np

from honkai_ocr_3rd.ocr import _body_line_regions, _normalize_text


def test_normalize_text_joins_spaces_between_japanese_characters() -> None:
    assert _normalize_text("エ リ シ ア\nこ の 色") == "エリシア\nこの色"


def test_body_line_regions_separates_multiple_text_lines() -> None:
    image = np.full((90, 240, 3), 235, dtype=np.uint8)
    for top, bottom in ((5, 25), (31, 51), (57, 77)):
        cv2.rectangle(image, (8, top), (220, bottom), (40, 40, 40), -1)

    regions = _body_line_regions(image)

    assert len(regions) == 3
    assert all(region.shape[0] >= 20 for region in regions)
