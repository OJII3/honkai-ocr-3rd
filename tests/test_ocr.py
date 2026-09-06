from honkai_ocr_3rd.ocr import _normalize_text


def test_normalize_text_joins_spaces_between_japanese_characters() -> None:
    assert _normalize_text("エ リ シ ア\nこ の 色") == "エリシア\nこの色"
