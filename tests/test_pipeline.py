from honkai_ocr_3rd.pipeline import merge_events


def event(start: float, end: float, text: str, speaker: str = "芽衣") -> dict:
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "text": text,
        "confidence": 70.0,
        "needs_review": False,
        "event_files": {},
    }


def test_merge_events_prefers_completed_typewriter_text() -> None:
    merged = merge_events([
        event(10.0, 10.0, "ここにいないなん"),
        event(12.0, 12.0, "ここにいないなんて、珍しいわね。"),
    ])

    assert len(merged) == 1
    assert merged[0]["text"] == "ここにいないなんて、珍しいわね。"
    assert merged[0]["start"] == 10.0
    assert merged[0]["end"] == 12.0


def test_merge_events_keeps_different_lines_separate() -> None:
    merged = merge_events([
        event(10.0, 10.0, "ここにいたのね。"),
        event(12.0, 12.0, "大丈夫、立ったままでいいわ。"),
    ])

    assert len(merged) == 2
