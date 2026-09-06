from honkai_ocr_3rd.pipeline import _is_weak_text, merge_events


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


def test_merge_events_handles_ocr_variation_with_shared_text() -> None:
    merged = merge_events([
        event(10.0, 12.0, "メビウスに関することが知りたい。特に彼女の生と死に関してよ。他の英傑たちの言い方だと"),
        event(12.5, 14.0, "ご握蕩呈繧嘉暮惠言壹。特に彼女の生と死に関してよ"),
    ])

    assert len(merged) == 1
    assert merged[0]["text"].startswith("メビウスに関すること")


def test_is_weak_text_rejects_transition_noise() -> None:
    assert _is_weak_text("さャオ")
    assert _is_weak_text("囲閻. 1 ` る e e ぁぁ団謀誌ー _ B")
    assert not _is_weak_text("あなたなら、分かるでしょう?")
