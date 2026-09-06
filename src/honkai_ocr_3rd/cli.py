from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import process_image, process_video


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="崩壊3rdの会話ウィンドウ検出とOCR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="1枚の画像を検出・OCRする")
    image_parser.add_argument("image", type=_path)
    image_parser.add_argument("--output-dir", type=_path, default=Path("data/output/image"))
    image_parser.add_argument("--no-ocr", action="store_true")

    video_parser = subparsers.add_parser("video", help="動画を間引きながら検出・OCRする")
    video_parser.add_argument("video", type=_path)
    video_parser.add_argument("--output-dir", type=_path, default=Path("data/output/video"))
    video_parser.add_argument("--sample-fps", type=float, default=2.0)
    video_parser.add_argument("--max-seconds", type=float, default=300.0)
    video_parser.add_argument("--no-ocr", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "image":
        result = process_image(args.image, args.output_dir, run_ocr=not args.no_ocr)
    else:
        result = process_video(
            args.video,
            args.output_dir,
            sample_fps=args.sample_fps,
            max_seconds=args.max_seconds,
            run_ocr=not args.no_ocr,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
