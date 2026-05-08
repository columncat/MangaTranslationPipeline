"""Standalone OCR debug — run a single image through manga-ocr in two ways
and dump everything so we can see why nothing is detected.

Usage (from project root):
    python scripts/debug_ocr.py models/cache/ocr_crops/bbox_000_xxx.png
    python scripts/debug_ocr.py models/cache/ocr_crops/bbox_000_xxx.png --cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a crop PNG")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"file not found: {args.image}", file=sys.stderr)
        return 2

    img = Image.open(args.image)
    print(f"input: size={img.size} mode={img.mode}")

    print("\n--- importing manga_ocr ---")
    from manga_ocr import MangaOcr
    from manga_ocr.ocr import post_process

    import transformers
    import torch

    print(f"transformers={transformers.__version__} torch={torch.__version__}")

    print("\n--- loading model (first run downloads ~400MB) ---")
    mocr = MangaOcr(force_cpu=args.cpu)
    print(f"model device: {mocr.model.device}")
    print(
        f"cls={mocr.tokenizer.cls_token_id} sep={mocr.tokenizer.sep_token_id} "
        f"pad={mocr.tokenizer.pad_token_id} eos={mocr.tokenizer.eos_token_id}"
    )

    print("\n--- attempt 1: standard MangaOcr.__call__ ---")
    try:
        result = mocr(img)
        print(f"result: {result!r}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- attempt 2: explicit generate with decoder_start_token_id ---")
    img_rgb = img.convert("L").convert("RGB")
    pixel_values = mocr.feature_extractor(img_rgb, return_tensors="pt").pixel_values.to(
        mocr.model.device
    )
    output_ids = mocr.model.generate(
        pixel_values,
        max_length=300,
        num_beams=1,
        decoder_start_token_id=mocr.tokenizer.cls_token_id,
        eos_token_id=mocr.tokenizer.sep_token_id or mocr.tokenizer.eos_token_id,
        pad_token_id=mocr.tokenizer.pad_token_id,
    )[0].cpu()

    print(f"raw token ids ({len(output_ids)}): {output_ids.tolist()[:30]}...")
    print(f"decoded (skip_special=True):  {mocr.tokenizer.decode(output_ids, skip_special_tokens=True)!r}")
    print(f"decoded (skip_special=False): {mocr.tokenizer.decode(output_ids, skip_special_tokens=False)!r}")
    print(f"post_processed: {post_process(mocr.tokenizer.decode(output_ids, skip_special_tokens=True))!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
