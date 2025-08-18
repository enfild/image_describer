import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from config import Config
from utils import save_result
from pipelines.vit_gpt2 import ViTGPT2Pipeline
from pipelines.blip import BLIPPipeline
from pipelines.llava import LLavaPipeline

import torch
import gc

IMG_EXT = {".jpg", ".jpeg", ".png"}


def iter_images(root: Path):
    if root.is_file() and root.suffix.lower() in IMG_EXT:
        yield root
        return
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                yield p


def build_pipeline(name: str):
    name = name.lower()
    if name in ("vitgpt2"):
        model_name = getattr(Config, "VITGPT2_MODEL_NAME", "nlpconnect/vit-gpt2-image-captioning")
        return ViTGPT2Pipeline(model_name=model_name)
    if name == "blip":
        return BLIPPipeline()
    if name == "llava":
        model_name = getattr(Config, "LLAVA_MODEL_NAME", "llava-hf/llava-1.5-7b-hf")
        return LLavaPipeline(model_name=model_name)
    raise ValueError("Unknown model (blip|vitgpt2|llava)")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Image captioning CLI (single file or folder)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image file OR folder with images")
    parser.add_argument("--model", type=str, nargs="+",
                        choices=["blip", "vitgpt2", "llava"],
                        default=["blip"],
                        help="One or several models to run (space-separated)")
    parser.add_argument("--output", type=str, default="results",
                        help="Optional path to save combined results JSON")
    args = parser.parse_args()

    src = Path(args.input)
    images = list(iter_images(src))
    if not images:
        print("No images found with allowed extensions.")
        return

    results: List[Dict[str, Any]] = [{"image": str(p), "results": {}} for p in images]

    for model_name in args.model:
        print(f"\n=== Running model: {model_name} on {len(images)} images ===")
        try:
            pipe = build_pipeline(model_name)
            for idx, img_path in enumerate(images):
                try:
                    caps = pipe.generate(
                        str(img_path),
                        num_return_sequences=Config.NUM_CAPTIONS,
                        max_new_tokens=Config.MAX_TOKENS,
                        temperature=Config.TEMPERATURE,
                    )
                    save_result(str(img_path), model_name, caps)
                    results[idx]["results"][model_name] = caps
                    print(f"{img_path} [{model_name}]")
                    for i, c in enumerate(caps, 1):
                        print(f"  {i}. {c}")
                except Exception as e:
                    print(f"[ERROR] {img_path} ({model_name}): {e}")
            del pipe
            free_memory()
        except Exception as e:
            print(f"[MODEL ERROR] {model_name}: {e}")
            free_memory()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved combined results to {out}")


if __name__ == "__main__":
    main()
