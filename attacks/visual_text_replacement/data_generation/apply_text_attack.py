#!/usr/bin/env python3
"""
Apply text replacement attack to downloaded reference images.
Reads from manifest.json and applies an image edit (via an OpenAI-compatible
image-gen endpoint) to replace text.

This originally called REVE's /v1/image/edit endpoint directly. This repo
doesn't have REVE access, so it now goes through the shared
attacks/common/image_gen.py request_image_edit() helper, which sends the
source image + edit instruction as multimodal input to either:
  - the Responses API's image_generation tool (required for OpenAI GPT-5.x), or
  - legacy chat.completions with image_url content parts (Gemini-style),
picked automatically from --model, or forced via --image-api.

Usage:
    python apply_text_attack.py --base-dir ./data/visual_text_replacement/base \
        --output-dir ./data/visual_text_replacement/attacks \
        --replacement banana
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.common.llm_client import get_client  # noqa: E402
from attacks.common.image_gen import DEFAULT_MAX_PARALLEL, request_image_edit_with_retry  # noqa: E402
from .prompts import EDIT_TEXT_TEMPLATE  # noqa: E402


def sanitize_filename(name: str) -> str:
    """Convert string to safe filename."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name[:100].strip('_')


def save_image_bytes(image_bytes: bytes, output_path: Path) -> None:
    """Decode raw image bytes and save to disk."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def apply_text_replacement(
    source_image: Path,
    original_text: str,
    replacement_text: str,
    output_path: Path,
    client,
    model: str,
    api_style: str = "auto",
    max_retries: int = 3,
) -> bool:
    """Apply text replacement attack to a single image.

    Delegates to ``request_image_edit_with_retry()`` (see
    attacks/common/image_gen.py), which honors a server-provided
    ``Retry-After`` header on HTTP 429 rate-limit responses and falls back to
    exponential backoff with jitter otherwise.
    """

    # Build the edit prompt
    prompt = EDIT_TEXT_TEMPLATE.format(
        original_text=original_text,
        replacement_text=replacement_text,
    )

    def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        print(f"      Retry {attempt + 1}/{max_retries} after {wait:.1f}s: {exc}")

    try:
        images = request_image_edit_with_retry(
            client,
            model=model,
            prompt=prompt,
            images=source_image,
            api_style=api_style,
            max_retries=max_retries - 1 if max_retries > 0 else 0,
            on_retry=_on_retry,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    ✗ Error: {e}")
        return False

    save_image_bytes(images[0], output_path)
    print(f"    ✓ Saved: {output_path.name}")
    return True


def process_object(
    object_name: str,
    image_paths: List[str],
    base_dir: Path,
    output_dir: Path,
    replacement: str,
    client=None,
    model: str = None,
    api_style: str = "auto",
    redo_existing: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
) -> Dict[str, List[str]]:
    """Process all images for a single object.

    Requests for the object's images are fanned out across up to
    ``max_parallel`` worker threads (default: DEFAULT_MAX_PARALLEL, i.e. 8).
    Pass ``max_parallel=1`` to fall back to the old fully-sequential
    behavior.
    """

    if client is None:
        client = get_client(api_key=api_key, base_url=base_url)
    if model is None:
        model = os.environ.get("IMAGE_GEN_MODEL", "openai/gpt-5.2")

    print(f"\n=== {object_name} ===")

    if not image_paths:
        print("  No images to process")
        return {"object": object_name, "attacked": [], "failed": []}

    # Create output directory
    attack_dir = output_dir / sanitize_filename(object_name) / "text_replacement" / replacement
    attack_dir.mkdir(parents=True, exist_ok=True)

    # Resolve source paths and filter out ones we can't find up front, same
    # as the old sequential loop did.
    work_items: List[Tuple[str, Path, Path]] = []
    attacked: List[str] = []
    failed: List[str] = []

    for img_path_str in image_paths:
        # Handle both relative and absolute paths
        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            # Try relative to base_dir parent
            img_path = base_dir.parent.parent / img_path_str

        if not img_path.exists():
            # Try relative to current working directory
            img_path = Path(img_path_str)
            if not img_path.exists():
                print(f"  ✗ Image not found: {img_path_str}")
                failed.append(img_path_str)
                continue

        # Output filename
        output_name = f"{img_path.stem}_attack_{replacement}.png"
        output_path = attack_dir / output_name

        if output_path.exists() and not redo_existing:
            print(f"  Skipping existing: {output_name}")
            attacked.append(str(output_path))
            continue

        work_items.append((img_path_str, img_path, output_path))

    if not work_items:
        return {"object": object_name, "attacked": attacked, "failed": failed}

    results_lock = threading.Lock()

    def process_one(item: Tuple[str, Path, Path]) -> None:
        img_path_str, img_path, output_path = item
        print(f"  Processing: {img_path.name}")

        ok = apply_text_replacement(
            source_image=img_path,
            original_text=object_name,
            replacement_text=replacement,
            output_path=output_path,
            client=client,
            model=model,
            api_style=api_style,
        )

        with results_lock:
            if ok:
                attacked.append(str(output_path))
            else:
                failed.append(img_path_str)

    workers = max(1, min(max_parallel, len(work_items)))
    if workers <= 1:
        for item in work_items:
            process_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="TextReplAttackWorker") as pool:
            futures = [pool.submit(process_one, item) for item in work_items]
            for f in futures:
                f.result()

    return {"object": object_name, "attacked": attacked, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Apply text replacement attack to reference images"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Directory containing base images and manifest.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for attacked images"
    )
    parser.add_argument(
        "--replacement",
        type=str,
        default="banana",
        help="Replacement text (default: banana)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("IMAGE_GEN_MODEL", "openai/gpt-5.2"),
        help="Image-gen model name (default: $IMAGE_GEN_MODEL or openai/gpt-5.2).",
    )
    parser.add_argument(
        "--image-api",
        type=str,
        default="auto",
        choices=["auto", "chat", "responses"],
        help="API shape to use for image editing (see attacks/common/image_gen.py).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key override (else LLM_API_KEY / etc.).")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL override (else LLM_API_BASE_URL / etc.).")
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Reprocess existing attacked images"
    )
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        help="Specific objects to process (default: all)"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help=f"Number of concurrent worker threads per object (default {DEFAULT_MAX_PARALLEL}). Use 1 for sequential.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    client = get_client(api_key=args.api_key, base_url=args.base_url)

    # Load manifest
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found in {base_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter objects if specified
    if args.objects:
        manifest = {k: v for k, v in manifest.items() if k in args.objects}

    print(f"Processing {len(manifest)} objects")
    print(f"Replacement text: {args.replacement}")
    print(f"Output directory: {output_dir}")

    all_results = []

    for object_name, image_paths in manifest.items():
        result = process_object(
            object_name=object_name,
            image_paths=image_paths,
            base_dir=base_dir,
            output_dir=output_dir,
            replacement=args.replacement,
            client=client,
            model=args.model,
            api_style=args.image_api,
            redo_existing=args.redo_existing,
            max_parallel=args.max_parallel,
        )
        all_results.append(result)

    # Save attack manifest
    attack_manifest = {r["object"]: r["attacked"] for r in all_results}
    attack_manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(attack_manifest_path, "w") as f:
        json.dump(attack_manifest, f, indent=2)

    # Summary
    total_attacked = sum(len(r["attacked"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)

    print(f"\n=== Summary ===")
    print(f"Successfully attacked: {total_attacked}")
    print(f"Failed: {total_failed}")
    print(f"Attack manifest saved to: {attack_manifest_path}")


if __name__ == "__main__":
    main()
