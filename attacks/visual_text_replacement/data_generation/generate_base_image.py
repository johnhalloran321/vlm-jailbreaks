#!/usr/bin/env python3
"""
Generate images from a text prompt via an OpenAI-compatible image-gen
endpoint (e.g. GPT-5.2 through the Responses API's image_generation tool).
Builds prompts from CREATE_TEXT_TEMPLATE for text replacement attacks.

This originally called REVE's /v1/image/create endpoint directly. This repo
doesn't have REVE access, so it now goes through the shared
attacks/common/image_gen.py request_image() helper -- see
attacks/visual_object_replacement/data_generation/generate_base_image.py for
the sibling script this mirrors.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.common.llm_client import get_client  # noqa: E402
from attacks.common.image_gen import DEFAULT_MAX_PARALLEL, request_image_with_retry  # noqa: E402
from .prompts import CREATE_TEXT_TEMPLATE  # noqa: E402


def sanitize_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    # Replace problematic characters
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
        "'": "",
        ",": "",
        ".": "_",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    # Remove consecutive underscores and trim
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")[:100]


def build_prompt(target_text: str, extra: Optional[str] = None) -> str:
    """Construct the generation prompt using CREATE_TEXT_TEMPLATE."""
    base_prompt = CREATE_TEXT_TEMPLATE.format(target_text=target_text)
    if extra:
        return f"{base_prompt} {extra}"
    return base_prompt


def save_image_bytes(image_bytes: bytes, output_path: Path) -> Tuple[Path, Image.Image]:
    """Decode raw image bytes and save to disk."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path, image


def generate_image(
    target_text: str,
    client,
    output_path: Path,
    model: str,
    api_style: str = "auto",
    max_retries: int = 3,
) -> Optional[Path]:
    """Generate a single image for the target text.

    Delegates to ``request_image_with_retry()`` (see
    attacks/common/image_gen.py), which honors a server-provided
    ``Retry-After`` header on HTTP 429 rate-limit responses and falls back to
    exponential backoff with jitter otherwise.
    """
    prompt = build_prompt(target_text)

    def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        print(f"    Retry {attempt + 1}/{max_retries} for '{target_text}' after {wait:.1f}s: {exc}")

    try:
        images = request_image_with_retry(
            client,
            model=model,
            prompt=prompt,
            api_style=api_style,
            max_retries=max_retries - 1 if max_retries > 0 else 0,
            on_retry=_on_retry,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    ✗ Error generating image for '{target_text}': {e}")
        return None

    saved_path, _ = save_image_bytes(images[0], output_path)
    return saved_path


def generate_missing_images(
    manifest: Dict[str, List[str]],
    objects: List[str],
    base_dir: Path,
    client=None,
    model: str = None,
    api_style: str = "auto",
    num_images: int = 3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    max_retries: int = 3,
) -> Dict[str, List[str]]:
    """
    Generate images for objects that have fewer than num_images.
    Saves progress incrementally so interrupted runs can be resumed.

    Args:
        manifest: Current manifest mapping object -> list of image paths
        objects: List of all objects to ensure coverage
        base_dir: Base directory for saving images
        client: an openai.OpenAI client (built via attacks.common.llm_client.get_client()
            if not provided, using api_key/base_url overrides)
        model: image-gen model name (defaults to $IMAGE_GEN_MODEL or openai/gpt-5.2)
        api_style: "auto" | "chat" | "responses" (see attacks/common/image_gen.py)
        num_images: Target number of images per object
        max_parallel: number of concurrent worker threads used to fan requests
            out across objects/images (default: DEFAULT_MAX_PARALLEL, i.e. 8).
            Pass 1 to fall back to the old fully-sequential behavior.
        max_retries: retries per image request, passed through to
            ``generate_image()`` / ``request_image_with_retry()``.

    Returns:
        Updated manifest with new image paths
    """
    if client is None:
        client = get_client(api_key=api_key, base_url=base_url)
    if model is None:
        model = os.environ.get("IMAGE_GEN_MODEL", "openai/gpt-5.2")

    updated_manifest: Dict[str, List[str]] = dict(manifest)
    manifest_path = base_dir / "manifest.json"
    manifest_lock = threading.Lock()

    print("\n" + "="*60)
    print("STEP 2b: Generating Missing Images")
    print("="*60)

    # Build the flat list of (object, target_index, output_path) work items
    # across *all* objects up front, so the thread pool below can fan
    # everything out at once instead of only parallelizing within a single
    # object at a time.
    work_items: List[Tuple[str, int, Path]] = []
    for obj in objects:
        current_images = [p for p in updated_manifest.get(obj, []) if Path(p).exists()]
        updated_manifest[obj] = current_images
        needed = num_images - len(current_images)
        if needed <= 0:
            continue

        obj_dir = base_dir / sanitize_filename(obj)
        obj_dir.mkdir(parents=True, exist_ok=True)

        for i in range(needed):
            idx = len(current_images) + i + 1
            filename = f"{sanitize_filename(obj)}_generated_{idx}.jpg"
            output_path = obj_dir / filename
            if output_path.exists():
                current_images.append(str(output_path))
                print(f"  [{obj}] Image {idx}/{num_images} already exists, skipping")
                continue
            work_items.append((obj, idx, output_path))

    print(f"Objects needing generation: {sum(1 for obj in objects if len(updated_manifest.get(obj, [])) < num_images)}")
    print(f"Images to generate: {len(work_items)}")

    if not work_items:
        print("\nTotal images generated: 0")
        return updated_manifest

    total_generated = 0
    total_lock = threading.Lock()

    def _save_manifest_locked() -> None:
        try:
            with open(manifest_path, "w") as f:
                json.dump(updated_manifest, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save manifest: {e}")

    def process_one(item: Tuple[str, int, Path]) -> None:
        nonlocal total_generated
        obj, idx, output_path = item
        print(f"  [{obj}] Generating image {idx}/{num_images}...", flush=True)

        result = generate_image(
            target_text=obj,
            client=client,
            output_path=output_path,
            model=model,
            api_style=api_style,
            max_retries=max_retries,
        )

        with manifest_lock:
            if result:
                updated_manifest.setdefault(obj, []).append(str(result))
            # Save incrementally (under the lock) so interrupted runs can
            # resume, same as the previous sequential-only behavior.
            _save_manifest_locked()

        if result:
            with total_lock:
                total_generated += 1
            print(f"  [{obj}] ✓ Saved: {output_path.name}")
        else:
            print(f"  [{obj}] ✗ Failed")

    workers = max(1, min(max_parallel, len(work_items)))
    if workers <= 1:
        for item in work_items:
            process_one(item)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="TextReplBaseWorker") as pool:
            futures = [pool.submit(process_one, item) for item in work_items]
            for f in futures:
                f.result()

    print(f"\nTotal images generated: {total_generated}")

    return updated_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate images via an OpenAI-compatible image-gen endpoint using CREATE_TEXT_TEMPLATE."
    )
    parser.add_argument(
        "target_text",
        type=str,
        nargs="?",
        default="bomb",
        help="Target text to generate image for (default: bomb).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for image (default: ./tmp/generated_<target_text>.jpg).",
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
        help="API shape to use for image generation (see attacks/common/image_gen.py).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key override (else LLM_API_KEY / etc.).")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL override (else LLM_API_BASE_URL / etc.).")
    # --aspect-ratio / --version / --timeout kept only for CLI compatibility with the
    # old REVE-based invocation; unused now.
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="3:2",
        help="(Unused with the GPT-5.2 image-gen path; kept for CLI compatibility.)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="(Unused with the GPT-5.2 image-gen path; use --model instead.)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="(Unused with the GPT-5.2 image-gen path; kept for CLI compatibility.)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries on failure (default: 3).",
    )

    args = parser.parse_args()

    client = get_client(api_key=args.api_key, base_url=args.base_url)

    output_path = Path(args.output) if args.output else Path(f"./tmp/generated_{sanitize_filename(args.target_text)}.jpg")

    print(f"Generating image for: {args.target_text}")
    print(f"Model: {args.model} (image_api={args.image_api})")
    print(f"Output: {output_path}")

    result = generate_image(
        target_text=args.target_text,
        client=client,
        output_path=output_path,
        model=args.model,
        api_style=args.image_api,
        max_retries=args.max_retries,
    )

    if result:
        print(f"\n✓ Image saved to: {result}")
    else:
        print("\n✗ Failed to generate image")
        sys.exit(1)


if __name__ == "__main__":
    main()
