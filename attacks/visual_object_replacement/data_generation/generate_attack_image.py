#!/usr/bin/env python3
"""
Generate an edited (attack) image via an OpenAI-compatible image-gen endpoint
and save the result.

This originally called REVE's /v1/image/edit endpoint directly. This repo
doesn't have REVE access, so it now goes through the shared
attacks/common/image_gen.py request_image_edit() helper, which sends the
source image + edit instruction as multimodal input to either:
  - the Responses API's image_generation tool (required for OpenAI GPT-5.x), or
  - legacy chat.completions with image_url content parts (Gemini-style),
picked automatically from --model, or forced via --image-api.
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.common.llm_client import get_client  # noqa: E402
from attacks.common.image_gen import request_image_edit_with_retry  # noqa: E402
from .prompts import (  # noqa: E402
    EDIT_BBBLORK_TEMPLATE,
    EDIT_TEMPLATE,
    EDIT_TEXT_TEMPLATE,
    REMOVE_OBJECT_TEMPLATE,
)


def save_image_bytes(image_bytes: bytes, output_path: Path) -> Image.Image:
    """Decode raw image bytes and save to disk."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image


def generate_attack_image(
    image_path: Path,
    prompt: str,
    client,
    model: str,
    output_path: Path,
    api_style: str = "auto",
    max_retries: int = 3,
    backoff: int = 2,
) -> Path:
    """Request a single image edit, retrying transient failures.

    Delegates to ``request_image_edit_with_retry()`` (see
    attacks/common/image_gen.py), which honors a server-provided
    ``Retry-After`` header on HTTP 429 rate-limit responses and falls back to
    exponential backoff with jitter otherwise.
    """

    def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        print(
            f"Image edit failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. Retrying in {wait:.1f}s...",
            flush=True,
        )

    try:
        images = request_image_edit_with_retry(
            client,
            model=model,
            prompt=prompt,
            images=image_path,
            api_style=api_style,
            max_retries=max_retries - 1 if max_retries > 0 else 0,
            backoff_base=float(backoff),
            on_retry=_on_retry,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Image edit failed after {max_retries} attempts: {exc}") from exc

    save_image_bytes(images[0], output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Edit an image via an OpenAI-compatible image-gen endpoint and save the result."
    )
    parser.add_argument("image_path", type=str, help="Path to the source image to edit.")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="Optional extra instructions appended to the built prompt.",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        choices=["replace_with_object", "replace_with_BBBLORK", "replace_with_nothing", "text_replacement"],
        default="replace_with_object",
        help="Type of edit to perform.",
    )
    parser.add_argument(
        "--original-object",
        type=str,
        default="bomb",
        help="Name of the object to replace/remove in the image (default: bomb).",
    )
    parser.add_argument(
        "--replacement-object",
        type=str,
        default="banana",
        help="Name of the object that should replace the original object (default: banana).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the edited image (default: ./tmp/<input>_attack.jpg).",
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
        help=(
            "API shape to use for image editing. 'chat' = legacy chat.completions with "
            "image_url content parts + response_format={'type':'image'} (OpenRouter-style "
            "image models, e.g. Gemini image). 'responses' = Responses API + image_generation "
            "tool with input_image content blocks (required for OpenAI GPT-5.x image editing). "
            "'auto' (default) guesses from --model (gpt-5* -> responses, else chat)."
        ),
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key override (else LLM_API_KEY / etc.).")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL override (else LLM_API_BASE_URL / etc.).")
    # --aspect-ratio / --version kept only for CLI compatibility with the old
    # REVE-based invocation (run.py still passes them); unused now.
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="16:9",
        help="(Unused with the GPT-5.2 image-gen path; kept for CLI compatibility.)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="(Unused with the GPT-5.2 image-gen path; use --model instead.)",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries on failure (default: 3).")
    parser.add_argument("--retry-backoff", type=int, default=2, help="Exponential backoff base in seconds (default: 2).")

    args = parser.parse_args()

    source_path = Path(args.image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    client = get_client(api_key=args.api_key, base_url=args.base_url)

    # Build prompt based on attack type
    attack_type = args.attack_type
    if not args.original_object:
        raise ValueError("Provide --original-object for all attack types.")

    if attack_type == "replace_with_object":
        if not args.replacement_object:
            raise ValueError("Provide --replacement-object for replace_with_object.")
        base_prompt = EDIT_TEMPLATE.format(
            original_object=args.original_object,
            replacement_object=args.replacement_object,
        )
    elif attack_type == "replace_with_BBBLORK":
        base_prompt = EDIT_BBBLORK_TEMPLATE.format(
            original_object=args.original_object,
        )
    elif attack_type == "replace_with_nothing":
        base_prompt = REMOVE_OBJECT_TEMPLATE.format(
            original_object=args.original_object,
        )
    elif attack_type == "text_replacement":
        if not args.replacement_object:
            raise ValueError("Provide --replacement-object for text_replacement.")
        base_prompt = EDIT_TEXT_TEMPLATE.format(
            original_text=args.original_object,
            replacement_text=args.replacement_object,
        )
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    prompt = f"{base_prompt} {args.prompt}" if args.prompt else base_prompt

    default_out = source_path.with_name(f"{source_path.stem}_attack{source_path.suffix or '.jpg'}")
    output_path = Path(args.output) if args.output else Path("./tmp") / default_out.name

    generate_attack_image(
        image_path=source_path,
        prompt=prompt,
        client=client,
        model=args.model,
        output_path=output_path,
        api_style=args.image_api,
        max_retries=args.max_retries,
        backoff=args.retry_backoff,
    )

    print(f"Prompt used: {prompt}")
    print(f"Edited image saved to: {output_path}")


if __name__ == "__main__":
    main()
