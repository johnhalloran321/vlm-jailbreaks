#!/usr/bin/env python3
"""
Generate a base image from a text prompt via an OpenAI-compatible image-gen
endpoint (e.g. GPT-5.2 through the Responses API's image_generation tool).

This originally called REVE's /v1/image/create endpoint directly. This repo
doesn't have REVE access, so it now goes through the same shared
attacks/common/image_gen.py helper used by the other attacks (visual_cipher,
analogy), which speaks either:
  - the Responses API's image_generation tool (required for OpenAI GPT-5.x), or
  - legacy chat.completions response_format={"type":"image"} (Gemini-style),
picked automatically from --model, or forced via --image-api.
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.common.llm_client import get_client  # noqa: E402
from attacks.common.image_gen import request_image_with_retry  # noqa: E402
from .prompts import CREATE_TEMPLATE, CREATE_TEXT_TEMPLATE  # noqa: E402


def build_prompt(object_name: str, extra: Optional[str], attack_type: Optional[str] = None) -> str:
    """Construct the generation prompt using the create template constant."""
    if attack_type == "text_replacement":
        base_prompt = CREATE_TEXT_TEMPLATE.format(target_text=object_name)
    else:
        base_prompt = CREATE_TEMPLATE.format(object=object_name)
    if extra:
        return f"{base_prompt} {extra}"
    return base_prompt


def save_image_bytes(image_bytes: bytes, output_path: Path) -> Tuple[Path, Image.Image]:
    """Decode raw image bytes and save to disk."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path, image


def generate_base_image(
    prompt: str,
    client,
    model: str,
    output_path: Path,
    api_style: str = "auto",
    max_retries: int = 3,
    backoff: int = 2,
) -> Path:
    """Request a single image generation, retrying transient failures.

    Delegates to ``request_image_with_retry()`` (see
    attacks/common/image_gen.py), which honors a server-provided
    ``Retry-After`` header on HTTP 429 rate-limit responses and falls back to
    exponential backoff with jitter otherwise -- an upgrade over this
    function's previous local fixed-exponential-backoff-only retry loop.
    """

    def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        print(
            f"Image generation failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
            f"Retrying in {wait:.1f}s...",
            flush=True,
        )

    try:
        images = request_image_with_retry(
            client,
            model=model,
            prompt=prompt,
            api_style=api_style,
            max_retries=max_retries - 1 if max_retries > 0 else 0,
            backoff_base=float(backoff),
            on_retry=_on_retry,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Image generation failed after {max_retries} attempts: {exc}") from exc

    save_image_bytes(images[0], output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a base image via an OpenAI-compatible image-gen endpoint using a template-based prompt."
    )
    parser.add_argument(
        "object",
        type=str,
        nargs="?",
        default="bomb",
        help="Object to imply via context (default: bomb).",
    )
    parser.add_argument(
        "extra",
        type=str,
        nargs="?",
        help="Optional extra instructions to append to the template prompt.",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default=None,
        help="Attack type for determining prompt template (e.g., 'text_replacement').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for image (default: ./tmp/generated_<object>.png).",
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
            "API shape to use for image generation. 'chat' = legacy chat.completions + "
            "response_format={'type':'image'} (OpenRouter-style image models, e.g. Gemini image). "
            "'responses' = Responses API + image_generation tool (required for OpenAI GPT-5.x image "
            "generation). 'auto' (default) guesses from --model (gpt-5* -> responses, else chat)."
        ),
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key override (else LLM_API_KEY / etc.).")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL override (else LLM_API_BASE_URL / etc.).")
    # --aspect-ratio / --version / --timeout are kept only for CLI compatibility with
    # the old REVE-based invocation (run.py still passes them); they're unused now
    # that image generation goes through the Responses/chat.completions APIs, which
    # don't expose an equivalent aspect-ratio or model-version knob.
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
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=2,
        help="Exponential backoff base in seconds (default: 2).",
    )

    args = parser.parse_args()

    client = get_client(api_key=args.api_key, base_url=args.base_url)
    prompt = build_prompt(args.object, args.extra, args.attack_type)

    print(f"Requesting image generation with model={args.model}, image_api={args.image_api}...", flush=True)

    output_path = Path(args.output) if args.output else Path(f"./tmp/generated_{args.object}.png")

    generate_base_image(
        prompt=prompt,
        client=client,
        model=args.model,
        output_path=output_path,
        api_style=args.image_api,
        max_retries=args.max_retries,
        backoff=args.retry_backoff,
    )

    print(f"Prompt used: {prompt}")
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
