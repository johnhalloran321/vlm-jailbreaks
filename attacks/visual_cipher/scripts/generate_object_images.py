#!/usr/bin/env python3
"""Generate a set of square images for everyday objects using OpenRouter."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROMPTS: List[str] = [
    "A smiling red apple with a tiny heart sticker",
    "A sunny bunch of ripe bananas tied with a ribbon",
    "A bright orange with a happy face drawn in marker",
    "A bowl of fresh strawberries with a small smiley tag",
    "A cluster of grapes with a cheerful pastel label",
    "A juicy pear with a cute polka dot bow",
    "A peach with a soft pink glow and a smiley note",
    "A pineapple wearing a tiny party hat",
    "A slice of watermelon with a friendly face",
    "A lemon with a big cheerful grin sticker",
    "A lime with a tiny green smile button",
    "A cherry pair with a smiling ribbon",
    "A handful of blueberries in a happy paper cup",
    "A raspberry with a tiny heart tag",
    "A mango with a cheerful sun sticker",
    "A kiwi cut in half like a smiling face",
    "A tomato with a happy kitchen doodle",
    "A carrot with a friendly smile sticker",
    "A broccoli floret shaped like a little tree with a smile",
    "A cucumber with a cheerful striped band",
    "A bell pepper with a bright smiley magnet",
    "A friendly eggplant with a tiny bow tie",
    "A potato with a smiling face drawn on it",
    "An onion with a happy label and a bow",
    "A garlic bulb with a cute grin sticker",
    "A mushroom with a cheerful polka dot cap",
    "A corn cob with a happy picnic tag",
    "A warm bread loaf with a smiling bakery sticker",
    "A crisp baguette wrapped in a cheerful paper sleeve",
    "A golden croissant with a tiny heart flag",
    "A colorful donut with sprinkles and a smiley face",
    "A cupcake with rainbow frosting and a smile topper",
    "A cookie with a friendly smile and chocolate chips",
    "A chocolate bar with a cheerful wrapper",
    "A coffee mug with a sunny face and steam hearts",
    "A teacup with a floral pattern and a smiley tag",
    "A water bottle with a bright cheerful label",
    "A soda can with a happy retro design",
    "A milk carton with a smiling cow illustration",
    "A cereal bowl with a happy spoon resting beside it",
    "A plate with a cute smiley doodle on the rim",
    "A fork with a tiny bow and a cheerful sticker",
    "A spoon with a smiley face on the handle",
    "A knife with a colorful happy handle",
    "A frying pan with a sunny yellow handle",
    "A saucepan with a cheerful lid knob",
    "A kettle with a smiley magnet",
    "A toaster with a happy sticker and toast hearts",
    "A blender with a bright smiley label",
    "A cutting board with a cheerful gingham pattern",
    "A wooden spoon with a cute painted face",
    "A rolling pin with a pastel ribbon",
    "A kitchen timer shaped like a smiling tomato",
    "A salt shaker with a friendly face",
    "A pepper shaker with a cheerful label",
    "A candle with a warm glow and a heart tag",
    "A light bulb glowing softly with a smile sticker",
    "A flashlight with a bright happy sticker",
    "A battery with a cheerful green label",
    "An alarm clock with a smiling face",
    "A wristwatch with a sunny yellow band",
    "Eyeglasses with a cute heart case",
    "Sunglasses with a bright rainbow frame",
    "A wallet with a happy sticker",
    "A key with a tiny smiling keychain",
    "A keychain shaped like a happy star",
    "A backpack with a cheerful patch",
    "A handbag with a bright floral charm",
    "A cheerful bright yellow umbrella with a cute smiling face",
    "A cozy knit hat with a smiley tag",
    "A soft scarf with pastel stripes and a happy pin",
    "A pair of gloves with a tiny heart patch",
    "A cozy sock with a smiling face pattern",
    "A comfy shoe with a cheerful lace charm",
    "A sneaker with colorful happy accents",
    "A book with a friendly cover illustration",
    "A notebook with a smiling sun on the cover",
    "A pencil with a happy eraser face",
    "A pen with a cheerful clip",
    "An eraser shaped like a smiling cloud",
    "A ruler with a rainbow stripe and a smiley sticker",
    "A stapler with a bright cheerful sticker",
    "A paperclip shaped like a heart",
    "A tape dispenser with a sunny pattern",
    "A pair of scissors with colorful handles and a smiley tag",
    "A calculator with a bright cheerful sticker",
    "A smartphone with a happy wallpaper",
    "A remote control with a cute smiley decal",
    "Headphones with a colorful joyful sticker",
    "A camera with a cheerful strap and a smile tag",
    "A plant pot with a smiling face",
    "A flower vase with fresh daisies and a smiley note",
    "A picture frame with a heart border",
    "A pillow with a happy face embroidery",
    "A blanket with warm pastel stripes",
    "A chair with a cheerful cushion",
    "A small table with a sunny yellow top",
    "A lamp with a warm smiling shade",
    "A bicycle with a cheerful bell and a flower basket",
    "A soccer ball with a friendly smiley logo",
]


def slugify(text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return base or "object"


def build_prompt(obj: str) -> str:
    text = obj.strip()
    if not text:
        return "a photo of a cheerful object on a plain background, centered, no text"
    lower = text.lower()
    if lower.startswith(("a ", "an ", "the ")):
        base = text
    else:
        base = f"a photo of a {text}"
    return f"{base} on a plain background, centered, no text"


def load_prompts(path: Path | None) -> List[str]:
    if not path:
        return PROMPTS.copy()
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    return prompts


def decode_image_url(url: str) -> bytes:
    if url.startswith("data:image/"):
        header, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def ensure_deps() -> None:
    try:
        import openai  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai", "pillow"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object image tiles with OpenRouter.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/object_tiles"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--gen-size", type=int, default=1024, help="Requested generation size (e.g., 1024).")
    parser.add_argument("--output-size", type=int, default=128, help="Final saved size (e.g., 128).")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash-image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests (sequential mode only, i.e. --max-parallel 1).")
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per prompt on failure.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Number of concurrent worker threads (default: DEFAULT_MAX_PARALLEL, currently 8). Use 1 for sequential.",
    )
    parser.add_argument("--response-format", type=str, default="image", help="Response format type (chat API style only).")
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
    parser.add_argument("--image-config", type=str, default="", help="Optional JSON dict for image_config.")
    parser.add_argument("--prompts-file", type=Path, default=None, help="Optional text file with one object per line.")
    parser.add_argument("--api-key", type=str, default="", help="API key for the gateway (or set LLM_API_KEY / legacy OPENROUTER_API_KEY).")
    parser.add_argument("--app-name", type=str, default="", help="Optional OpenRouter app name.")
    parser.add_argument("--app-url", type=str, default="", help="Optional OpenRouter app URL.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images even if files already exist.")
    args = parser.parse_args()

    ensure_deps()

    from PIL import Image
    from attacks.common.llm_client import get_client
    from attacks.common.image_gen import DEFAULT_MAX_PARALLEL, request_image_with_retry

    max_parallel = max(1, args.max_parallel if args.max_parallel is not None else DEFAULT_MAX_PARALLEL)

    prompts = load_prompts(args.prompts_file)
    if args.count > len(prompts):
        raise SystemExit(f"Requested {args.count} images, but only {len(prompts)} prompts available.")

    headers = {}
    if args.app_name:
        headers["X-Title"] = args.app_name
    if args.app_url:
        headers["HTTP-Referer"] = args.app_url
    # NOTE: this needs an actual image-generation-capable model. OpenRouter-style
    # image models (e.g. Gemini image) are called via the legacy chat.completions
    # endpoint (--image-api chat); OpenAI GPT-5.x models generate images only via
    # the Responses API's image_generation tool (--image-api responses, or leave
    # --image-api auto and it's picked from the model name). A plain text/reasoning
    # chat deployment with neither of these capabilities will not work.
    #
    # get_client() also honors LLM_API_INSECURE_SSL=1 to skip TLS verification
    # for internal gateways with self-signed/internal-CA certs (see
    # attacks/common/llm_client.py) -- without it, a cert-trust failure shows up
    # here as an opaque "Connection error" that looks like a bad URL/network issue.
    try:
        client = get_client(
            api_key=args.api_key or None,
            timeout=args.timeout,
            default_headers=headers or None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))

    if args.image_config:
        image_config = json.loads(args.image_config)
    else:
        image_config = {"size": f"{args.gen_size}x{args.gen_size}"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "gen_size": args.gen_size,
        "output_size": args.output_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "image_config": image_config,
        "seed_base": args.seed,
        "images": [],
    }

    work_items = list(enumerate(prompts[: args.count]))

    def process_one(item: tuple) -> Optional[Dict[str, Any]]:
        idx, obj = item
        prompt = build_prompt(obj)
        filename = f"{idx:03d}_{slugify(obj)}.png"
        out_path = args.output_dir / filename
        entry = {"index": idx, "object": obj, "prompt": prompt, "file": filename, "seed": args.seed + idx}
        if out_path.exists() and not args.overwrite:
            return entry

        print(f"[{idx+1}/{args.count}] Requesting image for: {obj}")

        def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
            print(f"  attempt {attempt + 1}/{args.retries + 1} failed: {exc} (retrying in {wait:.1f}s)")

        try:
            images = request_image_with_retry(
                client,
                model=args.model,
                prompt=prompt,
                api_style=args.image_api,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed + idx,
                response_format=args.response_format,
                image_config=image_config,
                max_retries=args.retries,
                on_retry=_on_retry,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to get image for {obj}: {exc}")
            return None

        image = Image.open(io.BytesIO(images[0])).convert("RGB")
        if args.output_size and image.size != (args.output_size, args.output_size):
            image = image.resize((args.output_size, args.output_size), Image.LANCZOS)
        image.save(out_path)
        print(f"Wrote {out_path}")
        if max_parallel <= 1 and args.sleep > 0:
            time.sleep(args.sleep)
        return entry

    results: List[Optional[Dict[str, Any]]] = [None] * len(work_items)
    if max_parallel <= 1:
        for i, item in enumerate(work_items):
            results[i] = process_one(item)
    else:
        with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="ObjectImageWorker") as pool:
            future_to_index = {pool.submit(process_one, item): i for i, item in enumerate(work_items)}
            for future in future_to_index:
                i = future_to_index[future]
                results[i] = future.result()

    manifest["images"] = [entry for entry in results if entry is not None]

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
