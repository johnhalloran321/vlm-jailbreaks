#!/usr/bin/env python3
"""Shared helpers for calling image-generation-capable models through an
OpenAI-compatible gateway.

Two, mutually incompatible wire formats are in use across model families that
have shown up in this repo:

1. "chat" style -- OpenRouter-routed image models (e.g. Gemini image models)
   return generated images through the *legacy* chat-completions endpoint:
   ``client.chat.completions.create(..., response_format={"type": "image"})``,
   with image bytes surfaced either in a ``message.images`` list or as
   ``data:image/...;base64,...`` URLs embedded in ``message.content``.

2. "responses" style -- OpenAI's GPT-5.x models do NOT support the legacy
   images.generate() endpoint or response_format={"type": "image"} on
   chat.completions. They generate images via the *Responses API* using the
   built-in ``image_generation`` tool:
   ``client.responses.create(model=..., input=..., tools=[{"type": "image_generation"}])``,
   with base64 PNG bytes returned in an ``image_generation_call`` output
   block's ``result`` field.

``request_image()`` below picks the right shape (either from an explicit
``api_style`` or by guessing from the model name) so callers don't need to
duplicate this branching.

``request_image_edit()`` is the image-to-image counterpart, used by the
visual_object_replacement / visual_text_replacement attacks (originally built
against REVE's ``/v1/image/edit`` endpoint, which this repo no longer has
access to). It sends one or more reference images alongside the prompt using
the equivalent multimodal input shape for each style: ``input_image`` content
blocks for the Responses API, or ``image_url`` content parts for
chat.completions.

``request_image_with_retry()`` / ``request_image_edit_with_retry()`` wrap the
two functions above with retry + exponential backoff, and specifically honor
a server-provided ``Retry-After`` header on HTTP 429 rate-limit responses
when present (falling back to backoff otherwise). ``run_concurrent()`` is a
small ThreadPoolExecutor-based helper for fanning a batch of these calls out
across up to ``DEFAULT_MAX_PARALLEL`` (8) concurrent requests -- threads (not
asyncio or multiprocessing) are the right tool here since these calls are
I/O-bound (waiting on HTTP responses), so the GIL is released while a request
is in flight.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from attacks.common.concurrency import (
    DEFAULT_MAX_PARALLEL,
    backoff_wait_seconds as _backoff_wait_seconds,
    is_rate_limit_error as _is_rate_limit_error,
    retry_after_seconds as _retry_after_seconds,
    retry_call as _retry_call,
    run_concurrent,
)


def decode_image_url(url: str) -> bytes:
    if url.startswith("data:image/"):
        _header, b64 = url.split(",", 1)
        return base64.b64decode(b64)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
        return resp.read()


def extract_images_from_chat_payload(payload: Dict[str, Any]) -> List[bytes]:
    """Pull generated image bytes out of a chat.completions response payload."""
    images: List[bytes] = []
    for choice in payload.get("choices", []) or []:
        msg = choice.get("message", {}) or {}

        msg_images = msg.get("images")
        if isinstance(msg_images, list):
            for item in msg_images:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    url = (item.get("image_url") or {}).get("url") or item.get("url")
                    if url:
                        images.append(decode_image_url(url))
                else:
                    data = item.get("data") or item.get("b64_json")
                    if data:
                        images.append(base64.b64decode(data))

        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url") or part.get("url")
                    if url:
                        images.append(decode_image_url(url))
                elif part.get("type") in ("image", "output_image"):
                    data = part.get("image") or part.get("data") or part.get("b64_json")
                    if data:
                        images.append(base64.b64decode(data))
        elif isinstance(content, str):
            match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
            if match:
                images.append(base64.b64decode(match.group(1)))

        for key in ("image", "data", "b64_json"):
            if isinstance(msg.get(key), str):
                images.append(base64.b64decode(msg[key]))
    return images


def extract_images_from_responses_payload(payload: Dict[str, Any]) -> List[bytes]:
    """Pull generated image bytes out of a Responses API payload.

    Looks for ``output_generation_call`` (a.k.a. ``image_generation_call``)
    blocks in ``payload["output"]``, each carrying base64 PNG data in
    ``result``.
    """
    images: List[bytes] = []
    for block in payload.get("output", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image_generation_call":
            data = block.get("result")
            if data:
                images.append(base64.b64decode(data))
    return images


def infer_image_api_style(model: str) -> str:
    """Best-effort guess at which API shape a given image-gen model expects.

    OpenAI's GPT-5.x models only generate images via the Responses API's
    ``image_generation`` tool. Everything else that has shown up in this repo
    so far (Gemini image models routed through OpenRouter, etc.) uses the
    legacy chat.completions + response_format={"type": "image"} shape.
    """
    name = (model or "").lower()
    if "gpt-5" in name or "gpt5" in name:
        return "responses"
    return "chat"


def request_image(
    client,
    *,
    model: str,
    prompt: str,
    api_style: str = "auto",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    response_format: Optional[str] = "image",
    image_config: Optional[Dict[str, Any]] = None,
) -> List[bytes]:
    """Request a single image generation and return decoded image bytes.

    ``api_style``: "auto" (guess from model name), "chat" (legacy
    chat.completions + response_format=image), or "responses" (Responses API
    + image_generation tool).
    """
    style = api_style if api_style in ("chat", "responses") else infer_image_api_style(model)

    if style == "responses":
        tool: Dict[str, Any] = {"type": "image_generation"}
        if image_config:
            # The Responses API image_generation tool accepts a handful of its
            # own knobs (e.g. "size"); our image_config dicts already use
            # OpenRouter-compatible key names ("size": "1024x1024") that line
            # up with this, so just merge them in.
            tool.update(image_config)
        kwargs: Dict[str, Any] = {"model": model, "input": prompt, "tools": [tool]}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
        payload = response.model_dump()
        return extract_images_from_responses_payload(payload)

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if seed is not None:
        kwargs["seed"] = seed
    if response_format:
        kwargs["response_format"] = {"type": response_format}
    if image_config:
        kwargs["extra_body"] = {"image_config": image_config}
    response = client.chat.completions.create(**kwargs)
    payload = response.model_dump()
    return extract_images_from_chat_payload(payload)


def _to_data_url(image: Union[str, Path, bytes]) -> str:
    """Turn a file path or raw bytes into a ``data:image/...;base64,...`` URL.

    Plain strings are treated as filesystem paths (not raw base64) since every
    caller in this repo works with reference images that live on disk.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        data = path.read_bytes()
    else:
        mime = "image/png"
        data = image
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def request_image_edit(
    client,
    *,
    model: str,
    prompt: str,
    images: Union[str, Path, bytes, Sequence[Union[str, Path, bytes]]],
    api_style: str = "auto",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    response_format: Optional[str] = "image",
    image_config: Optional[Dict[str, Any]] = None,
) -> List[bytes]:
    """Edit one or more reference images according to a text instruction.

    This is the image-to-image counterpart of ``request_image()`` -- it sends
    the reference image(s) alongside the prompt so the model can perform an
    *edit* (object replacement, text replacement, removal, etc.) grounded in
    the input image(s), mirroring what REVE's ``/v1/image/edit`` endpoint did
    for this repo previously.

    ``images`` may be a single path/bytes or a sequence of them.

    ``api_style``: "auto" (guess from model name), "chat" (legacy
    chat.completions with multimodal content: text + image_url parts,
    response_format=image -- e.g. Gemini-style OpenRouter models), or
    "responses" (Responses API + image_generation tool, with the reference
    image(s) passed as ``input_image`` content blocks -- required for OpenAI
    GPT-5.x image editing).
    """
    style = api_style if api_style in ("chat", "responses") else infer_image_api_style(model)

    if isinstance(images, (str, Path, bytes)):
        images = [images]
    data_urls = [_to_data_url(img) for img in images]

    if style == "responses":
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for url in data_urls:
            content.append({"type": "input_image", "image_url": url})
        tool: Dict[str, Any] = {"type": "image_generation"}
        if image_config:
            tool.update(image_config)
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "tools": [tool],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.responses.create(**kwargs)
        payload = response.model_dump()
        return extract_images_from_responses_payload(payload)

    chat_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in data_urls:
        chat_content.append({"type": "image_url", "image_url": {"url": url}})
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": chat_content}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if seed is not None:
        kwargs["seed"] = seed
    if response_format:
        kwargs["response_format"] = {"type": response_format}
    if image_config:
        kwargs["extra_body"] = {"image_config": image_config}
    response = client.chat.completions.create(**kwargs)
    payload = response.model_dump()
    return extract_images_from_chat_payload(payload)


# ============================================================================
# Retry / backoff / concurrency helpers
#
# The generic pieces (rate-limit detection, backoff, the retry loop, and the
# ThreadPoolExecutor fan-out) now live in attacks/common/concurrency.py, since
# the same shape is useful for plain text (chat.completions) call sites too,
# not just image generation. They're imported above and re-exported under
# their original names here so every existing caller of
# ``from attacks.common.image_gen import DEFAULT_MAX_PARALLEL, run_concurrent,
# request_image_with_retry, ...`` keeps working unchanged.
# ============================================================================


def _require_nonempty(fn: Callable[[], List[bytes]]) -> Callable[[], List[bytes]]:
    """Wrap a zero-arg image-request callable so an empty (but exception-free)
    result is treated as a retryable failure.

    Every call site in this repo historically treated "the API call succeeded
    but returned zero images" the same as a raised exception -- worth
    retrying, not worth silently returning ``[]``. Centralizing that here
    means ``request_image_with_retry()`` / ``request_image_edit_with_retry()``
    preserve that behavior without every caller having to re-implement it.
    """

    def _call() -> List[bytes]:
        images = fn()
        if not images:
            raise RuntimeError("No image returned in response.")
        return images

    return _call


def request_image_with_retry(
    client,
    *,
    model: str,
    prompt: str,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> List[bytes]:
    """``request_image()`` wrapped with rate-limit-aware retry + backoff.

    On HTTP 429 responses, honors a server-provided ``Retry-After`` header
    when present; otherwise (and for any other transient error, including an
    empty/no-image response) falls back to exponential backoff with jitter.
    ``max_retries`` is the number of retries after the first attempt (default
    3, i.e. up to 4 attempts total).
    """
    return _retry_call(
        _require_nonempty(lambda: request_image(client, model=model, prompt=prompt, **kwargs)),
        max_retries=max_retries,
        backoff_base=backoff_base,
        on_retry=on_retry,
    )


def request_image_edit_with_retry(
    client,
    *,
    model: str,
    prompt: str,
    images: Union[str, Path, bytes, Sequence[Union[str, Path, bytes]]],
    max_retries: int = 3,
    backoff_base: float = 2.0,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    **kwargs: Any,
) -> List[bytes]:
    """``request_image_edit()`` wrapped with rate-limit-aware retry + backoff.

    Same retry/backoff semantics as ``request_image_with_retry()`` -- see its
    docstring for details (including treating an empty/no-image response as a
    retryable failure).
    """
    return _retry_call(
        _require_nonempty(
            lambda: request_image_edit(client, model=model, prompt=prompt, images=images, **kwargs)
        ),
        max_retries=max_retries,
        backoff_base=backoff_base,
        on_retry=on_retry,
    )
