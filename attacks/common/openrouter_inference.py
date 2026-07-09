#!/usr/bin/env python3
"""
Inference module for analyzing images with Vision Language Models over an
OpenAI-compatible chat-completions API.

Defaults to OpenRouter (https://openrouter.ai/api/v1) for backward compatibility,
but the base URL and API key are configurable via LLM_API_BASE_URL / LLM_API_KEY
so this same code works against any OpenAI-compatible gateway -- e.g. a LiteLLM
proxy fronting Azure OpenAI / Bedrock -- by just changing env vars, not code.
See attacks/common/llm_client.py for the resolution order.
"""

import os
import base64
import json
from pathlib import Path
import time
import requests

from attacks.common.llm_client import resolve_api_key, resolve_base_url

DEFAULT_MODEL = "gpt-5-2-azure-comm-il2"


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_url(image_path: str) -> str:
    """
    Convert image to data URL format for OpenRouter.

    Args:
        image_path: Path to the image file

    Returns:
        Data URL string
    """
    image_base64 = encode_image(image_path)
    
    # Detect image type from extension
    ext = Path(image_path).suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(ext, 'image/jpeg')

    return f"data:{mime_type};base64,{image_base64}"


def _chat_completions_url(base_url: str) -> str:
    """Join a gateway base URL (e.g. 'https://host/v1') with the chat-completions path."""
    return base_url.rstrip("/") + "/chat/completions"


def analyze_image_openrouter(
    image_path: str,
    prompt: str = "What's in this image?",
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    base_url: str = None,
    verbose: bool = True
) -> str:
    """
    Analyze an image with a VLM over an OpenAI-compatible chat-completions API.

    Args:
        image_path: Path to the image file
        prompt: Question or instruction about the image
        model: Model name (default: DEFAULT_MODEL)
        api_key: API key (optional; falls back to LLM_API_KEY / OPENROUTER_API_KEY env vars)
        base_url: Gateway base URL (optional; falls back to LLM_API_BASE_URL / OPENROUTER_API_BASE,
            defaulting to OpenRouter's endpoint)
        verbose: Whether to print progress information

    Returns:
        Response from the API
    """
    # Validate image path
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get API key and gateway base URL
    api_key = resolve_api_key(api_key)
    resolved_base_url = resolve_base_url(base_url)

    if not api_key:
        raise ValueError(
            "No API key provided. "
            "Set LLM_API_KEY (or legacy OPENROUTER_API_KEY) "
            "or pass api_key as an argument."
        )

    if verbose:
        print(f"Analyzing image: {image_path}")
        print(f"Gateway: {resolved_base_url}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print("-" * 70)

    # Convert image to data URL
    image_url = get_image_url(image_path)

    # Prepare the request
    url = _chat_completions_url(resolved_base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    }

    return _post_with_retries(url, headers, payload, verbose)


def analyze_multiple_images_openrouter(
    image_paths: list,
    prompt: str = "What's in these images?",
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    base_url: str = None,
    verbose: bool = True
) -> str:
    """
    Analyze multiple images with a single prompt over an OpenAI-compatible chat-completions API.

    Args:
        image_paths: List of paths to image files
        prompt: Question or instruction about the images
        model: Model name (default: DEFAULT_MODEL)
        api_key: API key (optional; falls back to LLM_API_KEY / OPENROUTER_API_KEY env vars)
        base_url: Gateway base URL (optional; falls back to LLM_API_BASE_URL / OPENROUTER_API_BASE,
            defaulting to OpenRouter's endpoint)
        verbose: Whether to print progress information

    Returns:
        Response from the API
    """
    # Validate image paths
    for image_path in image_paths:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get API key and gateway base URL
    api_key = resolve_api_key(api_key)
    resolved_base_url = resolve_base_url(base_url)

    if not api_key:
        raise ValueError(
            "No API key provided. "
            "Set LLM_API_KEY (or legacy OPENROUTER_API_KEY) "
            "or pass api_key as an argument."
        )

    if verbose:
        print(f"Analyzing {len(image_paths)} images:")
        for img_path in image_paths:
            print(f"  - {img_path}")
        print(f"Gateway: {resolved_base_url}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        print("-" * 70)

    # Prepare the request
    url = _chat_completions_url(resolved_base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Build content array with text prompt first, then all images
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]
    
    # Add all images
    for image_path in image_paths:
        image_url = get_image_url(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    return _post_with_retries(url, headers, payload, verbose)


def _post_with_retries(url: str, headers: dict, payload: dict, verbose: bool, max_retries: int = 5) -> str:
    """Send request with retry on 503 responses."""
    backoff = 1
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            if verbose:
                print("\nResponse:")

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            try:
                result = response.json()
            except ValueError:
                # Surface the raw body when JSON parsing fails so callers can debug quota/endpoint issues.
                raise RuntimeError(
                    f"Error making request to OpenRouter: unable to parse JSON response\nResponse text: {response.text}"
                )
            content = result['choices'][0]['message']['content']

            if verbose:
                print(content)
                print()  # Add newline after response

            return content

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            last_error = e
            if status == 503 and attempt < max_retries:
                if verbose:
                    print(f"Received 503 from OpenRouter. Retrying in {backoff}s (attempt {attempt}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2
                continue
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            break

    error_msg = f"Error making request to OpenRouter: {last_error}"
    if hasattr(last_error, 'response') and last_error.response is not None:
        try:
            error_detail = last_error.response.json()
            error_msg += f"\nError details: {json.dumps(error_detail, indent=2)}"
        except:
            error_msg += f"\nResponse text: {last_error.response.text}"
    raise RuntimeError(error_msg)
