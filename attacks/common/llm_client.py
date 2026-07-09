#!/usr/bin/env python3
"""Shared factory for talking to an OpenAI-compatible LLM gateway.

Historically every "judge"/"validator" script in this repo hardcoded the `openai`
Python SDK against OpenRouter's endpoint (`https://openrouter.ai/api/v1`), since
OpenRouter exposes an OpenAI-compatible `/chat/completions` API in front of many
vendors' models (Qwen, Grok, Gemini, Claude, GPT, ...).

This repo has since been adjusted to work against any OpenAI-compatible gateway
(e.g. a self-hosted LiteLLM proxy) rather than being hardcoded to OpenRouter, since
that's what's actually reachable in this environment: GPT-5.x (Azure) and several
Claude models (Bedrock), fronted by a LiteLLM router. LiteLLM, like OpenRouter,
speaks the OpenAI chat-completions wire format, so the same request/response
shapes used throughout this repo work unchanged -- only the base URL, API key, and
model name strings need to change.

Resolution order (first match wins):
    explicit function argument
    LLM_API_BASE_URL / LLM_API_KEY   (preferred; gateway-agnostic)
    OPENROUTER_API_BASE / OPENROUTER_API_KEY   (legacy, still supported)
    OLLAMA_API_BASE / OLLAMA_API_KEY   (local/dev fallback used by a few scripts)
    REVE_API_BASE / REVE_API_KEY       (image-generation fallback used by a few scripts)
    default -> https://openrouter.ai/api/v1

Some self-hosted gateways (e.g. an internal LiteLLM proxy fronted by a
load balancer with a self-signed / internal-CA certificate) will fail TLS
verification from environments that don't trust that CA. The `openai`
Python SDK surfaces that failure as an opaque `Connection error` rather than
a certificate error, which is easy to misdiagnose as a network-reachability
or URL-path problem. Set LLM_API_INSECURE_SSL=1 (or true/yes/on) to disable
certificate verification for the client built by get_client() -- only do
this for gateways you trust on a network you trust (e.g. an internal
corporate endpoint), since it removes protection against MITM attacks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_base_url(base_url: Optional[str] = None) -> str:
    """Resolve the OpenAI-compatible gateway base URL to use."""
    return (
        base_url
        or os.environ.get("LLM_API_BASE_URL")
        or os.environ.get("OPENROUTER_API_BASE")
        or os.environ.get("OLLAMA_API_BASE")
        or os.environ.get("REVE_API_BASE")
        or DEFAULT_BASE_URL
    )


def resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the API key to use against the resolved gateway base URL."""
    return (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OLLAMA_API_KEY")
        or os.environ.get("REVE_API_KEY")
    )


def is_openrouter_base_url(base_url: Optional[str] = None) -> bool:
    """True when the resolved base URL still points at openrouter.ai.

    Some request payloads in this repo set OpenRouter-only fields (e.g.
    `extra_body={"include_reasoning": ...}`). Those fields should only be sent
    when actually talking to OpenRouter -- other OpenAI-compatible gateways
    (e.g. LiteLLM fronting Azure OpenAI / Bedrock) may reject unrecognized
    fields, so callers should gate those extras behind this check.
    """
    return "openrouter.ai" in resolve_base_url(base_url)


def insecure_ssl_enabled() -> bool:
    """Whether LLM_API_INSECURE_SSL asks us to skip TLS certificate verification."""
    return os.environ.get("LLM_API_INSECURE_SSL", "").strip().lower() in ("1", "true", "yes", "on")


def _build_http_client():
    """Return an httpx.Client with verification disabled, or None if not requested."""
    if not insecure_ssl_enabled():
        return None
    import httpx

    return httpx.Client(verify=False)


def get_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    default_headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    """Build an `openai.OpenAI` client pointed at the configured gateway.

    Works against OpenRouter, a LiteLLM proxy, or any other OpenAI-compatible
    endpoint -- see module docstring for the env var resolution order. If
    LLM_API_INSECURE_SSL is set, TLS certificate verification is disabled
    (needed for some internal gateways with self-signed/internal-CA certs).
    """
    resolved_base = resolve_base_url(base_url)
    resolved_key = resolve_api_key(api_key)
    if not resolved_key:
        if str(resolved_base).startswith("http://localhost:11434") or str(resolved_base).startswith(
            "http://127.0.0.1:11434"
        ):
            resolved_key = "ollama"
        else:
            raise ValueError(
                "No API key found. Set LLM_API_KEY (preferred) or OPENROUTER_API_KEY, "
                "or pass api_key explicitly."
            )

    kwargs: Dict[str, Any] = {"base_url": resolved_base, "api_key": resolved_key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if default_headers:
        kwargs["default_headers"] = default_headers

    http_client = _build_http_client()
    if http_client is not None:
        kwargs["http_client"] = http_client

    return OpenAI(**kwargs)
