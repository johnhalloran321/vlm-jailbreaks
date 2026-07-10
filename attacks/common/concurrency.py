#!/usr/bin/env python3
"""Generic retry/backoff/concurrency helpers shared across every script in this
repo that fans blocking HTTP calls out over a thread pool.

These were originally written inline in ``attacks/common/image_gen.py`` for the
image-generation call chain, but the same shape shows up for plain text
(chat.completions) calls too (e.g. riddle/prompt generation) -- rate limits,
5xx transients, and "generate N of these sequentially" loops aren't specific
to image generation. Pulling the generic pieces out here lets any script reuse
them without importing image-generation-specific code.

Threads (not asyncio or multiprocessing) are used deliberately throughout:
every caller uses this to make a blocking HTTP call via the ``openai`` SDK,
which is I/O-bound (the GIL is released while waiting on the network), so a
thread pool gets real concurrency without the complexity of an async rewrite.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, Sequence, TypeVar

DEFAULT_MAX_PARALLEL = 8

T = TypeVar("T")

_RATE_LIMIT_STATUS = 429


def _status_code_of(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an SDK exception.

    The ``openai`` Python SDK raises ``openai.APIStatusError`` subclasses
    (``RateLimitError`` for 429, etc.) that expose ``.status_code``. Some
    transports/mocks instead nest a ``.response.status_code`` (``httpx``
    style). Both shapes are checked so this works regardless of exactly which
    SDK/transport version raised the error.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) if response is not None else None
    if isinstance(status, int):
        return status
    return None


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when ``exc`` looks like an HTTP 429 rate-limit response.

    Falls back to sniffing the exception's class name / message for "rate
    limit" or "429" when no structured status code is available, since some
    gateways (e.g. LiteLLM proxies) wrap the underlying error differently.
    """
    if _status_code_of(exc) == _RATE_LIMIT_STATUS:
        return True
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "rate_limit" in message


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Pull a server-provided ``Retry-After`` header value off ``exc``, if any.

    ``Retry-After`` may be an integer number of seconds. (HTTP also allows an
    HTTP-date form; that's not handled here since none of the gateways this
    repo talks to have been observed sending it.)
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def backoff_wait_seconds(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter: ``base * 2**attempt`` seconds, capped.

    ``attempt`` is 0-indexed (first retry -> ``attempt=0``). A small random
    jitter is added so a batch of concurrently-failing requests doesn't all
    retry in lockstep and immediately re-trip the same rate limit.
    """
    wait = min(cap, base * (2 ** attempt))
    return wait + random.uniform(0, wait * 0.25)


def retry_call(
    fn: Callable[[], T],
    *,
    max_retries: int,
    backoff_base: float = 2.0,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    is_retryable: Optional[Callable[[BaseException], bool]] = None,
) -> T:
    """Call ``fn()``, retrying on any exception with rate-limit-aware waits.

    Retries on any exception (not just rate limits), but chooses its wait
    time based on whether the error looks like a 429: honors a
    server-provided ``Retry-After`` header when present, otherwise falls back
    to exponential backoff with jitter. ``max_retries`` is the number of
    *retries* after the initial attempt, so the call is attempted up to
    ``max_retries + 1`` times total.

    ``is_retryable`` -- optional predicate; when provided and it returns
    ``False`` for a raised exception, that exception is treated as permanent
    and re-raised immediately with no further attempts or waiting, regardless
    of remaining ``max_retries`` budget. Useful for errors that are known to
    be deterministic for a given input (e.g. an explicit content-safety
    refusal from a model) where retrying the identical request is virtually
    guaranteed to fail identically and only wastes time/quota. Defaults to
    ``None``, which retries every exception (previous behavior).
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: retry any transient failure
            last_exc = exc
            if is_retryable is not None and not is_retryable(exc):
                break
            if attempt >= max_retries:
                break
            if is_rate_limit_error(exc):
                wait = retry_after_seconds(exc)
                if wait is None:
                    wait = backoff_wait_seconds(attempt, base=backoff_base)
            else:
                wait = backoff_wait_seconds(attempt, base=backoff_base)
            if on_retry is not None:
                on_retry(attempt, exc, wait)
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def run_concurrent(
    items: Sequence[T],
    worker: Callable[[T], Any],
    *,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    thread_name_prefix: str = "Worker",
) -> list:
    """Fan a batch of ``worker(item)`` calls out across up to ``max_parallel``
    threads, returning results in the same order as ``items``.

    ``worker`` should catch and handle/log its own exceptions (e.g. by
    wrapping calls in ``retry_call()`` and catching whatever it ultimately
    re-raises) if a single item's failure shouldn't abort the rest of the
    batch -- any exception that escapes ``worker`` here is re-raised from the
    first failing ``future.result()`` call, same as it would be in a
    sequential loop.

    ``max_parallel`` is clamped to at least 1 and at most ``len(items)`` (no
    point spinning up more threads than there is work).
    """
    if not items:
        return []
    workers = max(1, min(max_parallel, len(items)))
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix) as pool:
        future_to_index = {pool.submit(worker, item): idx for idx, item in enumerate(items)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
    return results
