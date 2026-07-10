#!/usr/bin/env python3
"""
Pipeline to generate base images and run configured attacks via an
OpenAI-compatible image-gen endpoint (e.g. GPT-5.2 through the Responses
API's image_generation tool).

Usage:
  python -m attacks.visual_object_replacement.data_generation.run \
      --config attacks/visual_object_replacement/data_generation/config.json

Reads a JSON config (see attacks/visual_object_replacement/data_generation/config.json) with:
  - output_root
  - base_generation settings (count_per_object)
  - objects: list of target object strings
  - attacks: list of attack configs:
        {
          "type": "replace_with_object" | "replace_with_BBBLORK" | "replace_with_nothing",
          "replacements": ["banana", ...]   # only for replace_with_object
        }
For each object, generates N base images (via generate_base_image.py) then applies each attack
on each generated image (via generate_attack_image.py).

This originally shelled out to REVE-backed scripts and (independently of that)
built its subprocess command paths incorrectly (pointing at a
data_generation/visual_replacement/ directory that doesn't exist in this
repo, and passing a plain script path to modules that use relative imports,
which would fail with "attempted relative import with no known parent
package"). Both issues are fixed here: subprocess calls now use
`python -m attacks.visual_object_replacement.data_generation.<module>` run
from REPO_ROOT, and REVE_API_KEY is no longer required -- image generation
goes through the same LLM_API_BASE_URL / LLM_API_KEY / IMAGE_GEN_MODEL
gateway config used by the rest of this repo (see attacks/common/llm_client.py).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.common.image_gen import REFUSAL_EXIT_CODE  # noqa: E402

# NOTE: this used to be 4, which silently contradicted the --max-parallel
# help text ("hard capped at 8") below, since this same constant is used both
# as the default *and* as the hard cap (`min(args.max_parallel,
# MAX_PARALLEL_DEFAULT)`) -- a user passing e.g. --max-parallel 8 was
# silently clamped down to 4. Set to 8 to match the documented behavior and
# the default used by the rest of the image-gen pipelines (see
# attacks/common/image_gen.py's DEFAULT_MAX_PARALLEL).
MAX_PARALLEL_DEFAULT = 8
MAX_TASK_ATTEMPTS = 3


class FailureTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._failures: List[str] = []

    def record_failure(self, description: str):
        with self._lock:
            self._failures.append(description)

    def get_failures(self) -> List[str]:
        with self._lock:
            return list(self._failures)


def run_generation_task(
    description: str,
    cmd: List[str],
    env: Optional[dict],
    failure_tracker: FailureTracker,
    max_attempts: int,
):
    for attempt in range(1, max_attempts + 1):
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Launching {description} (attempt {attempt}/{max_attempts})")
        result = None
        try:
            result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
        except Exception as exc:
            print(f"[{thread_name}] {description} raised {exc!r}")
        else:
            if result and result.returncode == 0:
                print(f"[{thread_name}] Completed {description}")
                return True
            if result and result.returncode == REFUSAL_EXIT_CODE:
                # The model explicitly declined this request (content-safety
                # refusal, surfaced via attacks.common.image_gen.ImageGenerationRefused).
                # That's normally deterministic for a fixed prompt, so relaunching
                # the identical subprocess up to max_attempts more times would just
                # waste time repeating a doomed request -- give up immediately.
                print(
                    f"[{thread_name}] {description} was declined by the model (refusal) -- "
                    f"not retrying further attempts."
                )
                break
            if result:
                print(
                    f"[{thread_name}] {description} failed with exit {result.returncode}"
                )
    failure_tracker.record_failure(description)
    print(f"[{threading.current_thread().name}] Giving up on {description}")
    return False


def generate_base_images(
    obj: str,
    count: int,
    model: str,
    image_api: str,
    extra_prompt: Optional[str],
    output_dir: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    max_retries: int,
    retry_backoff: int,
    redo_existing: bool,
    executor: ThreadPoolExecutor,
    failure_tracker: FailureTracker,
    task_attempts: int,
    attack_type: Optional[str] = None,
    physical_prompt: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    futures = []
    for i in range(1, count + 1):
        output_path = output_dir / f"{obj}_base_{i}.png"
        if output_path.exists() and not redo_existing:
            print(f"[BASE] Skipping existing {output_path}")
            continue
        cmd = [
            sys.executable,
            "-m",
            "attacks.visual_object_replacement.data_generation.generate_base_image",
            obj,
        ]
        if extra_prompt:
            cmd.append(extra_prompt)
        cmd += [
            "--output",
            str(output_path),
            "--model",
            model,
            "--image-api",
            image_api,
            "--max-retries",
            str(max_retries),
            "--retry-backoff",
            str(retry_backoff),
        ]
        if attack_type:
            cmd += ["--attack-type", attack_type]
        if physical_prompt:
            cmd.append("--physical-prompt")
        env = os.environ.copy()
        if api_key:
            env["LLM_API_KEY"] = api_key
        if base_url:
            env["LLM_API_BASE_URL"] = base_url
        description = f"BASE {obj} #{i} ({output_path.name})"
        futures.append(
            executor.submit(
                run_generation_task,
                description,
                cmd,
                env,
                failure_tracker,
                task_attempts,
            )
        )
    return futures


def run_attack(
    attack_type: str,
    original_object: str,
    source_image: Path,
    output_path: Path,
    model: str,
    image_api: str,
    api_key: Optional[str],
    base_url: Optional[str],
    replacement_object: Optional[str] = None,
    executor: ThreadPoolExecutor = None,
    failure_tracker: FailureTracker = None,
    task_attempts: int = MAX_TASK_ATTEMPTS,
    physical_prompt: bool = False,
):
    cmd = [
        sys.executable,
        "-m",
        "attacks.visual_object_replacement.data_generation.generate_attack_image",
        str(source_image),
        "--attack-type",
        attack_type,
        "--original-object",
        original_object,
        "--output",
        str(output_path),
        "--model",
        model,
        "--image-api",
        image_api,
    ]
    if replacement_object:
        cmd += ["--replacement-object", replacement_object]
    if physical_prompt:
        cmd.append("--physical-prompt")

    env = os.environ.copy()
    if api_key:
        env["LLM_API_KEY"] = api_key
    if base_url:
        env["LLM_API_BASE_URL"] = base_url
    attack_label = f"{attack_type}:{replacement_object}" if replacement_object else attack_type
    description = f"ATTACK {attack_label} on {source_image.name}"
    return executor.submit(
        run_generation_task,
        description,
        cmd,
        env,
        failure_tracker,
        task_attempts,
    )


def run_attacks_for_object(
    obj: str,
    base_images_dir: Path,
    attacks_config: list,
    output_root: Path,
    model: str,
    image_api: str,
    api_key: Optional[str],
    base_url: Optional[str],
    redo_existing: bool,
    executor: ThreadPoolExecutor,
    failure_tracker: FailureTracker,
    task_attempts: int,
    physical_prompt: bool = False,
):
    futures = []
    for img_path in sorted(base_images_dir.glob("*.png")):
        for attack in attacks_config:
            attack_type = attack["type"]
            # attack.get("model_version") is a leftover REVE-specific knob
            # (e.g. "reve-edit-fast@20251030"); ignored now that image editing
            # goes through --model / --image-api instead.

            if attack_type == "replace_with_object":
                replacements = attack.get("replacements", [])
                if not replacements:
                    print(f"Skipping replace_with_object (no replacements) for {img_path}")
                    continue
                for repl in replacements:
                    attack_dir = output_root / obj / attack_type / repl.replace(" ", "_")
                    attack_dir.mkdir(parents=True, exist_ok=True)
                    output_path = attack_dir / img_path.name.replace("_base_", f"_attack_{attack_type}_{repl}_")
                    if output_path.exists() and not redo_existing:
                        print(f"[ATTACK:{attack_type}] Skipping existing {output_path}")
                        continue
                    futures.append(
                        run_attack(
                            attack_type=attack_type,
                            original_object=obj,
                            source_image=img_path,
                            output_path=output_path,
                            model=model,
                            image_api=image_api,
                            api_key=api_key,
                            base_url=base_url,
                            replacement_object=repl,
                            executor=executor,
                            failure_tracker=failure_tracker,
                            task_attempts=task_attempts,
                            physical_prompt=physical_prompt,
                        )
                    )
            elif attack_type == "text_replacement":
                replacements = attack.get("replacements", [])
                if not replacements:
                    print(f"Skipping text_replacement (no replacements) for {img_path}")
                    continue
                for repl in replacements:
                    attack_dir = output_root / obj / attack_type / repl.replace(" ", "_")
                    attack_dir.mkdir(parents=True, exist_ok=True)
                    output_path = attack_dir / img_path.name.replace("_base_", f"_attack_{attack_type}_{repl}_")
                    if output_path.exists() and not redo_existing:
                        print(f"[ATTACK:{attack_type}] Skipping existing {output_path}")
                        continue
                    futures.append(
                        run_attack(
                            attack_type=attack_type,
                            original_object=obj,
                            source_image=img_path,
                            output_path=output_path,
                            model=model,
                            image_api=image_api,
                            api_key=api_key,
                            base_url=base_url,
                            replacement_object=repl,
                            executor=executor,
                            failure_tracker=failure_tracker,
                            task_attempts=task_attempts,
                            physical_prompt=physical_prompt,
                        )
                    )
            else:
                attack_dir = output_root / obj / attack_type
                attack_dir.mkdir(parents=True, exist_ok=True)
                output_path = attack_dir / img_path.name.replace("_base_", f"_attack_{attack_type}_")
                if output_path.exists() and not redo_existing:
                    print(f"[ATTACK:{attack_type}] Skipping existing {output_path}")
                    continue
                futures.append(
                    run_attack(
                        attack_type=attack_type,
                        original_object=obj,
                        source_image=img_path,
                        output_path=output_path,
                        model=model,
                        image_api=image_api,
                        api_key=api_key,
                        base_url=base_url,
                        executor=executor,
                        failure_tracker=failure_tracker,
                        task_attempts=task_attempts,
                        physical_prompt=physical_prompt,
                    )
                )
    return futures


def main():
    parser = argparse.ArgumentParser(description="Run data generation pipeline (base + attacks).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "attacks/visual_object_replacement/data_generation/config.json"),
        help="Path to pipeline config JSON.",
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
        help="API shape to use for image generation/editing (see attacks/common/image_gen.py).",
    )
    parser.add_argument("--api-key", type=str, default=None, help="LLM gateway API key (overrides LLM_API_KEY env).")
    parser.add_argument("--base-url", type=str, default=None, help="LLM gateway base URL (overrides LLM_API_BASE_URL env).")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for generation requests.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=2,
        help="Exponential backoff base (seconds).",
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="If set, re-run generation even when output files already exist.",
    )
    parser.add_argument(
        "--physical-prompt",
        action="store_true",
        help=(
            "Pass --physical-prompt through to every generate_base_image.py / "
            "generate_attack_image.py subprocess, forcing the alternate "
            "CREATE_TEMPLATE_PHYSICAL_PROP / EDIT_TEMPLATE_PHYSICAL_PROP templates "
            "(literal physical-object substitution, matching the paper's Appendix "
            "A.2 examples) instead of the default templates. Default behavior is "
            "unchanged unless this flag is passed."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_DEFAULT,
        help="Maximum parallel worker threads (hard capped at 8).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    output_root = Path(cfg.get("output_root", "./data_generated")).resolve()
    base_cfg = cfg["base_generation"]
    objects = cfg["objects"]
    attacks = cfg["attacks"]

    base_count = base_cfg.get("count_per_object", 5)
    # base_cfg.get("model_version") is a leftover REVE-specific knob (e.g.
    # "reve-create@20250915"); ignored now -- use --model / $IMAGE_GEN_MODEL instead.
    extra_prompt = None

    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    base_url = args.base_url or os.environ.get("LLM_API_BASE_URL")
    if not api_key and not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OLLAMA_API_KEY"):
        raise SystemExit(
            "No LLM API key found. Provide --api-key or export LLM_API_KEY "
            "(see attacks/common/llm_client.py for the full resolution order)."
        )

    base_output_root = output_root / "base"
    attack_output_root = output_root / "attacks"
    max_parallel = max(1, min(args.max_parallel, MAX_PARALLEL_DEFAULT))
    print(f"[EXECUTOR] Using up to {max_parallel} parallel threads.")
    failure_tracker = FailureTracker()

    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="GenWorker") as executor:
        base_futures = []
        for obj in objects:
            obj_base_dir = base_output_root / obj
            # Check if any attack is text_replacement to determine attack_type for base generation
            attack_type_for_base = None
            for attack in attacks:
                if attack["type"] == "text_replacement":
                    attack_type_for_base = "text_replacement"
                    break
            base_futures.extend(
                generate_base_images(
                    obj=obj,
                    count=base_count,
                    model=args.model,
                    image_api=args.image_api,
                    extra_prompt=extra_prompt,
                    output_dir=obj_base_dir,
                    api_key=api_key,
                    base_url=base_url,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                    redo_existing=args.redo_existing,
                    executor=executor,
                    failure_tracker=failure_tracker,
                    task_attempts=MAX_TASK_ATTEMPTS,
                    attack_type=attack_type_for_base,
                    physical_prompt=args.physical_prompt,
                )
            )
        for future in base_futures:
            future.result()
        print("[STAGE] Base image generation completed.")

        attack_futures = []
        for obj in objects:
            obj_base_dir = base_output_root / obj
            attack_futures.extend(
                run_attacks_for_object(
                    obj=obj,
                    base_images_dir=obj_base_dir,
                    attacks_config=attacks,
                    output_root=attack_output_root,
                    model=args.model,
                    image_api=args.image_api,
                    api_key=api_key,
                    base_url=base_url,
                    redo_existing=args.redo_existing,
                    executor=executor,
                    failure_tracker=failure_tracker,
                    task_attempts=MAX_TASK_ATTEMPTS,
                    physical_prompt=args.physical_prompt,
                )
            )
        for future in attack_futures:
            future.result()
        print("[STAGE] Attack generation completed.")

    failures = failure_tracker.get_failures()
    if failures:
        print(f"[SUMMARY] {len(failures)} tasks failed after retries:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("[SUMMARY] All tasks completed successfully after retries.")


if __name__ == "__main__":
    main()
