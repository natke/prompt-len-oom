#!/usr/bin/env python3

"""Windows helper for Foundry Local SDK model operations.

Purpose:
- Avoid explicit WebGPU EP registration (due to double registration bug).
- Register all other discoverable EPs.
- List catalog models.
- Optionally download one or more models.

Prerequisites:
- Python package: foundry-local-sdk or foundry-local-sdk-winml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register all non-WebGPU EPs, list Foundry Local models, and optionally download models."
        )
    )
    parser.add_argument(
        "--download",
        nargs="*",
        default=[],
        help=(
            "Model alias or variant id(s) to download, for example "
            "qwen2.5-coder-7b or qwen2.5-coder-7b-instruct-generic-cpu:4"
        ),
    )
    parser.add_argument(
        "--no-register-eps",
        action="store_true",
        help="Skip EP registration step.",
    )
    parser.add_argument(
        "--no-list-models",
        action="store_true",
        help="Skip model listing step.",
    )
    parser.add_argument(
        "--app-name",
        default="FoundryLocalModelDownloader",
        help="Application name passed to Foundry Local SDK configuration.",
    )
    parser.add_argument(
        "--provider-filter",
        nargs="*",
        default=[],
        help=(
            "Optional provider filter keywords, for example trtrtx qnn cpu. "
            "Matches provider, alias, and id text."
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=["alias", "id", "provider", "context-length", "cached", "loaded"],
        default="alias",
        help="Sort key for model listing.",
    )
    parser.add_argument(
        "--desc",
        action="store_true",
        help="Sort in descending order.",
    )
    parser.add_argument(
        "--run-model",
        default="",
        help=(
            "Model alias or variant id to load and run with a summarization prompt, "
            "for example qwen2.5-coder-7b-instruct-generic-cpu:4."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default="",
        help="Optional text file to summarize, for example constit.txt.",
    )
    parser.add_argument(
        "--prompt-text",
        default="",
        help="Optional inline text to summarize (used when --prompt-file is not provided).",
    )
    parser.add_argument(
        "--prompt-tokens",
        "--prompt-length",
        dest="prompt_length",
        type=int,
        default=0,
        help=(
            "Optional token target for source text before sending to the model (whitespace tokenization). "
            "0 keeps original length, larger values repeat text until target is met, "
            "smaller values truncate."
        ),
    )
    parser.add_argument(
        "--summary-instruction",
        default="Summarize the following text:",
        help="Instruction prefix used before the source text.",
    )
    parser.add_argument(
        "--reserve-output-tokens",
        type=int,
        default=512,
        help="Tokens to reserve for generation when fitting prompt to model context length.",
    )
    parser.add_argument(
        "--token-safety-ratio",
        type=float,
        default=0.75,
        help=(
            "Safety multiplier applied when fitting source tokens into model context. "
            "Use lower values when tokenizer expansion is high."
        ),
    )
    return parser.parse_args()


def print_ep_registration(manager: FoundryLocalManager) -> None:
    eps = manager.discover_eps()

    def is_webgpu_ep(name: str) -> bool:
        normalized = (name or "").strip().lower()
        return "webgpu" in normalized

    # De-duplicate names while preserving order; skip blanks and all WebGPU variants.
    target_names: list[str] = []
    seen: set[str] = set()
    for ep in eps:
        raw_name = (ep.name or "").strip()
        if not raw_name or is_webgpu_ep(raw_name):
            continue
        key = raw_name.lower()
        if key in seen:
            continue
        seen.add(key)
        target_names.append(raw_name)

    print("Discovered EPs:")
    for ep in eps:
        marker = "(skip)" if is_webgpu_ep(ep.name) else ""
        print(f"  - {ep.name:20} registered={ep.is_registered} {marker}")

    if not target_names:
        print("No non-WebGPU EPs found to register.")
        return

    print("\nRegistering non-WebGPU EPs:")
    print("  " + ", ".join(target_names))

    def on_progress(ep_name: str, percent: float) -> None:
        print(f"\r  {ep_name:20} {percent:5.1f}%", end="", flush=True)

    result = manager.download_and_register_eps(names=target_names, progress_callback=on_progress)
    print()
    print(f"EP registration status: success={result.success}, status={result.status}")
    if result.registered_eps:
        print("Registered:")
        for name in result.registered_eps:
            print(f"  - {name}")
    if result.failed_eps:
        print("Failed:")
        for name in result.failed_eps:
            print(f"  - {name}")


def _normalize_key(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _provider_text(model) -> str:
    info_provider = getattr(model.info, "execution_provider", None)
    if info_provider:
        return str(info_provider)

    model_id = (model.id or "").lower()
    if "trtrtx" in model_id or "tensorrtrtx" in model_id:
        return "TensorRT RTX"
    if "openvino" in model_id:
        return "OpenVINO"
    if "webgpu" in model_id:
        return "WebGPU"
    if "qnn" in model_id:
        return "QNN"
    if "cuda" in model_id:
        return "CUDA"
    if "dml" in model_id:
        return "DML"
    if "cpu" in model_id:
        return "CPU"
    return "unknown"


def list_models(manager: FoundryLocalManager, provider_filters: list[str], sort_by: str, descending: bool) -> None:
    models = manager.catalog.list_models()

    normalized_filters = [_normalize_key(item) for item in provider_filters if item.strip()]
    if normalized_filters:
        filtered = []
        for model in models:
            provider_text = _provider_text(model)
            haystack = " ".join([provider_text, model.alias or "", model.id or ""])
            haystack_key = _normalize_key(haystack)
            if any(token in haystack_key for token in normalized_filters):
                filtered.append(model)
        models = filtered

    def sort_key(model):
        if sort_by == "id":
            return (model.id or "").lower()
        if sort_by == "provider":
            return _provider_text(model).lower()
        if sort_by == "context-length":
            value = model.info.context_length
            return int(value) if value is not None else -1
        if sort_by == "cached":
            return bool(model.is_cached)
        if sort_by == "loaded":
            return bool(model.is_loaded)
        return (model.alias or "").lower()

    models = sorted(models, key=sort_key, reverse=descending)

    print(f"\nCatalog models ({len(models)}):")
    for model in models:
        context_length = model.info.context_length
        context_text = str(context_length) if context_length is not None else "n/a"
        provider_text = _provider_text(model)
        print(
            f"  - alias={model.alias:35} id={model.id:45} provider={provider_text:14} "
            f"context={context_text:>6} cached={model.is_cached} loaded={model.is_loaded}"
        )


def resolve_model(manager: FoundryLocalManager, identifier: str):
    model = manager.catalog.get_model_variant(identifier)
    if model is not None:
        return model
    model = manager.catalog.get_model(identifier)
    return model


def _tokenize_text(text: str) -> list[str]:
    return text.split()


def _resize_text(text: str, target_length: int) -> str:
    if target_length <= 0:
        return text
    if not text:
        return text

    source_tokens = _tokenize_text(text)
    if not source_tokens:
        return text

    if len(source_tokens) >= target_length:
        return " ".join(source_tokens[:target_length])

    expanded: list[str] = []
    while len(expanded) < target_length:
        expanded.extend(source_tokens)
    return " ".join(expanded[:target_length])


def _resize_mode(original_tokens: int, target_tokens: int, resized_tokens: int) -> str:
    if target_tokens <= 0:
        return "unchanged"
    if resized_tokens < original_tokens:
        return "truncated"
    if resized_tokens > original_tokens:
        return "duplicated"
    return "unchanged"


def _load_source_text(prompt_file: str, prompt_text: str) -> str:
    if prompt_file:
        content = Path(prompt_file).read_text(encoding="utf-8")
        return content.strip()
    return prompt_text.strip()


def _clamp_source_text_to_context(
    source_text: str,
    summary_instruction: str,
    context_length: int | None,
    reserve_output_tokens: int,
    token_safety_ratio: float,
) -> tuple[str, int | None]:
    if context_length is None or context_length <= 0:
        return source_text, None

    instruction_tokens = len(_tokenize_text(summary_instruction.strip()))
    usable_tokens = max(1, context_length - max(0, reserve_output_tokens) - instruction_tokens)
    safe_budget = max(1, int(usable_tokens * max(0.1, min(1.0, token_safety_ratio))))

    source_tokens = _tokenize_text(source_text)
    if len(source_tokens) <= safe_budget:
        return source_text, safe_budget

    clamped = " ".join(source_tokens[:safe_budget])
    return clamped, safe_budget


def run_model_summarization(
    manager: FoundryLocalManager,
    identifier: str,
    prompt_file: str,
    prompt_text: str,
    prompt_length: int,
    summary_instruction: str,
    reserve_output_tokens: int,
    token_safety_ratio: float,
) -> int:
    model = resolve_model(manager, identifier)
    if model is None:
        print(f"ERROR: model not found for identifier '{identifier}'", file=sys.stderr)
        return 1

    try:
        source_text = _load_source_text(prompt_file=prompt_file, prompt_text=prompt_text)
    except OSError as exc:
        print(f"ERROR: failed to read prompt source: {exc}", file=sys.stderr)
        return 1

    if not source_text:
        print(
            "ERROR: no source text provided. Use --prompt-file (for example constit.txt) or --prompt-text.",
            file=sys.stderr,
        )
        return 1

    original_source_token_count = len(_tokenize_text(source_text))
    source_text = _resize_text(source_text, prompt_length)
    resized_source_token_count = len(_tokenize_text(source_text))
    resize_mode = _resize_mode(original_source_token_count, prompt_length, resized_source_token_count)

    context_length = model.info.context_length
    pre_clamp_source_token_count = resized_source_token_count
    source_text, safe_budget = _clamp_source_text_to_context(
        source_text=source_text,
        summary_instruction=summary_instruction,
        context_length=context_length,
        reserve_output_tokens=reserve_output_tokens,
        token_safety_ratio=token_safety_ratio,
    )
    prompt = f"{summary_instruction.strip()}\n\n{source_text}"
    source_token_count = len(_tokenize_text(source_text))
    prompt_token_count = len(_tokenize_text(prompt))
    context_clamped = source_token_count < pre_clamp_source_token_count

    print(f"\nRunning model: id='{model.id}' alias='{model.alias}'")
    print(f"Original source length (tokens): {original_source_token_count}")
    print(
        "Prompt-length transform: "
        f"mode={resize_mode} target={prompt_length if prompt_length > 0 else 'none'} result={resized_source_token_count}"
    )
    if context_length is not None:
        print(f"Model context length (tokens): {context_length}")
    if safe_budget is not None:
        print(
            "Fitted source token budget: "
            f"{safe_budget} (reserve_output={reserve_output_tokens}, safety_ratio={token_safety_ratio})"
        )
    print(f"Context clamp applied: {'yes' if context_clamped else 'no'}")
    print(f"Prompt source length (tokens): {source_token_count}")
    print(f"Prompt total length (tokens): {prompt_token_count}")

    try:
        if not model.is_cached:
            print("Model is not cached yet. Downloading...")
            model.download(progress_callback=lambda p: print(f"\r  download {p:5.1f}%", end="", flush=True))
            print()

        print("Loading model...")
        model.load()

        client = model.get_chat_client()
        response = client.complete_chat(messages=[{"role": "user", "content": prompt}])

        content = response.choices[0].message.content if response and response.choices else ""
        print("\nSummary output:\n")
        print(content)
    except FoundryLocalException as exc:
        print(f"ERROR running model '{identifier}': {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            model.unload()
        except FoundryLocalException:
            pass

    return 0


def download_models(manager: FoundryLocalManager, identifiers: Iterable[str]) -> int:
    failures = 0
    for identifier in identifiers:
        model = resolve_model(manager, identifier)
        if model is None:
            print(f"ERROR: model not found for identifier '{identifier}'", file=sys.stderr)
            failures += 1
            continue

        print(f"\nDownloading: requested='{identifier}' selected-id='{model.id}' alias='{model.alias}'")

        def on_progress(percent: float) -> None:
            print(f"\r  {model.id:45} {percent:5.1f}%", end="", flush=True)

        try:
            model.download(progress_callback=on_progress)
            print()
            print(f"Downloaded: {model.id}")
            try:
                print(f"Path: {model.get_path()}")
            except FoundryLocalException:
                pass
        except FoundryLocalException as exc:
            print()
            print(f"ERROR downloading '{identifier}': {exc}", file=sys.stderr)
            failures += 1
    return failures


def main() -> int:
    args = parse_args()

    if sys.platform != "win32":
        print("Warning: this helper is intended for Windows environments.", file=sys.stderr)

    FoundryLocalManager.initialize(Configuration(app_name=args.app_name))
    manager = FoundryLocalManager.instance

    if not args.no_register_eps:
        print_ep_registration(manager)

    if not args.no_list_models:
        list_models(
            manager,
            provider_filters=args.provider_filter,
            sort_by=args.sort_by,
            descending=args.desc,
        )

    failures = 0
    if args.download:
        failures = download_models(manager, args.download)

    if args.run_model:
        failures += run_model_summarization(
            manager,
            identifier=args.run_model,
            prompt_file=args.prompt_file,
            prompt_text=args.prompt_text,
            prompt_length=args.prompt_length,
            summary_instruction=args.summary_instruction,
            reserve_output_tokens=args.reserve_output_tokens,
            token_safety_ratio=args.token_safety_ratio,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
