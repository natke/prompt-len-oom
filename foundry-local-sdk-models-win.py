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


def list_models(manager: FoundryLocalManager) -> None:
    models = manager.catalog.list_models()
    print(f"\nCatalog models ({len(models)}):")
    for model in models:
        context_length = model.info.context_length
        print(
            f"  - alias={model.alias:35} id={model.id:45} "
            f"context={context_length:>6} cached={model.is_cached} loaded={model.is_loaded}"
        )


def resolve_model(manager: FoundryLocalManager, identifier: str):
    model = manager.catalog.get_model_variant(identifier)
    if model is not None:
        return model
    model = manager.catalog.get_model(identifier)
    return model


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
        list_models(manager)

    failures = 0
    if args.download:
        failures = download_models(manager, args.download)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
