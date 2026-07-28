#!/usr/bin/env python3

"""
Profiles Foundry Local memory behavior: reproduces input-size OOM failures
and detects memory leaks / allocator retention across repeated inferences.

Uses the Foundry Local SDK instead of the Foundry Local CLI.

The script:
  1. Initialises the SDK, registers non-WebGPU EPs, and ensures the service is running.
  2. Auto-detects available models and selects a small chat model if none is specified.
  3. Loads the model and records a memory baseline / post-load footprint.
  4. Sends chat requests with configurable input sizes and iteration counts,
     recording process memory before and after each call.
  5. On completion, runs a linear-regression leak analysis over the collected
     samples (MB per iteration), and distinguishes real leaks from BFCArena
     allocator retention by comparing first-half vs. second-half averages.

Prerequisites:
  - Python packages: foundry-local-sdk (or foundry-local-sdk-winml), psutil
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    pass

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Foundry Local memory behaviour: reproduce input-size OOM failures "
            "and detect memory leaks / allocator retention across repeated inferences."
        ),
    )
    parser.add_argument(
        "--model",
        default="",
        help="Foundry model alias or variant id to test. Default = auto-detect a small model.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Approximate input token counts to try, ascending.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Output tokens to request. Kept small so the failure is driven by INPUT size.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Custom base prompt text to prepend before filler.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=0,
        help="Base prompt token length. 0 = no base prompt, all tokens are input.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to repeat each input size.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Keep testing every size instead of stopping at the first failure.",
    )
    parser.add_argument(
        "--leak-test",
        action="store_true",
        help=(
            "Enables leak-detection mode: pins to a single size, forces >= 20 iterations, "
            "and prints a regression-based verdict at the end."
        ),
    )
    parser.add_argument(
        "--leak-threshold-mb",
        type=int,
        default=2,
        help=(
            "MB/iteration slope below which drift is treated as noise. Default 2. "
            "Slopes between 1x and 10x this value are flagged SUSPICIOUS; above 10x PROBABLE LEAK."
        ),
    )
    parser.add_argument(
        "--load-cycle-test",
        type=int,
        default=0,
        help="Run N load/unload cycles to isolate inter-run memory retention bugs.",
    )
    parser.add_argument(
        "--multi-turn",
        type=int,
        default=0,
        help="Run an N-turn simulated multi-turn conversation.",
    )
    parser.add_argument(
        "--no-register-eps",
        action="store_true",
        help="Skip EP registration step.",
    )
    parser.add_argument(
        "--include-webgpu",
        action="store_true",
        help="Include WebGPU EP in registration (skipped by default due to a bug).",
    )
    parser.add_argument(
        "--app-name",
        default="FoundryOomRepro",
        help="Application name passed to the Foundry Local SDK configuration.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Colour helpers (ANSI)
# ---------------------------------------------------------------------------

_COLOURS = {
    "cyan": "\033[96m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "darkyellow": "\033[33m",
    "darkcyan": "\033[36m",
    "darkgray": "\033[90m",
    "darkred": "\033[31m",
    "reset": "\033[0m",
}


def _c(colour: str, text: str) -> str:
    return f"{_COLOURS.get(colour, '')}{text}{_COLOURS['reset']}"


def write_step(msg: str) -> None:
    print(_c("cyan", f"==> {msg}"))


def fail(msg: str) -> None:
    print(_c("red", f"ERROR: {msg}"), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_process_rss_mb() -> Optional[int]:
    """Return the current process RSS in MB, or None if psutil is unavailable."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process()
        return int(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def get_foundry_process_mb() -> Optional[int]:
    """Return the inference host process RSS in MB.

    The WinML SDK loads the model in-process (no separate daemon), so we
    track our own Python process memory which includes the native ORT/GenAI
    allocations.
    """
    if psutil is None:
        return None
    try:
        proc = psutil.Process()  # current process
        return int(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def get_system_memory_mb() -> Optional[dict]:
    """Return system memory info: UsedMb, AvailableMb, TotalMb."""
    if psutil is None:
        return None
    try:
        vm = psutil.virtual_memory()
        return {
            "UsedMb": int(vm.used / (1024 * 1024)),
            "AvailableMb": int(vm.available / (1024 * 1024)),
            "TotalMb": int(vm.total / (1024 * 1024)),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU memory helpers (NVIDIA + Intel)
# ---------------------------------------------------------------------------

_GPU_MODE: Optional[str] = None  # "nvidia", "intel", or None


def _detect_gpu_mode(model_id: str) -> str:
    """Determine which GPU to monitor based on the model's EP."""
    mid = (model_id or "").lower()
    if "openvino" in mid:
        return "intel"
    if "trtrtx" in mid or "cuda" in mid or "dml" in mid:
        return "nvidia"
    # Default to nvidia if nvidia-smi works, else intel
    import subprocess as _sp
    try:
        _sp.check_output(["nvidia-smi"], timeout=3,
                         creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0))
        return "nvidia"
    except Exception:
        return "intel"


def _get_nvidia_gpu_memory_mb() -> Optional[dict]:
    """Return NVIDIA GPU VRAM via nvidia-smi: UsedMb, FreeMb, TotalMb."""
    import subprocess as _sp
    try:
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits", "--id=0"],
            text=True, timeout=5, creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
        ).strip()
        used, free, total = (int(x.strip()) for x in out.split(","))
        return {"UsedMb": used, "FreeMb": free, "TotalMb": total}
    except Exception:
        return None


def _get_intel_gpu_memory_mb() -> Optional[dict]:
    """Return Intel iGPU memory usage via Windows GPU performance counters.

    Intel iGPUs have no dedicated VRAM; they use shared system memory.
    We identify the Intel adapter as the one with minimal dedicated usage,
    then report its shared + local usage.
    """
    import subprocess as _sp
    try:
        # Query shared and dedicated for all adapters in one call
        ps_cmd = (
            "$shared = (Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples; "
            "$dedicated = (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples; "
            "foreach ($s in $shared) { "
            "  $d = $dedicated | Where-Object { $_.InstanceName -eq $s.InstanceName }; "
            "  $dv = if ($d) { $d.CookedValue } else { 0 }; "
            "  Write-Output ('{0},{1},{2}' -f $s.InstanceName, [int64]$s.CookedValue, [int64]$dv) "
            "}"
        )
        out = _sp.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            text=True, timeout=10,
            creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
        ).strip()
        if not out:
            return None

        # Find the Intel adapter: the one with low dedicated usage but non-trivial shared.
        # NVIDIA discrete GPUs have high dedicated; Intel iGPUs have ~0 dedicated.
        intel_shared = 0
        intel_dedicated = 0
        for line in out.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            shared_bytes = int(parts[1])
            dedicated_bytes = int(parts[2])
            # Intel iGPU: < 100 MB dedicated, meaningful shared usage
            if dedicated_bytes < 100 * 1024 * 1024 and shared_bytes > intel_shared:
                intel_shared = shared_bytes
                intel_dedicated = dedicated_bytes

        if intel_shared == 0:
            return None

        used_mb = int((intel_shared + intel_dedicated) / (1024 * 1024))
        # Intel iGPU shares system RAM; report available system memory as "free"
        sys_mem = get_system_memory_mb()
        total_mb = sys_mem["TotalMb"] if sys_mem else used_mb
        free_mb = sys_mem["AvailableMb"] if sys_mem else 0
        return {"UsedMb": used_mb, "FreeMb": free_mb, "TotalMb": total_mb, "SharedMb": int(intel_shared / (1024 * 1024))}
    except Exception:
        return None


def set_gpu_mode(model_id: str) -> None:
    """Set the GPU monitoring mode based on the active model's EP."""
    global _GPU_MODE
    _GPU_MODE = _detect_gpu_mode(model_id)


def get_gpu_memory_mb() -> Optional[dict]:
    """Return GPU memory info for the active GPU (NVIDIA or Intel)."""
    if _GPU_MODE == "intel":
        return _get_intel_gpu_memory_mb()
    elif _GPU_MODE == "nvidia":
        return _get_nvidia_gpu_memory_mb()
    # Try nvidia first, fall back to intel
    result = _get_nvidia_gpu_memory_mb()
    if result:
        return result
    return _get_intel_gpu_memory_mb()


def get_gpu_used_mb() -> Optional[int]:
    g = get_gpu_memory_mb()
    return g["UsedMb"] if g else None


def get_gpu_label() -> str:
    """Return a label for the active GPU type."""
    if _GPU_MODE == "intel":
        return "iGPU"
    elif _GPU_MODE == "nvidia":
        return "dGPU"
    return "GPU"


def get_system_used_mb() -> Optional[int]:
    s = get_system_memory_mb()
    return s["UsedMb"] if s else None


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def new_prompt(tokens: int, *, prompt: str, prompt_length: int) -> str:
    if prompt_length > 0:
        filler_tokens = max(0, tokens - prompt_length)
        base_filler = "data " * prompt_length
        extra_filler = "data " * filler_tokens
        return (base_filler + extra_filler).strip()
    elif not prompt.strip():
        return ("data " * tokens).strip()
    else:
        base_tokens = len(prompt.split())
        filler_tokens = max(0, tokens - base_tokens)
        filler = ("data " * filler_tokens).strip()
        return f"{prompt} {filler}" if filler_tokens > 0 else prompt


# ---------------------------------------------------------------------------
# Model config discovery + theoretical memory
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    hidden_size: int
    head_dim: int
    dtype: str
    max_position_embeddings: Optional[int]
    source_file: str
    source_schema: str


def _dtype_bytes(dtype: str) -> int:
    d = dtype.lower()
    if "float32" in d or "fp32" in d:
        return 4
    if "int8" in d or "uint8" in d:
        return 1
    return 2  # float16, bfloat16, etc.


def get_kv_cache_mb(tokens: int, config: Optional[ModelConfig]) -> Optional[int]:
    if config is None:
        return None
    bpe = _dtype_bytes(config.dtype)
    nbytes = tokens * config.num_key_value_heads * config.head_dim * config.num_hidden_layers * 2 * bpe
    return int(nbytes / (1024 * 1024))


def get_attention_scores_mb(tokens: int, config: Optional[ModelConfig]) -> Optional[int]:
    if config is None:
        return None
    nbytes = config.num_attention_heads * tokens * tokens * 4
    return int(nbytes / (1024 * 1024))


def _find_model_config(model_alias: str) -> Optional[ModelConfig]:
    """Search common cache paths for genai_config.json or config.json matching the model."""
    import glob
    import pathlib

    cache_paths: list[str] = []

    # Try to get the Foundry cache location
    local_app = os.environ.get("LOCALAPPDATA", "")
    user_profile = os.environ.get("USERPROFILE", "")
    home = pathlib.Path.home()

    for p in [
        os.path.join(local_app, "Microsoft", "FoundryCache") if local_app else "",
        os.path.join(local_app, "FoundryCache") if local_app else "",
        os.path.join(user_profile, ".cache", "foundry", "models") if user_profile else "",
        os.path.join(user_profile, ".foundry", "models") if user_profile else "",
        os.path.join(local_app, "foundry", "models") if local_app else "",
        os.path.join(str(home), ".cache", "foundry", "models"),
        os.path.join(str(home), ".cache", "huggingface", "hub"),
    ]:
        if p and p not in cache_paths:
            cache_paths.append(p)

    # Tokenise the model name for matching
    tokens = [t for t in model_alias.replace(":", "-").replace("_", "-").split("-") if len(t) >= 3]

    for cache_path in cache_paths:
        if not os.path.isdir(cache_path):
            continue

        config_files: list[str] = []
        for name in ("genai_config.json", "config.json"):
            config_files.extend(glob.glob(os.path.join(cache_path, "**", name), recursive=True))

        if not config_files:
            continue

        best_file: Optional[str] = None
        best_score = -1
        for cf in config_files:
            lower = cf.lower()
            score = sum(1 for t in tokens if t.lower() in lower)
            if os.path.basename(cf) == "genai_config.json":
                score += 1
            if score > best_score:
                best_score = score
                best_file = cf

        if not best_file or best_score <= 0:
            continue

        try:
            with open(best_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        try:
            if os.path.basename(best_file) == "genai_config.json":
                dec = data.get("model", {}).get("decoder", {})
                if not dec:
                    continue
                num_attn = dec.get("num_attention_heads")
                num_kv = dec.get("num_key_value_heads", num_attn)
                hid = dec.get("hidden_size", 0)
                head_dim = dec.get("head_size") or dec.get("head_dim") or (hid // num_attn if num_attn else 0)
                max_pos = data.get("model", {}).get("context_length") or dec.get("max_position_embeddings")
                return ModelConfig(
                    num_attention_heads=num_attn,
                    num_key_value_heads=num_kv,
                    num_hidden_layers=dec.get("num_hidden_layers", 0),
                    hidden_size=hid,
                    head_dim=head_dim,
                    dtype="float16",
                    max_position_embeddings=max_pos,
                    source_file=best_file,
                    source_schema="genai_config",
                )
            else:
                cfg = data
                num_attn = cfg.get("num_attention_heads")
                num_kv = cfg.get("num_key_value_heads", num_attn)
                hid = cfg.get("hidden_size", 0)
                head_dim = cfg.get("head_dim") or (hid // num_attn if num_attn else 0)
                dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "float16"
                max_pos = cfg.get("max_position_embeddings")
                return ModelConfig(
                    num_attention_heads=num_attn,
                    num_key_value_heads=num_kv,
                    num_hidden_layers=cfg.get("num_hidden_layers", 0),
                    hidden_size=hid,
                    head_dim=head_dim,
                    dtype=dtype,
                    max_position_embeddings=max_pos,
                    source_file=best_file,
                    source_schema="config",
                )
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# EP registration (skip WebGPU)
# ---------------------------------------------------------------------------

def register_eps(manager: FoundryLocalManager, *, include_webgpu: bool = False) -> None:
    eps = manager.discover_eps()

    skip_webgpu = not include_webgpu
    target_names: list[str] = []
    seen: set[str] = set()
    for ep in eps:
        raw_name = (ep.name or "").strip()
        if not raw_name:
            continue
        if skip_webgpu and "webgpu" in raw_name.lower():
            continue
        key = raw_name.lower()
        if key in seen:
            continue
        seen.add(key)
        target_names.append(raw_name)

    print("Discovered EPs:")
    for ep in eps:
        marker = "(skip)" if (skip_webgpu and "webgpu" in (ep.name or "").lower()) else ""
        print(f"  - {ep.name:20} registered={ep.is_registered} {marker}")

    if not target_names:
        print("No EPs found to register.")
        return

    label = "Registering EPs" if include_webgpu else "Registering non-WebGPU EPs"
    print(f"\n{label}: {', '.join(target_names)}")

    def on_progress(ep_name: str, percent: float) -> None:
        print(f"\r  {ep_name:20} {percent:5.1f}%", end="", flush=True)

    result = manager.download_and_register_eps(names=target_names, progress_callback=on_progress)
    print()
    print(f"EP registration status: success={result.success}, status={result.status}")
    if result.registered_eps:
        print("Registered:", ", ".join(result.registered_eps))
    if result.failed_eps:
        print("Failed:", ", ".join(result.failed_eps))


# ---------------------------------------------------------------------------
# Model resolution helpers
# ---------------------------------------------------------------------------

def resolve_model(manager: FoundryLocalManager, identifier: str):
    """Resolve a model by variant id first, then by alias."""
    model = manager.catalog.get_model_variant(identifier)
    if model is not None:
        return model
    return manager.catalog.get_model(identifier)


def _provider_text(model) -> str:
    info_provider = getattr(model.info, "execution_provider", None)
    if info_provider:
        return str(info_provider)
    model_id = (model.id or "").lower()
    for kw, label in [
        ("trtrtx", "TensorRT RTX"), ("tensorrtrtx", "TensorRT RTX"),
        ("openvino", "OpenVINO"), ("webgpu", "WebGPU"), ("qnn", "QNN"),
        ("cuda", "CUDA"), ("dml", "DML"), ("cpu", "CPU"),
    ]:
        if kw in model_id:
            return label
    return "unknown"


def _is_chat_model(model) -> bool:
    task = getattr(model.info, "task", None) or ""
    if isinstance(task, str):
        return "chat" in task.lower()
    return True  # assume chat if we can't tell


# ---------------------------------------------------------------------------
# Chat invocation via SDK streaming
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    ok: bool
    ms: int
    ttfb_ms: Optional[int]
    content: str
    detail: str


def invoke_chat(
    model_obj,
    messages: list[dict],
    max_tokens: int,
    *,
    _client_cache: dict = {},
) -> ChatResult:
    """Send a streaming chat request via the Foundry Local SDK and collect the response."""
    t0 = time.perf_counter()
    ttfb_ms: Optional[int] = None

    try:
        # Reuse client across calls to avoid re-creating the inference session.
        model_id = id(model_obj)
        if model_id not in _client_cache:
            _client_cache[model_id] = model_obj.get_chat_client()
        client = _client_cache[model_id]
        client.settings.max_tokens = max_tokens
        client.settings.temperature = 0
        response = client.complete_streaming_chat(messages=messages)

        chunk_count = 0
        content_chars = 0
        assistant_content = ""
        finish_reason: Optional[str] = None

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    if chunk_count == 0:
                        ttfb_ms = int((time.perf_counter() - t0) * 1000)
                    assistant_content += delta.content
                    content_chars += len(delta.content)
                    chunk_count += 1
                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if chunk_count > 0:
            parts = [
                f"stream: {chunk_count} chunks",
                f"{content_chars} chars",
            ]
            if ttfb_ms is not None:
                parts.append(f"ttfb={ttfb_ms}ms")
            if finish_reason:
                parts.append(f"finish={finish_reason}")
            return ChatResult(
                ok=True,
                ms=elapsed_ms,
                ttfb_ms=ttfb_ms,
                content=assistant_content,
                detail=f"OK ({', '.join(parts)})",
            )
        else:
            return ChatResult(
                ok=False,
                ms=elapsed_ms,
                ttfb_ms=ttfb_ms,
                content=assistant_content,
                detail="Stream returned no content chunks",
            )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return ChatResult(
            ok=False,
            ms=elapsed_ms,
            ttfb_ms=None,
            content="",
            detail=f"Exception: {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Load / unload cycle test
# ---------------------------------------------------------------------------

def invoke_load_cycle_test(
    manager: FoundryLocalManager,
    model_obj,
    model_alias: str,
    cycles: int,
    initial_proc: int,
    initial_sys: int,
    after_load_proc: int,
    after_load_sys: int,
    model_config: Optional[ModelConfig],
) -> None:
    print()
    write_step(f"Load/unload cycle test: {cycles} cycles (isolates the inter-run memory retention bug)")
    print(f"    Model : {model_alias}")
    print("    Each cycle: unload -> sample -> load -> sample.")

    if model_config and model_config.max_position_embeddings:
        kv_pre = get_kv_cache_mb(model_config.max_position_embeddings, model_config)
        attn_pk = get_attention_scores_mb(model_config.max_position_embeddings, model_config)
        print(_c("darkgray",
            f"    Per-request headroom @ ctx={model_config.max_position_embeddings}: "
            f"KV pre-alloc {kv_pre} MB (in load totals) + Attn scratch ~{attn_pk} MB (O(N^2), per request)."))
    print()

    fmt = "{:<6}{:<13}{:<10}{:<9}{:<10}{:<9}{}"
    print(_c("yellow", fmt.format("Cycle", "Action", "Proc(MB)", "[d]", "Sys(MB)", "[d]", "Notes")))
    print("-" * 92)

    def _sign(val: int) -> str:
        return f"+{val}" if val >= 0 else str(val)

    # Print cycle 1 (the initial load that already happened)
    print(_c("darkcyan",
        fmt.format(1, "pre-load", initial_proc, "-", initial_sys, "-", "from baseline")))
    print(_c("cyan",
        fmt.format(1, "post-load", after_load_proc,
                   _sign(after_load_proc - initial_proc),
                   after_load_sys,
                   _sign(after_load_sys - initial_sys),
                   "initial load")))

    prev_post_load_proc = after_load_proc
    prev_post_load_sys = after_load_sys
    load_deltas: list[int] = []
    unload_freed_list: list[int] = []
    cycle_drift: list[int] = []

    for c in range(2, cycles + 1):
        # --- Unload ---
        try:
            model_obj.unload()
        except FoundryLocalException:
            pass
        time.sleep(1)

        proc_u = get_foundry_process_mb()
        sys_u = get_system_used_mb()
        if proc_u is None or sys_u is None:
            print(_c("yellow", f"    Cycle {c}: could not sample memory after unload, skipping."))
            continue

        u_delta_proc = proc_u - prev_post_load_proc
        u_delta_sys = sys_u - prev_post_load_sys
        freed = max(0, -u_delta_proc)
        unload_freed_list.append(freed)

        colour = "red" if freed < 500 else "darkcyan"
        print(_c(colour,
            fmt.format(c, "post-unload", proc_u, _sign(u_delta_proc),
                       sys_u, _sign(u_delta_sys), f"unload freed {freed} MB")))

        # --- Load ---
        try:
            model_obj.load()
        except FoundryLocalException as exc:
            print(_c("red", f"    Cycle {c}: load failed: {exc}"))
            continue
        time.sleep(1)

        proc_l = get_foundry_process_mb()
        sys_l = get_system_used_mb()
        if proc_l is None or sys_l is None:
            print(_c("yellow", f"    Cycle {c}: could not sample memory after load, skipping."))
            continue

        l_delta_proc = proc_l - proc_u
        l_delta_sys = sys_l - sys_u
        load_deltas.append(l_delta_proc)

        print(_c("cyan",
            fmt.format(c, "post-load", proc_l, _sign(l_delta_proc),
                       sys_l, _sign(l_delta_sys), "")))

        drift = proc_l - prev_post_load_proc
        cycle_drift.append(drift)

        prev_post_load_proc = proc_l
        prev_post_load_sys = sys_l

    if not cycle_drift:
        print()
        print(_c("yellow", "No cycles completed."))
        return

    mean_load = int(sum(load_deltas) / len(load_deltas)) if load_deltas else 0
    mean_freed = int(sum(unload_freed_list) / len(unload_freed_list)) if unload_freed_list else 0
    mean_drift = int(sum(cycle_drift) / len(cycle_drift))
    min_load, max_load = (min(load_deltas), max(load_deltas)) if load_deltas else (0, 0)
    min_freed, max_freed = (min(unload_freed_list), max(unload_freed_list)) if unload_freed_list else (0, 0)
    min_drift, max_drift = min(cycle_drift), max(cycle_drift)

    print()
    print(_c("yellow", f"Cycle summary (n={len(cycle_drift)} cycles measured):"))
    print(f"  Load added      : mean {mean_load:6} MB   range [{min_load}, {max_load}]")
    print(f"  Unload freed    : mean {mean_freed:6} MB   range [{min_freed}, {max_freed}]")
    print(f"  Net drift/cycle : mean {mean_drift:6} MB   range [{min_drift}, {max_drift}]")
    print()

    if mean_drift >= 500:
        print(_c("red", "VERDICT: LEAK in the unload path."))
        print(_c("red", f"  Load costs ~{mean_load} MB but unload only frees ~{mean_freed} MB."))
        print(_c("red", f"  Every unload+reload retains ~{mean_drift} MB (CPU-side)."))
        sys_now = get_system_memory_mb()
        if sys_now and mean_drift > 0:
            cycles_left = int(sys_now["AvailableMb"] / mean_drift)
            print(_c("red",
                f"  Extrapolation: ~{cycles_left} more cycles before OOM "
                f"(available {sys_now['AvailableMb']} MB / drift {mean_drift} MB)."))
    elif mean_drift >= 100:
        print(_c("darkyellow",
            f"VERDICT: SUSPICIOUS. ~{mean_drift} MB retained per unload+reload cycle."))
        print(_c("darkyellow",
            "  Could be caching (kernel/JIT/staging) that plateaus, or a slow leak. Run more cycles to distinguish."))
    else:
        print(_c("green",
            "VERDICT: OK. Unload releases memory equivalent to load; no inter-run leak detected."))


# ---------------------------------------------------------------------------
# Multi-turn conversation test
# ---------------------------------------------------------------------------

def invoke_multi_turn_test(
    model_obj,
    turns: int,
    tokens_per_turn: int,
    output_tokens: int,
    context_length: int,
    model_config: Optional[ModelConfig],
    *,
    prompt_text: str,
    prompt_length_arg: int,
) -> None:
    print()
    write_step(f"Multi-turn conversation test: {turns} turns of ~{tokens_per_turn} user tokens each")
    print("    Each turn appends a new user message + prior assistant reply to the history,")
    print("    then sends the whole transcript. Context grows monotonically.")
    if context_length > 0:
        print(f"    Context cap: {context_length} tokens (turns that would exceed it are skipped).")
    else:
        print(_c("darkyellow",
            "    Context cap: unknown - will run until failure or --multi-turn N reached."))
    print()

    fmt = "{:<6}{:<11}{:<22}{:<22}{:<7}{:<9}{:<9}{:<8}"
    print(_c("yellow",
        fmt.format("Turn", "InputTok", "Proc(MB)[d]", "Sys(MB)[d]", "%Ctx", "TTFT(s)", "Total(s)", "Result")))
    print("-" * 98)

    base_proc = get_foundry_process_mb()
    base_sys = get_system_used_mb()
    bp_str = str(base_proc) if base_proc is not None else "n/a"
    bs_str = str(base_sys) if base_sys is not None else "n/a"
    print(_c("darkcyan",
        fmt.format("pre", "-", bp_str, bs_str, "-", "-", "-", "BASELINE")))

    messages: list[dict] = []
    cumulative_context = 0
    turn_deltas: list[int] = []
    turns_completed = 0
    last_proc_after: Optional[int] = None
    last_sys_after: Optional[int] = None

    def _sign(val: int) -> str:
        return f"+{val}" if val >= 0 else str(val)

    for t in range(1, turns + 1):
        new_user_prompt = new_prompt(tokens_per_turn, prompt=prompt_text, prompt_length=prompt_length_arg)
        projected_input_tokens = cumulative_context + tokens_per_turn

        if context_length > 0 and (projected_input_tokens + output_tokens) > context_length:
            pct_str = f"{int((projected_input_tokens / context_length) * 100)}%"
            print(_c("darkgray",
                fmt.format(t, projected_input_tokens, "skip", "skip", pct_str, "-", "-", "CTX-CAP")))
            print(_c("darkgray", "    Next turn would exceed the context window - stopping."))
            break

        messages.append({"role": "user", "content": new_user_prompt})

        mem_before = get_foundry_process_mb()
        sys_before = get_system_used_mb()

        r = invoke_chat(model_obj, messages, output_tokens)

        mem_after = get_foundry_process_mb()
        sys_after = get_system_used_mb()
        mem_delta = int(mem_after - mem_before) if mem_before is not None and mem_after is not None else None
        sys_delta = int(sys_after - sys_before) if sys_before is not None and sys_after is not None else None

        result_str = "OK" if r.ok else "FAIL"
        colour = "green" if r.ok else "red"
        pct_str = f"{int((projected_input_tokens / context_length) * 100)}%" if context_length > 0 else "n/a"
        proc_str = f"{mem_after} ({_sign(mem_delta)})" if mem_after is not None and mem_delta is not None else "n/a"
        sys_str = f"{sys_after} ({_sign(sys_delta)})" if sys_after is not None and sys_delta is not None else "n/a"
        time_str = round(r.ms / 1000, 1)
        ttft_str = str(round(r.ttfb_ms / 1000, 2)) if r.ttfb_ms is not None else "-"

        print(_c(colour,
            fmt.format(t, projected_input_tokens, proc_str, sys_str, pct_str, ttft_str, time_str, result_str)))

        # Theoretical reference every 5 turns (and on turn 1)
        if model_config and (t == 1 or t % 5 == 0):
            kv_mb = get_kv_cache_mb(projected_input_tokens, model_config)
            attn_mb = get_attention_scores_mb(projected_input_tokens, model_config)
            if kv_mb is not None:
                print(_c("darkgray",
                    f"             theoretical @ input={projected_input_tokens}: KV={kv_mb} MB, Attn(O(N^2))={attn_mb} MB"))

        if not r.ok:
            err_detail = r.detail if r.detail.strip() else "(empty error)"
            print(_c("darkred", f"    Full error: {err_detail}"))
            break

        messages.append({"role": "assistant", "content": r.content or ""})
        cumulative_context = projected_input_tokens + output_tokens

        if mem_delta is not None:
            turn_deltas.append(mem_delta)
        turns_completed = t
        last_proc_after = mem_after
        last_sys_after = sys_after

    print()
    print(_c("yellow", "Multi-turn summary:"))
    print(f"  Turns completed             : {turns_completed} / {turns}")
    print(f"  Estimated final context     : ~{cumulative_context} tokens")
    if turn_deltas:
        mean_d = int(sum(turn_deltas) / len(turn_deltas))
        max_d = max(turn_deltas)
        min_d = min(turn_deltas)
        print(f"  Process delta per turn      : mean {_sign(mean_d)} MB   range [{_sign(min_d)}, {_sign(max_d)}]")
    if last_proc_after is not None:
        print(f"  Final process memory        : {last_proc_after} MB")
    if last_sys_after is not None:
        print(f"  Final system memory used    : {last_sys_after} MB")


# ---------------------------------------------------------------------------
# Leak analysis
# ---------------------------------------------------------------------------

@dataclass
class LeakStats:
    n: int
    slope: float
    first: float
    last: float
    peak: float
    trough: float
    delta: float


def get_leak_stats(series: list[float]) -> Optional[LeakStats]:
    n = len(series)
    if n < 2:
        return None
    xs = list(range(n))
    sum_x = sum(xs)
    sum_y = sum(series)
    sum_xy = sum(x * y for x, y in zip(xs, series))
    sum_x2 = sum(x * x for x in xs)
    denom = (n * sum_x2) - (sum_x * sum_x)
    slope = ((n * sum_xy) - (sum_x * sum_y)) / denom if denom != 0 else 0.0
    return LeakStats(
        n=n,
        slope=slope,
        first=series[0],
        last=series[-1],
        peak=max(series),
        trough=min(series),
        delta=series[-1] - series[0],
    )


def write_leak_verdict(label: str, slope: float, threshold: int) -> None:
    abs_slope = abs(slope)
    if abs_slope < threshold:
        print(_c("green",
            f"    {label} verdict : NO LEAK - drift within allocator noise (< {threshold} MB/iter)"))
    elif abs_slope < threshold * 10:
        print(_c("yellow",
            f"    {label} verdict : SUSPICIOUS - {slope:.1f} MB/iter growth"))
    else:
        projected = int(slope * 1000)
        print(_c("red",
            f"    {label} verdict : PROBABLE LEAK - {slope:.1f} MB/iter (~{projected} MB / 1000 iters)"))


def run_leak_analysis(measurements: list[dict], leak_threshold_mb: int) -> None:
    successful = [m for m in measurements if m["ok"] and m["mem_after"] is not None]
    if len(successful) < 5:
        return

    print()
    write_step("Leak analysis (post-inference memory over iterations)")

    # Group by size
    groups: dict[int, list[dict]] = {}
    for m in successful:
        groups.setdefault(m["size"], []).append(m)

    for size, samples in sorted(groups.items()):
        if len(samples) < 3:
            continue

        warm = samples[1:] if len(samples) > 3 else samples

        proc_series = [float(s["mem_after"]) for s in warm]
        proc_stats = get_leak_stats(proc_series)
        if proc_stats is None:
            continue

        sys_stats: Optional[LeakStats] = None
        sys_series = [float(s["sys_after"]) for s in warm if s["sys_after"] is not None]
        if len(sys_series) >= 2:
            sys_stats = get_leak_stats(sys_series)

        print()
        print(_c("cyan",
            f"  Size {size} tokens x {len(samples)} iterations (post-warmup n={proc_stats.n})"))
        print(f"    Process : first={proc_stats.first:.0f} MB, last={proc_stats.last:.0f} MB, "
              f"trough/peak={proc_stats.trough:.0f}/{proc_stats.peak:.0f} MB, "
              f"drift={proc_stats.delta:+.0f} MB, slope={proc_stats.slope:.2f} MB/iter")
        write_leak_verdict("Process", proc_stats.slope, leak_threshold_mb)

        if sys_stats:
            print(f"    System  : first={sys_stats.first:.0f} MB, last={sys_stats.last:.0f} MB, "
                  f"trough/peak={sys_stats.trough:.0f}/{sys_stats.peak:.0f} MB, "
                  f"drift={sys_stats.delta:+.0f} MB, slope={sys_stats.slope:.2f} MB/iter")
            write_leak_verdict("System ", sys_stats.slope, leak_threshold_mb)

        # Distinguish retention vs. real leak
        if len(samples) >= 10 and abs(proc_stats.slope) >= leak_threshold_mb:
            half = len(samples) // 2
            fh_avg = sum(s["mem_after"] for s in samples[:half]) / half
            sh_avg = sum(s["mem_after"] for s in samples[half:]) / (len(samples) - half)
            half_delta = sh_avg - fh_avg
            print(_c("darkgray",
                f"    First-half avg : {fh_avg:.0f} MB    Second-half avg: {sh_avg:.0f} MB    Delta: {half_delta:+.0f} MB"))
            if abs(half_delta) < leak_threshold_mb:
                print(_c("green",
                    "    -> Process growth plateaued: likely BFCArena retention, NOT a leak."))
            else:
                print(_c("red",
                    "    -> Process growth continues across halves: LEAK CONFIRMED."))

        # Cross-check
        if (sys_stats and abs(sys_stats.slope) >= leak_threshold_mb
                and abs(proc_stats.slope) < leak_threshold_mb):
            print(_c("yellow",
                "    -> System growing but process flat: leak is in another process (or kernel/driver)."))

    print()
    print(_c("darkgray", "Leak detection guide:"))
    print(_c("darkgray", f"  < {leak_threshold_mb} MB/iter          -> noise / allocator internal state"))
    print(_c("darkgray", f"  {leak_threshold_mb} - {leak_threshold_mb * 10} MB/iter  -> suspicious (could be retention)"))
    print(_c("darkgray", f"  > {leak_threshold_mb * 10} MB/iter         -> probable leak"))
    print(_c("darkgray", "  Plateau in 2nd half -> BFCArena retention (not a leak)"))
    print(_c("darkgray", "  Continued growth    -> real leak"))
    print(_c("darkgray", "  System>0 & Process=0 -> leak in another process"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # LeakTest defaults
    if args.leak_test:
        if len(args.sizes) > 1:
            print(_c("darkyellow",
                f"    [LeakTest] Multiple --sizes given; using only the first ({args.sizes[0]}) for leak detection."))
            args.sizes = [args.sizes[0]]
        if args.iterations < 10:
            print(_c("darkyellow",
                f"    [LeakTest] Bumping --iterations from {args.iterations} to 20 (minimum for regression)."))
            args.iterations = 20

    # -- 0) Initialise SDK and service -----------------------------------------
    write_step("Initialising Foundry Local SDK...")
    FoundryLocalManager.initialize(Configuration(app_name=args.app_name))
    manager = FoundryLocalManager.instance

    if not args.no_register_eps:
        register_eps(manager, include_webgpu=args.include_webgpu)

    # -- 1) Unload any previously loaded models for a clean baseline -----------
    print()
    print(_c("darkgray", "    Clearing any previously loaded models..."))
    pre_unload_proc = get_foundry_process_mb()
    pre_unload_sys = get_system_used_mb()
    unloaded_count = 0
    try:
        for m in manager.catalog.list_models():
            if m.is_loaded:
                print(_c("darkgray", f"      Unloading {m.id}..."))
                try:
                    m.unload()
                    unloaded_count += 1
                except FoundryLocalException:
                    pass
    except FoundryLocalException:
        pass
    time.sleep(1)

    # BASELINE
    # Set initial GPU mode from --model arg if available (refined after model loads)
    if args.model:
        set_gpu_mode(args.model)
    mem_baseline = get_foundry_process_mb()
    sys_baseline = get_system_memory_mb()
    gpu_baseline = get_gpu_memory_mb()

    unload_freed_proc: Optional[int] = None
    if unloaded_count > 0 and pre_unload_proc is not None and mem_baseline is not None:
        unload_freed_proc = pre_unload_proc - mem_baseline

    print()
    print(_c("yellow", "    MEMORY BASELINE (service only, no model loaded):"))
    bp_str = f"{mem_baseline} MB" if mem_baseline is not None else "n/a (process not found)"
    print(_c("yellow", f"      process memory (RSS)        : {bp_str}"))
    if sys_baseline:
        print(_c("yellow",
            f"      system memory used         : {sys_baseline['UsedMb']} MB / "
            f"{sys_baseline['TotalMb']} MB total ({sys_baseline['AvailableMb']} MB available)"))
    if gpu_baseline:
        shared_note = f" (shared system RAM)" if _GPU_MODE == "intel" else ""
        print(_c("yellow",
            f"      {get_gpu_label()} memory used           : {gpu_baseline['UsedMb']} MB / "
            f"{gpu_baseline['TotalMb']} MB total ({gpu_baseline['FreeMb']} MB free){shared_note}"))

    # Unload effectiveness report
    if unloaded_count > 0 and unload_freed_proc is not None:
        print()
        print(_c("yellow", f"    UNLOAD EFFECTIVENESS ({unloaded_count} model(s) unloaded):"))
        print(f"      Process memory before unload: {pre_unload_proc:6} MB")
        print(f"      Process memory after  unload: {mem_baseline:6} MB")
        sign = "" if unload_freed_proc >= 0 else "-"
        print(f"      => unload freed             : {sign}{abs(unload_freed_proc):5} MB")

        if pre_unload_proc is not None and pre_unload_proc > 1000 and unload_freed_proc < 500:
            print()
            print(_c("red", "    !! ALERT: foundry model unload appears to NOT be freeing CPU-side memory. !!"))
            print(_c("red",
                f"    !! Held {pre_unload_proc} MB before unload, still holding {mem_baseline} MB after "
                f"(freed only {unload_freed_proc} MB). !!"))
        elif pre_unload_proc is not None and pre_unload_proc > 1000 and unload_freed_proc < (pre_unload_proc * 0.5):
            print()
            print(_c("darkyellow", "    !! WARNING: unload freed less than half of what was in memory."))
            print(_c("darkyellow",
                f"    !! ({unload_freed_proc} MB freed out of {pre_unload_proc} MB held). Partial retention."))

    # Inter-run leak detector
    if mem_baseline is not None and mem_baseline > 1000:
        print()
        print(_c("red", f"    WARNING: baseline is {mem_baseline} MB even after unloading all models."))
        print(_c("red", "    A freshly-started foundrylocald with no model is typically ~150-300 MB."))
        print(_c("red", "    This means a previous load left CPU-side memory in the process (inter-run retention)."))

    # -- 2) Select model -------------------------------------------------------
    models = manager.catalog.list_models()
    chat_models = [m for m in models if _is_chat_model(m)]

    if not args.model:
        write_step("Selecting a small Chat model...")
        if not chat_models:
            fail("No Chat models found in the catalog.")

        # Prefer smaller models
        small_patterns = [
            "0.5b", "0.6b", "1.5b", "1.7b", "1b", "2b", "3b",
            "qwen2.5-0.5b", "qwen2.5-1.5b", "phi-3.5-mini", "phi-4-mini",
        ]
        candidates: list = []
        for pat in small_patterns:
            for m in chat_models:
                alias = (m.alias or "").lower()
                mid = (m.id or "").lower()
                if pat in alias or pat in mid:
                    if m not in candidates:
                        candidates.append(m)

        if not candidates:
            candidates = chat_models[:5]

        print(f"    Candidates: {', '.join(m.alias or m.id for m in candidates)}")
    else:
        resolved = resolve_model(manager, args.model)
        if resolved is None:
            fail(f"Model '{args.model}' not found in catalog.")
        candidates = [resolved]

    # -- 3) Download and load model --------------------------------------------
    model_obj = None
    model_alias = ""
    for cand in candidates:
        alias = cand.alias or cand.id
        write_step(f"Downloading '{alias}' if needed...")
        try:
            if not cand.is_cached:
                def on_dl_progress(percent: float) -> None:
                    print(f"\r  download {percent:5.1f}%", end="", flush=True)
                cand.download(progress_callback=on_dl_progress)
                print()
        except FoundryLocalException as exc:
            print(_c("yellow", f"    '{alias}' failed to download: {exc} - trying next."))
            continue

        write_step(f"Loading '{alias}' into the service...")
        try:
            cand.load()
        except FoundryLocalException as exc:
            print(_c("yellow", f"    '{alias}' failed to LOAD: {exc} - trying next."))
            continue

        model_obj = cand
        model_alias = alias
        break

    if model_obj is None:
        fail("Could not load any model.")

    api_model = model_obj.id or model_alias
    set_gpu_mode(api_model)
    # Re-take GPU baseline if mode changed (e.g. auto-detected model differs from --model arg)
    gpu_baseline = get_gpu_memory_mb()
    print(f"    Loaded model: {model_alias} (API id: {api_model})")
    print(f"    GPU monitoring: {get_gpu_label()} ({'Intel iGPU shared memory' if _GPU_MODE == 'intel' else 'NVIDIA dedicated VRAM'})")

    # Measure memory after load
    mem_after_load = get_foundry_process_mb()
    sys_after_load = get_system_memory_mb()
    gpu_after_load = get_gpu_memory_mb()
    print(_c("cyan",
        f"    Memory after load: {mem_after_load} MB (process)" if mem_after_load is not None else "    Memory after load: n/a"))
    if sys_after_load:
        sys_delta = (sys_after_load["UsedMb"] - sys_baseline["UsedMb"]) if sys_baseline else None
        delta_str = f" (+{sys_delta} MB system)" if sys_delta is not None else ""
        print(_c("cyan",
            f"                       {sys_after_load['UsedMb']} MB system used / "
            f"{sys_after_load['AvailableMb']} MB available{delta_str}"))
    if gpu_after_load:
        gpu_delta = (gpu_after_load["UsedMb"] - gpu_baseline["UsedMb"]) if gpu_baseline else None
        gpu_delta_str = f" (+{gpu_delta} MB)" if gpu_delta is not None else ""
        print(_c("cyan",
            f"                       {gpu_after_load['UsedMb']} MB {get_gpu_label()} memory used / "
            f"{gpu_after_load['FreeMb']} MB free{gpu_delta_str}"))

    # Try to get model config
    model_config = _find_model_config(model_alias)
    if model_config:
        gqa_note = (f" (GQA {model_config.num_attention_heads}:{model_config.num_key_value_heads})"
                     if model_config.num_key_value_heads < model_config.num_attention_heads else "")
        print(_c("darkgray",
            f"    Model architecture: {model_config.num_attention_heads} attn heads / "
            f"{model_config.num_key_value_heads} KV heads{gqa_note}, "
            f"{model_config.num_hidden_layers} layers, head_dim={model_config.head_dim}, "
            f"dtype={model_config.dtype}"))
        if model_config.source_file:
            print(_c("darkgray",
                f"    Model config source: {model_config.source_file} ({model_config.source_schema})"))
    else:
        print(_c("darkgray",
            "    Model config: not found (looked for genai_config.json / config.json in Foundry cache + HF caches)"))
        print(_c("darkgray", "    -> theoretical KV/Attn memory estimates will be skipped."))

    # Memory breakdown
    if mem_after_load is not None and mem_baseline is not None:
        print()
        print(_c("cyan", "    MEMORY BREAKDOWN AFTER MODEL LOAD:"))
        print(f"      Baseline (service only):        {mem_baseline} MB")
        print(f"      Total after load:              {mem_after_load} MB")

        model_loaded_in_memory = mem_after_load - mem_baseline

        kv_prealloc_mb: Optional[int] = None
        if model_config and model_config.max_position_embeddings:
            kv_prealloc_mb = get_kv_cache_mb(model_config.max_position_embeddings, model_config)

        attn_peak_mb: Optional[int] = None
        if model_config and model_config.max_position_embeddings:
            attn_peak_mb = get_attention_scores_mb(model_config.max_position_embeddings, model_config)

        print()
        print(_c("darkgray", "    ANALYSIS:"))
        print(_c("darkgray", f"      Service infrastructure:        {mem_baseline} MB"))
        print(_c("darkgray", f"      Model + overhead combined:     {model_loaded_in_memory} MB"))
        if kv_prealloc_mb is not None:
            residual = model_loaded_in_memory - kv_prealloc_mb
            print(_c("darkgray",
                f"      |--- KV cache pre-alloc @ ctx={model_config.max_position_embeddings}: "
                f"{kv_prealloc_mb} MB ({model_config.dtype} dtype, GQA-aware)"))
            print(_c("darkgray", f"      \\--- Residual (runtime/allocator): {residual} MB"))

        if attn_peak_mb is not None:
            print()
            print(_c("darkgray", "    PREDICTED PEAK REQUEST HEADROOM (not in load totals above):"))
            print(_c("darkgray",
                f"      Attention scratch @ ctx={model_config.max_position_embeddings}: "
                f"~{attn_peak_mb} MB   (fp32, single layer alive, O(N^2))"))
            print(_c("darkgray",
                f"      => Total needed for a full-context request: ~{mem_after_load + attn_peak_mb} MB process"))

        print()
        print(_c("darkgray", "    Notes:"))
        print(_c("darkgray",
            "      * KV pre-alloc assumes past_present_share_buffer=true (full-context buffer at load)."))
        print(_c("darkgray",
            "      * Residual = ONNX Runtime session state + graph compilation + allocator overhead."))
        print(_c("darkgray",
            "      * Attention scratch is NOT allocated at load time; it grows per-request as O(N^2)"))
        print(_c("darkgray",
            "        of input length. The value above is the ceiling at max context."))

    # -- 3a) Optional: load/unload cycle test ----------------------------------
    if args.load_cycle_test > 0:
        if (mem_baseline is None or mem_after_load is None
                or sys_baseline is None or sys_after_load is None):
            fail("LoadCycleTest requires baseline + after-load memory samples, but one is missing.")
        invoke_load_cycle_test(
            manager=manager,
            model_obj=model_obj,
            model_alias=model_alias,
            cycles=args.load_cycle_test,
            initial_proc=mem_baseline,
            initial_sys=sys_baseline["UsedMb"],
            after_load_proc=mem_after_load,
            after_load_sys=sys_after_load["UsedMb"],
            model_config=model_config,
        )
        return 0

    # -- 3b) Get context length ------------------------------------------------
    context_length: Optional[int] = None
    try:
        cl = model_obj.info.context_length
        if cl:
            context_length = int(cl)
    except Exception:
        pass

    if context_length:
        print(f"    Context window : {context_length} tokens")
    else:
        print(_c("darkgray", "    Context window : unknown"))

    # -- 3bb) Optional: multi-turn conversation test ---------------------------
    if args.multi_turn > 0:
        tokens_per_turn = args.sizes[0] if args.sizes else 128
        ctx_arg = context_length if context_length and context_length > 0 else 0
        invoke_multi_turn_test(
            model_obj=model_obj,
            turns=args.multi_turn,
            tokens_per_turn=tokens_per_turn,
            output_tokens=args.max_tokens,
            context_length=ctx_arg,
            model_config=model_config,
            prompt_text=args.prompt,
            prompt_length_arg=args.prompt_length,
        )
        return 0

    # -- 3c) Display available Chat models -------------------------------------
    print()
    write_step("Available Chat models:")
    for m in chat_models:
        cl_val = getattr(m.info, "context_length", None)
        ctx_str = str(cl_val) if cl_val else "unknown"
        provider = _provider_text(m)
        marker = "-> " if (m.alias or m.id) == model_alias else "   "
        print(f"  {marker}{m.alias or '':40} provider={provider:14} context={ctx_str:>6} "
              f"cached={m.is_cached} loaded={m.is_loaded}")
    print()

    # -- 4) Step up the input size ---------------------------------------------
    write_step(f"Stepping up input size (output capped at {args.max_tokens} tokens to isolate input handling)...")
    if args.prompt_length > 0:
        print(f"    Prompt: {args.prompt_length}-token base + filler to reach each size")
    elif not args.prompt.strip():
        print("    Prompt: filler text (data data data...)")
    else:
        print(f"    Prompt: custom base + filler (base ~{len(args.prompt.split())} tokens, then filler to reach size)")

    if context_length:
        print(f"    Skipping any size where input+output would exceed the {context_length}-token context window.")
    else:
        print(_c("darkyellow",
            "    Context window not detected - all test sizes will be attempted (Ctx% will be incorrect)."))
    print()

    print(_c("cyan", "Starting memory profiling..."))
    print()

    pre_mem_proc = get_foundry_process_mb()
    pre_mem_sys = get_system_used_mb()
    pre_mem_gpu = get_gpu_used_mb()

    gpu_lbl = get_gpu_label()
    print(_c("cyan", "Memory stages:"))
    print(_c("darkgray",
        f"  Baseline (service only)     : "
        f"{mem_baseline if mem_baseline is not None else 'n/a'} MB process / "
        f"{sys_baseline['UsedMb'] if sys_baseline else 'n/a'} MB system / "
        f"{gpu_baseline['UsedMb'] if gpu_baseline else 'n/a'} MB {gpu_lbl}"))
    print(_c("darkgray",
        f"  After model load            : "
        f"{mem_after_load if mem_after_load is not None else 'n/a'} MB process / "
        f"{sys_after_load['UsedMb'] if sys_after_load else 'n/a'} MB system / "
        f"{gpu_after_load['UsedMb'] if gpu_after_load else 'n/a'} MB {gpu_lbl}"))
    print(_c("darkgray",
        f"  Before first inference      : "
        f"{pre_mem_proc if pre_mem_proc is not None else 'n/a'} MB process / "
        f"{pre_mem_sys if pre_mem_sys is not None else 'n/a'} MB system / "
        f"{pre_mem_gpu if pre_mem_gpu is not None else 'n/a'} MB {gpu_lbl}"))
    print(_c("darkgray", "  (KV cache is allocated during graph compilation on the first run.)"))
    print()

    def _sign(val: int) -> str:
        return f"+{val}" if val >= 0 else str(val)

    gpu_col_label = f"{get_gpu_label()}(MB)[d]"
    header_fmt = "{:<11}{:<10}{:<22}{:<22}{:<18}{:<10}{:<7}{:<9}{:<9}{:<8}"
    print(_c("yellow",
        header_fmt.format("InputTok", "MaxOut", "Proc(MB)[d]", "Sys(MB)[d]",
                          gpu_col_label, "Attn(MB)", "%Ctx", "TTFT(s)", "Total(s)", "Result")))
    print("-" * 126)

    # Pre-input baseline row
    p_str = str(pre_mem_proc) if pre_mem_proc is not None else "n/a"
    s_str = str(pre_mem_sys) if pre_mem_sys is not None else "n/a"
    g_str = str(pre_mem_gpu) if pre_mem_gpu is not None else "n/a"
    print(_c("darkcyan",
        header_fmt.format("pre-input", "-", p_str, s_str, g_str, "-", "-", "-", "-", "BASELINE")))

    first_failure: Optional[int] = None
    measurements: list[dict] = []

    try:
        for n in args.sizes:
            if context_length and (n + args.max_tokens) > context_length:
                pct_str = f"{int((n / context_length) * 100)}%"
                print(_c("darkgray",
                    header_fmt.format(n, args.max_tokens, "skip", "skip", "-", "-", pct_str, "-", "-", "SKIP")))
                continue

            for it in range(1, args.iterations + 1):
                mem_before = get_foundry_process_mb()
                sys_before = get_system_used_mb()
                gpu_before = get_gpu_used_mb()

                p = new_prompt(n, prompt=args.prompt, prompt_length=args.prompt_length)
                r = invoke_chat(model_obj, [{"role": "user", "content": p}], args.max_tokens)

                mem_after = get_foundry_process_mb()
                sys_after = get_system_used_mb()
                gpu_after = get_gpu_used_mb()
                mem_delta = (int(mem_after - mem_before)
                             if mem_before is not None and mem_after is not None else None)
                sys_delta = (int(sys_after - sys_before)
                             if sys_before is not None and sys_after is not None else None)
                gpu_delta = (int(gpu_after - gpu_before)
                             if gpu_before is not None and gpu_after is not None else None)

                kv_mb: Optional[int] = None
                attn_mb: Optional[int] = None
                if model_config:
                    kv_mb = get_kv_cache_mb(n, model_config)
                    attn_mb = get_attention_scores_mb(n, model_config)

                result = "OK" if r.ok else "FAIL"
                colour = "green" if r.ok else "red"
                pct_str = f"{int((n / context_length) * 100)}%" if context_length else "n/a"
                proc_str = (f"{mem_after} ({_sign(mem_delta)})"
                            if mem_after is not None and mem_delta is not None else "n/a")
                sys_str = (f"{sys_after} ({_sign(sys_delta)})"
                           if sys_after is not None and sys_delta is not None else "n/a")
                gpu_str = (f"{gpu_after} ({_sign(gpu_delta)})"
                           if gpu_after is not None and gpu_delta is not None else "n/a")
                attn_str = str(attn_mb) if attn_mb is not None else "-"
                time_str = round(r.ms / 1000, 1)
                ttft_str = str(round(r.ttfb_ms / 1000, 2)) if r.ttfb_ms is not None else "-"

                print(_c(colour,
                    header_fmt.format(n, args.max_tokens, proc_str, sys_str, gpu_str, attn_str,
                                     pct_str, ttft_str, time_str, result)))

                measurements.append({
                    "size": n,
                    "iter": it,
                    "mem_before": mem_before,
                    "mem_after": mem_after,
                    "sys_before": sys_before,
                    "sys_after": sys_after,
                    "gpu_before": gpu_before,
                    "gpu_after": gpu_after,
                    "ok": r.ok,
                    "time_ms": r.ms,
                })

                if not r.ok:
                    if first_failure is None:
                        first_failure = n
                        err_detail = r.detail if r.detail.strip() else "(empty error)"
                        print(_c("darkred", f"    Full error: {err_detail}"))
                        print(_c("darkgray", f"    First 100 chars of prompt: {p[:100]}"))
                    if not args.keep_going:
                        break

            if first_failure and not args.keep_going:
                break
    except KeyboardInterrupt:
        print(_c("yellow", "\nInterrupted by user (Ctrl+C)."))
    except Exception as exc:
        print(_c("darkred", f"ERROR during test loop: {exc}"))
        import traceback
        traceback.print_exc()
        return 1

    # -- 5) Summary ------------------------------------------------------------
    print()
    if first_failure:
        print(_c("red", f"First failure at ~{first_failure} input tokens."))
    else:
        print(_c("green", "No failure across the tested sizes on this machine. Try larger --sizes."))

    # -- 6) Leak analysis ------------------------------------------------------
    if args.leak_test or len([m for m in measurements if m["ok"] and m["mem_after"] is not None]) >= 5:
        run_leak_analysis(measurements, args.leak_threshold_mb)

    # -- 7) Final unload -------------------------------------------------------
    if model_obj is not None:
        print()
        write_step(f"Unloading '{api_model}' before exit...")
        proc_before_unload = get_foundry_process_mb()
        sys_before_unload = get_system_used_mb()
        try:
            model_obj.unload()
        except FoundryLocalException:
            pass
        time.sleep(1)
        proc_after_unload = get_foundry_process_mb()
        sys_after_unload = get_system_used_mb()

        if proc_before_unload is not None and proc_after_unload is not None:
            proc_freed = proc_before_unload - proc_after_unload
            sys_freed = ((sys_before_unload - sys_after_unload)
                         if sys_before_unload is not None and sys_after_unload is not None else None)
            sys_str = f" ({sys_freed} MB system)" if sys_freed is not None else ""
            colour = "red" if proc_before_unload > 1000 and proc_freed < 500 else "darkgray"
            print(_c(colour,
                f"    Process: {proc_before_unload} MB -> {proc_after_unload} MB  "
                f"(freed {proc_freed} MB{sys_str})"))
            if proc_before_unload > 1000 and proc_freed < 500:
                print(_c("red",
                    "    Unload freed <500 MB despite >1 GB loaded -- inter-run retention bug is active."))

    print()
    print(_c("darkgray", "Stop the service to free everything: foundry server stop"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
