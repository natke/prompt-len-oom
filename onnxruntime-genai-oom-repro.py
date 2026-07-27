#!/usr/bin/env python3

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import onnxruntime_genai as og

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_PREFILL_CHUNK_SIZE = 2048
DEFAULT_SAMPLE_INTERVAL_MS = 20

# IHV EP registration is intentionally WinML-catalog based.
# WebGPU remains separately registered only when explicitly requested.
DEFAULT_IHV_EPS = ["openvino", "qnn", "nvtensorrtrtx"]

IHV_EP_ALIASES = {
    "cuda": "cuda",
    "openvino": "openvino",
    "qnn": "qnn",
    "nvtensorrtrtx": "nvtensorrtrtx",
    "nvtensorrt_rtx": "nvtensorrtrtx",
    "nv_tensorrt_rtx": "nvtensorrtrtx",
}


@dataclass
class RunResult:
    ok: bool
    elapsed_ms: int
    prompt_tokens: int
    generated_tokens: int
    detail: str
    rss_before_mb: Optional[int] = None
    rss_after_mb: Optional[int] = None
    rss_peak_mb: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce prompt-length OOM behavior directly with onnxruntime-genai."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the ORT GenAI model directory containing genai_config.json.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Approximate input token counts to try.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to generate per request.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional base prompt text to prepend before filler.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=0,
        help="Approximate token budget reserved for synthetic base prompt text.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_PREFILL_CHUNK_SIZE,
        help="Prefill chunk size. Use 0 to disable chunked prefill.",
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
        help="Continue testing later sizes after a failure.",
    )
    parser.add_argument(
        "--execution-provider",
        default="cpu",
        help="Execution provider to use when overriding the config, for example cpu, cuda, dml, qnn, webgpu.",
    )
    parser.add_argument(
        "--register-ihv-eps",
        action="store_true",
        help="Register IHV EP libraries (non-WebGPU) before model load.",
    )
    parser.add_argument(
        "--ihv-eps",
        nargs="*",
        default=DEFAULT_IHV_EPS,
        help=(
            "IHV EPs to register when --register-ihv-eps is used. "
            "Default: openvino qnn nvtensorrtrtx"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra timing and setup information.",
    )
    return parser.parse_args()


def get_process_rss_mb() -> Optional[int]:
    if psutil is None:
        return None
    process = psutil.Process()
    return int(process.memory_info().rss / (1024 * 1024))


class PeakMemorySampler:
    def __init__(self, target_pid: int, interval_ms: int = DEFAULT_SAMPLE_INTERVAL_MS) -> None:
        self._target_pid = target_pid
        self._interval_ms = interval_ms
        self._process: Optional[subprocess.Popen[str]] = None
        self._stop_path: Optional[str] = None
        self._peak_path: Optional[str] = None

    def start(self) -> None:
        if psutil is None:
            return

        stop_fd, stop_path = tempfile.mkstemp(prefix="oga-rss-stop-")
        peak_fd, peak_path = tempfile.mkstemp(prefix="oga-rss-peak-")
        os.close(stop_fd)
        os.close(peak_fd)
        os.unlink(stop_path)

        self._stop_path = stop_path
        self._peak_path = peak_path

        sampler_code = "\n".join(
            [
                "import os",
                "import sys",
                "import time",
                "import psutil",
                "pid = int(sys.argv[1])",
                "stop_path = sys.argv[2]",
                "peak_path = sys.argv[3]",
                "interval = max(0.001, int(sys.argv[4]) / 1000.0)",
                "process = psutil.Process(pid)",
                "peak = 0",
                "while True:",
                "    if os.path.exists(stop_path):",
                "        break",
                "    try:",
                "        rss = process.memory_info().rss",
                "    except psutil.Error:",
                "        break",
                "    if rss > peak:",
                "        peak = rss",
                "        with open(peak_path, 'w', encoding='utf-8') as handle:",
                "            handle.write(str(peak))",
                "    time.sleep(interval)",
            ]
        )

        self._process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                sampler_code,
                str(self._target_pid),
                stop_path,
                peak_path,
                str(self._interval_ms),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop(self) -> Optional[int]:
        if psutil is None:
            return None

        peak_mb = None
        if self._stop_path:
            Path(self._stop_path).touch()
        if self._process is not None:
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        if self._peak_path and Path(self._peak_path).exists():
            peak_text = Path(self._peak_path).read_text(encoding="utf-8").strip()
            if peak_text:
                peak_mb = int(int(peak_text) / (1024 * 1024))
        self._cleanup()
        return peak_mb

    def _cleanup(self) -> None:
        for path_str in (self._stop_path, self._peak_path):
            if path_str and Path(path_str).exists():
                Path(path_str).unlink()
        self._stop_path = None
        self._peak_path = None
        self._process = None


def make_config(model_path: str, execution_provider: str, chunk_size: int) -> og.Config:
    config = og.Config(model_path)
    if execution_provider != "follow_config":
        config.clear_providers()
        if execution_provider != "cpu":
            config.append_provider(execution_provider)
    if chunk_size > 0:
        config.overlay(f'{{"search": {{"chunk_size": {chunk_size}}}}}')
    else:
        config.overlay('{"search": {"chunk_size": 0}}')
    return config


def register_webgpu_provider_if_requested(execution_provider: str) -> None:
    if execution_provider.lower() != "webgpu":
        return

    try:
        webgpu = importlib.import_module("onnxruntime_ep_webgpu")
    except ImportError as exc:
        raise RuntimeError(
            "Execution provider 'webgpu' requested but onnxruntime_ep_webgpu is not installed."
        ) from exc

    og.register_execution_provider_library(webgpu.get_ep_name(), webgpu.get_library_path())


def normalize_ep_name(ep_name: str) -> str:
    normalized = ep_name.strip().lower().replace("_", "").replace("-", "")
    if normalized.endswith("executionprovider"):
        normalized = normalized[: -len("executionprovider")]
    return normalized


def normalize_requested_ihv_eps(ihv_eps: list[str]) -> tuple[bool, list[str]]:
    raw = [ep.strip().lower() for ep in ihv_eps if ep.strip()]
    request_all = any(ep == "all" for ep in raw)
    if request_all:
        return True, []

    normalized: list[str] = []
    for ep in raw:
        mapped = IHV_EP_ALIASES.get(ep)
        if mapped is None:
            raise RuntimeError(
                f"Unknown IHV EP '{ep}'. Supported values: {', '.join(sorted(IHV_EP_ALIASES.keys()))}, all"
            )
        if mapped not in normalized:
            normalized.append(mapped)
    return False, normalized


def register_ihv_providers_with_winml_catalog(request_all: bool, requested_eps: list[str], verbose: bool) -> bool:
    try:
        from windowsml import EpCatalog
    except ImportError:
        raise RuntimeError(
            "--register-ihv-eps requires windowsml (install onnxruntime-genai-winml and onnxruntime-winml)."
        )

    requested_set = set(requested_eps)
    discovered: dict[str, tuple[str, str]] = {}
    registered: set[str] = set()

    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            provider_name = getattr(provider, "name", "")
            provider_library_path = getattr(provider, "library_path", "")
            if not provider_name:
                continue

            normalized_name = normalize_ep_name(provider_name)
            discovered[normalized_name] = (provider_name, provider_library_path)

            if not request_all and normalized_name not in requested_set:
                continue

            try:
                provider.ensure_ready()
                if provider_library_path == "":
                    continue
                og.register_execution_provider_library(provider_name, provider_library_path)
                registered.add(normalized_name)
                if verbose:
                    print(f"Registered IHV EP '{provider_name}' from WinML catalog")
            except Exception as exc:
                print(f"Failed to register execution provider {provider_name}: {exc}", file=sys.stderr)
                if verbose:
                    traceback.print_exc()

    if request_all:
        return True

    missing = sorted(set(requested_eps) - registered)
    if missing:
        available = sorted(discovered.keys())
        raise RuntimeError(
            f"Failed to register requested IHV EP(s): {', '.join(missing)}. "
            f"WinML catalog available EPs: {', '.join(available) if available else 'none'}"
        )

    return True


def register_ihv_providers_if_requested(register_ihv_eps: bool, ihv_eps: list[str], verbose: bool) -> None:
    if not register_ihv_eps:
        return

    request_all, requested = normalize_requested_ihv_eps(ihv_eps)
    register_ihv_providers_with_winml_catalog(request_all, requested, verbose)


def estimate_prompt_text(target_tokens: int, base_prompt: str, prompt_length: int) -> str:
    if prompt_length > 0:
        reserved = max(0, target_tokens - prompt_length)
        synthetic_prefix = "prefix " * prompt_length
        synthetic_tail = "data " * reserved
        return (synthetic_prefix + synthetic_tail).strip()
    if base_prompt.strip():
        return f"{base_prompt} " + ("data " * max(0, target_tokens)).strip()
    return ("data " * max(1, target_tokens)).strip()


def build_prompt_for_target(tokenizer: og.Tokenizer, target_tokens: int, base_prompt: str, prompt_length: int) -> tuple[str, int]:
    if target_tokens <= 0:
        return base_prompt, len(tokenizer.encode(base_prompt)) if base_prompt else 0

    prompt = estimate_prompt_text(target_tokens, base_prompt, prompt_length)
    actual_tokens = len(tokenizer.encode(prompt))
    if actual_tokens >= target_tokens:
        return prompt, actual_tokens

    filler_words = max(64, target_tokens - actual_tokens)
    while actual_tokens < target_tokens:
        prompt = f"{prompt} " + (("data ") * filler_words).strip()
        actual_tokens = len(tokenizer.encode(prompt))
        filler_words = max(32, target_tokens - actual_tokens)

    return prompt, actual_tokens


def run_once(
    model: og.Model,
    tokenizer: og.Tokenizer,
    prompt_text: str,
    target_tokens: int,
    max_tokens: int,
) -> RunResult:
    rss_before = get_process_rss_mb()
    sampler = PeakMemorySampler(os.getpid())
    start = time.perf_counter()
    try:
        input_tokens = tokenizer.encode(prompt_text)
        prompt_tokens = len(input_tokens)

        params = og.GeneratorParams(model)
        params.set_search_options(
            do_sample=False,
            max_length=prompt_tokens + max_tokens,
            batch_size=1,
            temperature=0.0,
            top_k=1,
        )

        generator = og.Generator(model, params)
        sampler.start()
        generator.append_tokens(input_tokens)

        generated_tokens = 0
        while not generator.is_done():
            generator.generate_next_token()
            generated_tokens += len(generator.get_next_tokens())

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        rss_after = get_process_rss_mb()
        rss_peak = sampler.stop()
        return RunResult(
            ok=True,
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            detail=f"OK target={target_tokens} actual={prompt_tokens}",
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
            rss_peak_mb=rss_peak,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        rss_after = get_process_rss_mb()
        rss_peak = sampler.stop()
        return RunResult(
            ok=False,
            elapsed_ms=elapsed_ms,
            prompt_tokens=target_tokens,
            generated_tokens=0,
            detail=f"{type(exc).__name__}: {exc}",
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
            rss_peak_mb=rss_peak,
        )


def format_rss(before_mb: Optional[int], after_mb: Optional[int], peak_mb: Optional[int]) -> str:
    if before_mb is None or after_mb is None:
        return "rss=n/a"
    delta_mb = after_mb - before_mb
    sign = "+" if delta_mb >= 0 else ""
    parts = [f"rss={before_mb}->{after_mb} MB ({sign}{delta_mb})"]
    if peak_mb is not None:
        peak_delta_mb = peak_mb - before_mb
        peak_sign = "+" if peak_delta_mb >= 0 else ""
        parts.append(f"peak={peak_mb} MB ({peak_sign}{peak_delta_mb})")
    return ", ".join(parts)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model path does not exist: {model_path}", file=sys.stderr)
        return 1
    if args.chunk_size < 0:
        print("ERROR: --chunk-size must be >= 0", file=sys.stderr)
        return 1

    try:
        register_ihv_providers_if_requested(args.register_ihv_eps, args.ihv_eps, args.verbose)
        register_webgpu_provider_if_requested(args.execution_provider)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loading model from: {model_path}")
        print(f"Execution provider: {args.execution_provider}")
        print(f"Prefill chunk_size: {args.chunk_size}")
        if psutil is None:
            print("Memory probe: disabled (install psutil to capture Python RSS)")

    config = make_config(str(model_path), args.execution_provider, args.chunk_size)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)

    print(f"Model path: {model_path}")
    print(f"Execution provider: {args.execution_provider}")
    print(f"Prefill chunk_size: {args.chunk_size}")
    if psutil is None:
        print("Memory probe: unavailable (install psutil for Python-process RSS only)")
    else:
        print(f"Memory probe: Python RSS before/after plus peak sampled every {DEFAULT_SAMPLE_INTERVAL_MS} ms")

    header = f"{'size':>8}  {'iter':>4}  {'prompt':>8}  {'gen':>5}  {'ms':>7}  result"
    print(header)
    print("-" * len(header))

    had_failure = False
    for size in args.sizes:
        prompt_text, actual_tokens = build_prompt_for_target(
            tokenizer,
            target_tokens=size,
            base_prompt=args.prompt,
            prompt_length=args.prompt_length,
        )
        if args.verbose:
            print(f"Prepared size {size} with actual prompt tokens {actual_tokens}")

        for iteration in range(1, args.iterations + 1):
            result = run_once(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                target_tokens=size,
                max_tokens=args.max_tokens,
            )

            status = "OK" if result.ok else "FAIL"
            rss_text = format_rss(result.rss_before_mb, result.rss_after_mb, result.rss_peak_mb)
            print(
                f"{size:8d}  {iteration:4d}  {result.prompt_tokens:8d}  {result.generated_tokens:5d}  "
                f"{result.elapsed_ms:7d}  {status} {rss_text} {result.detail}"
            )

            if not result.ok:
                had_failure = True
                if not args.keep_going:
                    return 1
                break

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())