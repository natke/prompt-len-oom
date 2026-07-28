# Prompt Length OOM Repro Scripts

This folder contains two scripts for prompt-length and memory behavior experiments.

## Scripts

- `foundry-oom-repro.ps1`
  - Uses Foundry Local (`foundry`) and calls the local inference endpoint.
  - Supports service/model orchestration and memory/leak workflows.

- `foundry-sdk-oom-repro.py`
  - Uses the Foundry Local Python SDK (in-process inference, no separate daemon).
  - Tracks process RSS, system RAM, and GPU memory (NVIDIA dGPU / Intel iGPU).
  - Supports size sweep, multi-turn conversation, load/unload cycle test, and
    linear-regression leak analysis.

- `onnxruntime-genai-oom-repro.py`
  - Uses `onnxruntime_genai` directly (no Foundry service dependency).
  - Tracks process RSS, system RAM, and GPU memory (NVIDIA dGPU / Intel iGPU).
  - Supports configurable prefill chunk size and sampled RSS/peak reporting.

- `foundry-local-sdk-models-win.py`
  - Windows helper that uses the Foundry Local Python SDK.
  - Skips WebGPU EP registration and registers all other discoverable EPs.
  - Lists catalog models and can download models by alias or variant id.

## Help

PowerShell script help:

```powershell
pwsh ./foundry-oom-repro.ps1 -Help
```

Python script help:

```bash
python3 ./foundry-sdk-oom-repro.py --help
python3 ./onnxruntime-genai-oom-repro.py --help
```

Windows helper script help:

```bash
python3 ./foundry-local-sdk-models-win.py --help
```

## Quick Start (PowerShell / Foundry Local)

Prerequisite: Foundry Local CLI (`foundry`) installed and available on `PATH`.

Run Foundry Local repro (example):

```powershell
pwsh ./foundry-oom-repro.ps1 -Sizes 1024,2048,4096,8192 -MaxTokens 16
```

## Quick Start (Python / Foundry Local SDK)

Install Python dependencies:

```bash
python3 -m pip install foundry-local-sdk-winml psutil colorama
```

Run the SDK repro (auto-detects a model):

```bash
python3 ./foundry-sdk-oom-repro.py --sizes 1024 2048 4096 8192 --max-tokens 16
```

Run with a specific model:

```bash
python3 ./foundry-sdk-oom-repro.py --model qwen2.5-7b-instruct-trtrtx-gpu:2 --sizes 1024 2048 4096 8192
```

Run load/unload cycle test (10 cycles):

```bash
python3 ./foundry-sdk-oom-repro.py --model qwen2.5-7b-instruct-trtrtx-gpu:2 --load-cycle-test 10
```

Run leak detection (pins to one size, >= 20 iterations):

```bash
python3 ./foundry-sdk-oom-repro.py --model qwen2.5-7b-instruct-trtrtx-gpu:2 --sizes 4096 --leak-test
```

Run multi-turn conversation test:

```bash
python3 ./foundry-sdk-oom-repro.py --model qwen2.5-7b-instruct-trtrtx-gpu:2 --multi-turn 10
```

Include WebGPU EP registration (skipped by default):

```bash
python3 ./foundry-sdk-oom-repro.py --include-webgpu --sizes 1024 2048 4096
```

## Quick Start (Python / ORT GenAI)

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

If you plan to run with WebGPU, install the WebGPU EP package explicitly:

```bash
python3 -m pip install onnxruntime-ep-webgpu
```

Run ORT GenAI repro (example):

```bash
python3 ./onnxruntime-genai-oom-repro.py \
  --model-path ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-cpu-4/v4 \
  --execution-provider follow_config \
  --sizes 1024 2048 4096 8192 \
  --max-tokens 1 \
  --chunk-size 2048
```

Run ORT GenAI repro with explicit WebGPU EP:

```bash
python3 ./onnxruntime-genai-oom-repro.py \
  --model-path ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-gpu-4/v4 \
  --execution-provider webgpu \
  --sizes 1024 2048 4096 8192 \
  --max-tokens 1 \
  --chunk-size 2048
```

Register non-WebGPU IHV EPs before model load (default set excludes CUDA):

Note: IHV registration is WinML-catalog only (discover providers, `ensure_ready()`, then register by provider name and library path).

Use the package set that matches your target provider:

- Non-TRT RTX paths (for example CPU, OpenVINO, QNN):

```bash
python3 -m pip install onnxruntime-genai windowsml
```

- TRT RTX path:

```bash
python3 -m pip install onnxruntime-genai-cuda windowsml
```

You can also use the optional requirements files below.

```bash
python3 -m pip install -r requirements-genai.txt
# For TRT RTX instead:
python3 -m pip install -r requirements-genai-trtrtx.txt
```

Run with TensorRT RTX EP (register only the nvtensorrtrtx IHV EP):

```bash
python3 ./onnxruntime-genai-oom-repro.py \
  --model-path C:\Users\nakersha\.FoundryLocalModelDownloader\cache\models\Microsoft\qwen2.5-7b-instruct-trtrtx-gpu-2\v2 \
  --execution-provider follow_config \
  --sizes 1024 2048 4096 8192 \
  --max-tokens 1 \
  --chunk-size 2048 \
  --register-ihv-eps \
  --ihv-ep nvtensorrtrtx
```

Run with all default IHV EPs registered:

```bash
python3 ./onnxruntime-genai-oom-repro.py \
  --model-path ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-gpu-4/v4 \
  --execution-provider follow_config \
  --register-ihv-eps
```

If you want to include CUDA explicitly:

```bash
python3 ./onnxruntime-genai-oom-repro.py \
  --model-path ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-gpu-4/v4 \
  --execution-provider follow_config \
  --register-ihv-eps \
  --ihv-eps cuda openvino qnn nvtensorrtrtx
```

For verbose ORT runtime logging, set `ORTGENAI_ORT_VERBOSE_LOGGING=1` when launching:

```bash
ORTGENAI_ORT_VERBOSE_LOGGING=1 python3 ./onnxruntime-genai-oom-repro.py \
  --model-path ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-gpu-4/v4 \
  --execution-provider webgpu
```

## Quick Start (Windows / Foundry Local SDK model downloader)

Install dependencies in your Windows Python environment:

```bash
python3 -m pip install foundry-local-sdk-winml onnxruntime-genai-winml onnxruntime-winml
```

List models and register all non-WebGPU EPs:

```bash
python3 ./foundry-local-sdk-models-win.py
```

Download one or more models (alias or variant id):

```bash
python3 ./foundry-local-sdk-models-win.py --download qwen2.5-coder-7b-instruct-generic-cpu:4
```

Skip EP registration if needed:

```bash
python3 ./foundry-local-sdk-models-win.py --no-register-eps --download qwen2.5-coder-7b-instruct-generic-cpu:4
```

Run a model to summarize a long file (for example `constit.txt`):

```bash
python3 ./foundry-local-sdk-models-win.py \
  --run-model qwen2.5-coder-7b-instruct-generic-cpu:4 \
  --prompt-file ./constit.txt \
  --prompt-tokens 32000
```

Use inline text instead of a file:

```bash
python3 ./foundry-local-sdk-models-win.py \
  --run-model qwen2.5-coder-7b-instruct-generic-cpu:4 \
  --prompt-text "Paste your long text here" \
  --prompt-tokens 8000
```

## Attention Scratch Memory Formula

The dominant OOM driver at large input sizes is the materialized QK^T attention
scores matrix. This is an O(N²) allocation computed per layer (only one layer's
scores are live at a time):

```
Attention scratch (bytes) = num_attention_heads × N² × 4
```

Where:

- **num_attention_heads** — full attention head count (GQA does not reduce this;
  it reduces KV heads but the query projection still fans out to all heads)
- **N** — sequence length (input tokens)
- **4** — bytes per element (fp32, required for softmax numerical stability)

### Example: Qwen 2.5-7B (28 heads) at 8192 tokens

```
28 × 8192² × 4 = 7,516,192,768 bytes ≈ 7,168 MB
```

This matches the observed ~7.3 GB process memory jump at 8192 input tokens when
running with the NvTensorRTRTX EP (full-sequence prefill, no chunking).

### Comparison with KV Cache

The KV cache grows linearly and is GQA-aware:

```
KV cache (bytes) = N × num_kv_heads × head_dim × num_layers × 2 × dtype_bytes
```

For Qwen 2.5-7B (4 KV heads, head_dim=128, 28 layers, fp16) at 8192 tokens:

```
8192 × 4 × 128 × 28 × 2 × 2 = 471,859,200 bytes ≈ 450 MB
```

At large N, attention scratch dominates because it scales as N² while KV cache
scales as N.