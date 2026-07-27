# Prompt Length OOM Repro Scripts

This folder contains two scripts for prompt-length and memory behavior experiments.

## Scripts

- `foundry-oom-repro.ps1`
  - Uses Foundry Local (`foundry`) and calls the local inference endpoint.
  - Supports service/model orchestration and memory/leak workflows.

- `onnxruntime-genai-oom-repro.py`
  - Uses `onnxruntime_genai` directly (no Foundry service dependency).
  - Supports configurable prefill chunk size and sampled RSS/peak reporting.

## Help

PowerShell script help:

```powershell
pwsh ./foundry-oom-repro.ps1 -Help
```

Python script help:

```bash
python3 ./onnxruntime-genai-oom-repro.py --help
```

## Quick Start (PowerShell / Foundry Local)

Prerequisite: Foundry Local CLI (`foundry`) installed and available on `PATH`.

Run Foundry Local repro (example):

```powershell
pwsh ./foundry-oom-repro.ps1 -Sizes 1024,2048,4096,8192 -MaxTokens 16
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

For verbose ORT runtime logging, set `ORTGENAI_ORT_VERBOSE_LOGGING=1` when launching:

```bash
ORTGENAI_ORT_VERBOSE_LOGGING=1 python3 ./onnxruntime-genai-oom-repro.py \
  --model-path ~/.foundry/cache/models/Microsoft/qwen2.5-coder-7b-instruct-generic-gpu-4/v4 \
  --execution-provider webgpu
```