<#
.SYNOPSIS
    Profiles Foundry Local memory behavior: reproduces input-size OOM failures
    and detects memory leaks / allocator retention across repeated inferences.

.DESCRIPTION
    Only prerequisite: the Foundry Local CLI ('foundry') is installed and on PATH.

    The script:
      1. Ensures the local inference service is running and discovers its endpoint.
      2. Auto-detects available execution providers (CUDA GPU, OpenVINO, WebGPU,
         QNN NPU, CPU) and selects a small chat model if none is specified.
      3. Loads the model and records a memory baseline / post-load footprint.
      4. Sends chat requests with configurable input sizes and iteration counts,
         recording process memory before and after each call.
      5. On completion, runs a linear-regression leak analysis over the collected
         samples (MB per iteration), and distinguishes real leaks from BFCArena
         allocator retention by comparing first-half vs. second-half averages.

.PARAMETER Model
    Foundry model alias to test. Default "" = auto-detect a small model.

.PARAMETER Sizes
    Approximate input token counts to try, ascending. With -LeakTest, only the
    first entry is used.

.PARAMETER MaxTokens
    Output tokens to request. Kept small so the failure is driven by INPUT size.

.PARAMETER Prompt
    Custom base prompt text to prepend before filler. Default "" = pure filler.

.PARAMETER PromptLength
    Base prompt token length. Default 0 = no base prompt, all tokens are input.

.PARAMETER Iterations
    Number of times to repeat each size. Increase for leak detection.

.PARAMETER KeepGoing
    Keep testing every size instead of stopping at the first failure.

.PARAMETER LeakTest
    Enables leak-detection mode: pins to a single size, forces >= 20 iterations,
    and prints a regression-based verdict at the end.

.PARAMETER LeakThresholdMb
    MB/iteration slope below which drift is treated as noise. Default 2.
    Slopes between 1x and 10x this value are flagged SUSPICIOUS; above 10x are
    reported as PROBABLE LEAK.

.PARAMETER Help
    Prints command help and exits.

.EXAMPLE
    ./foundry-oom-repro.ps1

.EXAMPLE
    ./foundry-oom-repro.ps1 -Model qwen2.5-0.5b-instruct-cuda-gpu:4 -Sizes 512,2048,8192

.EXAMPLE
    ./foundry-oom-repro.ps1 -LeakTest -Sizes 2000 -Iterations 30

.EXAMPLE
    ./foundry-oom-repro.ps1 -LoadCycleTest 5 -Model qwen3-4b-generic-gpu

.EXAMPLE
    ./foundry-oom-repro.ps1 -MultiTurn 20 -Sizes 512 -MaxTokens 64
    # 20-turn chat, each user message ~512 filler tokens, assistant capped at 64.
#>
[CmdletBinding()]
param(
    [string]$Model = "",
    [int[]]$Sizes = @(128, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    [int]$MaxTokens = 16,
    [string]$Prompt = "",
    [int]$PromptLength = 0,
    [int]$Iterations = 1,
    [switch]$KeepGoing,
    [switch]$LeakTest,
    [int]$LeakThresholdMb = 2,
    [int]$LoadCycleTest = 0,
    [int]$MultiTurn = 0,
    [switch]$RestartService,
    [switch]$Help
)

if ($Help) {
    Get-Help -Full $PSCommandPath
    exit 0
}

# -LeakTest defaults: fixed size, many iterations, single size only
if ($LeakTest) {
    if ($Sizes.Count -gt 1) {
        Write-Host "    [LeakTest] Multiple -Sizes given; using only the first ($($Sizes[0])) for leak detection." -ForegroundColor DarkYellow
        $Sizes = @($Sizes[0])
    }
    if ($Iterations -lt 10) {
        Write-Host "    [LeakTest] Bumping -Iterations from $Iterations to 20 (minimum for regression)." -ForegroundColor DarkYellow
        $Iterations = 20
    }
}

$ErrorActionPreference = "Stop"

# Allow Ctrl+C to gracefully exit
trap {
    if ($_ -is [System.Management.Automation.PipelineStoppedException] -or 
        $_.Exception -is [System.OperationCanceledException] -or
        $_.FullyQualifiedErrorId -eq "PipelineStopped") {
        Write-Host "`nInterrupted by user (Ctrl+C). Cleaning up..." -ForegroundColor Yellow
        exit 0
    }
    throw $_
}

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# Helper functions - defined here so they're available throughout the script
function Get-VramUsedMb {
    # Track the Foundry process memory (works for both CPU and GPU models)
    try {
        # Try common process names
        $processNames = @("foundrylocald", "foundry", "foundry.exe", "python", "onnxruntime")
        foreach ($name in $processNames) {
            $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
            if ($procs) {
                # If multiple processes, get the one with largest memory
                $proc = $procs | Sort-Object -Property PrivateMemorySize64 -Descending | Select-Object -First 1
                if ($proc.PrivateMemorySize64 -gt 100MB) {  # Only if using meaningful memory
                    $mb = [int]($proc.PrivateMemorySize64 / 1MB)
                    return $mb
                }
            }
        }
    } catch { }
    
    return $null
}

function Get-SystemMemoryMb {
    # System-wide physical memory. Returns @{UsedMb, AvailableMb, TotalMb, CommittedMb}.
    # System-used memory is a more reliable OOM signal than one process's private
    # bytes: BFCArena's failure to allocate depends on RAM+pagefile availability,
    # not on any single process's working set.
    try {
        $os      = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        $totalKb = [int64]$os.TotalVisibleMemorySize
        $freeKb  = [int64]$os.FreePhysicalMemory
        $usedKb  = $totalKb - $freeKb
        $result = @{
            UsedMb      = [int]($usedKb / 1024)
            AvailableMb = [int]($freeKb / 1024)
            TotalMb     = [int]($totalKb / 1024)
            CommittedMb = $null
        }
        try {
            $pf = Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue
            $committedKb = $usedKb + (($pf | Measure-Object -Property CurrentUsage -Sum).Sum * 1024)
            $result.CommittedMb = [int]($committedKb / 1024)
        } catch { }
        return $result
    } catch {
        return $null
    }
}

function Get-SystemUsedMb {
    $s = Get-SystemMemoryMb
    if ($s) { return $s.UsedMb } else { return $null }
}

function New-Prompt([int]$tokens) {
    if ($PromptLength -gt 0) {
        $fillerTokens = [Math]::Max(0, $tokens - $PromptLength)
        $baseFiller = "data " * $PromptLength
        $extraFiller = "data " * $fillerTokens
        return ($baseFiller + $extraFiller).Trim()
    } elseif ([string]::IsNullOrWhiteSpace($Prompt)) {
        return (("data " * $tokens)).Trim()
    } else {
        $baseTokens = $Prompt.Split() | Measure-Object | Select-Object -ExpandProperty Count
        $fillerTokens = [Math]::Max(0, $tokens - $baseTokens)
        $filler = (("data " * $fillerTokens)).Trim()
        return if ($fillerTokens -gt 0) { "$Prompt $filler" } else { $Prompt }
    }
}

function Invoke-Chat($prompt) {
    # Streaming request. $prompt can be either:
    #   * a plain string  -> wrapped as a single user message (single-turn), or
    #   * an array of {role, content} hashtables -> sent verbatim (multi-turn).
    # The server responds with text/event-stream chunks of the shape:
    #   data: {"choices":[{"delta":{"content":"..."},"finish_reason":null}]}
    # terminated by a  data: [DONE]  line.
    # Build message list. NOTE: assignment from an if-expression in PS 5.1
    # unwraps single-element arrays, which would make ConvertTo-Json emit
    # "messages":{...} instead of "messages":[{...}]. The [object[]] cast on
    # the hashtable value below forces array typing regardless of what
    # $messages ended up as (bare hashtable, 1-elem array, or N-elem array).
    $messages = if ($prompt -is [string]) {
        @(@{ role = "user"; content = $prompt })
    } else {
        @($prompt)
    }
    $body = @{
        model       = $apiModel
        messages    = [object[]]$messages
        max_tokens  = $MaxTokens
        temperature = 0
        stream      = $true
    } | ConvertTo-Json -Depth 6 -Compress

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        $webReq = [System.Net.WebRequest]::Create($chatUrl)
        $webReq.Method            = "POST"
        $webReq.ContentType       = "application/json"
        $webReq.Accept            = "text/event-stream"
        $webReq.Timeout           = 1200000
        $webReq.ReadWriteTimeout  = 1200000

        $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
        $webReq.ContentLength = $bytes.Length

        $reqStream = $webReq.GetRequestStream()
        $reqStream.Write($bytes, 0, $bytes.Length)
        $reqStream.Close()

        $webResp    = $webReq.GetResponse()
        $respStream = $webResp.GetResponseStream()
        $reader     = [System.IO.StreamReader]::new($respStream)

        $chunkCount       = 0   # SSE chunks that carried non-empty delta.content
        $nonContentChunks = 0   # SSE chunks that parsed but had no delta.content (role-only, empty delta, finish-only)
        $contentChars     = 0
        $assistantContent = ""  # accumulated assistant text (for multi-turn history)
        $totalBytes       = 0   # cumulative bytes read across all SSE lines (rough stream volume)
        $sawDone          = $false
        $finishReason     = $null
        $ttfbMs           = $null    # ms until first *content* chunk
        $firstByteMs      = $null    # ms until first SSE data: line of any kind

        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ($null -ne $line) { $totalBytes += $line.Length }
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            if (-not $line.StartsWith("data:")) { continue }

            if ($null -eq $firstByteMs) { $firstByteMs = $sw.ElapsedMilliseconds }

            $payload = $line.Substring(5).Trim()
            if ($payload -eq "[DONE]") { $sawDone = $true; break }

            try {
                $obj   = $payload | ConvertFrom-Json
                $delta = $obj.choices[0].delta
                if ($delta -and $delta.content) {
                    if ($chunkCount -eq 0) { $ttfbMs = $sw.ElapsedMilliseconds }
                    $deltaText = [string]$delta.content
                    $contentChars     += $deltaText.Length
                    $assistantContent += $deltaText
                    $chunkCount++
                } else {
                    $nonContentChunks++
                }
                # finish_reason may be null on early chunks, non-null on the last.
                # Capture whichever we see last (including explicit null → keep prior).
                $fr = $obj.choices[0].finish_reason
                if ($fr) { $finishReason = $fr }
            } catch {
                # Ignore a single malformed chunk; keep streaming the rest.
            }
        }

        $reader.Close()
        $respStream.Close()
        $webResp.Close()

        $sw.Stop()
        # streamClosedEarly: true if the server dropped the connection without
        # sending [DONE]. Strong signal that the daemon crashed or aborted mid-run.
        $streamClosedEarly = (-not $sawDone)

        if ($chunkCount -gt 0) {
            $parts = @("stream: $chunkCount chunks", "$contentChars chars")
            if ($nonContentChunks -gt 0) { $parts += "$nonContentChunks non-content" }
            if ($null -ne $firstByteMs)  { $parts += "first-byte=${firstByteMs}ms" }
            if ($null -ne $ttfbMs)       { $parts += "ttfb=${ttfbMs}ms" }
            if ($finishReason)           { $parts += "finish=$finishReason" }
            else                         { $parts += "finish=<null>" }
            $parts += "bytes=$totalBytes"
            if ($streamClosedEarly)      { $parts += "CLOSED-EARLY" }
            $result = [PSCustomObject]@{
                ok      = $true
                ms      = $sw.ElapsedMilliseconds
                ttfbMs  = $ttfbMs
                content = $assistantContent
                detail  = "OK ({0})" -f ($parts -join ", ")
            }
        } else {
            $failParts = @(
                "no content chunks",
                "non-content-chunks=$nonContentChunks",
                "bytes=$totalBytes",
                "DONE=$sawDone"
            )
            if ($streamClosedEarly)     { $failParts += "CLOSED-EARLY (no [DONE] from server)" }
            if ($null -ne $firstByteMs) { $failParts += "first-byte=${firstByteMs}ms" }
            else                        { $failParts += "no SSE data lines received" }
            $failParts += "finish=$(if ($finishReason) { $finishReason } else { '<null>' })"
            $result = [PSCustomObject]@{
                ok      = $false
                ms      = $sw.ElapsedMilliseconds
                ttfbMs  = $ttfbMs
                content = $assistantContent
                detail  = "Stream returned no content chunks ({0})" -f ($failParts -join ", ")
            }
        }
    } catch {
        $sw.Stop()
        # Extract as much as possible from the failure:
        #   - Exception type + message
        #   - HTTP status code (if WebException)
        #   - Response body (server-side error message from Foundry Local / ORT)
        $ex        = $_.Exception
        $exType    = $ex.GetType().FullName
        $exMessage = $ex.Message
        $status    = $null
        $respBody  = $null

        # Walk to the underlying WebException (Invoke-WebRequest wraps it).
        $webEx = $ex
        while ($webEx -and -not ($webEx -is [System.Net.WebException])) {
            $webEx = $webEx.InnerException
        }
        if ($webEx -and $webEx.Response) {
            try {
                $httpResp = [System.Net.HttpWebResponse]$webEx.Response
                $status   = "$([int]$httpResp.StatusCode) $($httpResp.StatusCode)"
                $errStream = $httpResp.GetResponseStream()
                if ($errStream) {
                    $errReader = [System.IO.StreamReader]::new($errStream)
                    $respBody  = $errReader.ReadToEnd()
                    $errReader.Close()
                }
            } catch {
                # If we can't read the response body, keep going with what we have.
            }
        }

        $bodySnippet = if ($respBody) {
            $respBody.Substring(0, [Math]::Min(800, $respBody.Length))
        } else { "(no response body)" }

        $detail = @(
            "Exception: $exType",
            "Message  : $exMessage",
            "Status   : $(if ($status) { $status } else { 'no HTTP response (network / timeout)' })",
            "Body     : $bodySnippet"
        ) -join "`n              "

        $result = [PSCustomObject]@{ ok = $false; ms = $sw.ElapsedMilliseconds; ttfbMs = $null; content = ""; detail = $detail }
    }

    return $result
}


function Get-ModelFileSizeMb {
    param([string]$modelName)
    
    # Common model cache locations (Windows and Unix-style)
    $cachePaths = @(
        "$env:USERPROFILE\.cache\foundry\models",
        "$env:USERPROFILE\.cache\huggingface\hub",
        "$env:USERPROFILE\.foundry\models",
        "$env:USERPROFILE\AppData\Local\foundry\models",
        "$env:USERPROFILE\AppData\Local\huggingface\hub",
        "$env:LOCALAPPDATA\foundry\models",
        "$env:LOCALAPPDATA\huggingface\hub",
        "~/.cache/foundry/models",
        "~/.cache/huggingface/hub",
        "~/.foundry/models"
    )
    
    # Try to find the model in cache
    foreach ($cachePath in $cachePaths) {
        $expandedPath = [System.Environment]::ExpandEnvironmentVariables($cachePath)
        if (Test-Path $expandedPath -ErrorAction SilentlyContinue) {
            try {
                # Find directories matching the model name or containing model files
                $modelDirs = @(Get-ChildItem $expandedPath -Directory -ErrorAction SilentlyContinue | 
                    Where-Object { $_.Name -like "*$modelName*" -or $_.Name -match ($modelName -replace '-', '|') })
                
                foreach ($modelDir in $modelDirs) {
                    $totalSize = (Get-ChildItem $modelDir -Recurse -ErrorAction SilentlyContinue | 
                        Measure-Object -Property Length -Sum | Select-Object -ExpandProperty Sum)
                    if ($totalSize -gt 0) {
                        return [math]::Round($totalSize / 1MB, 0)
                    }
                }
            } catch {
                # Skip this path if there's an access error
                continue
            }
        }
    }
    
    # Fallback: try to query foundry directly for the model size hint from recent load
    # This would require running foundry commands which we do during load, 
    # so we could extract the size from that output instead
    return $null
}

function Get-ModelConfig {
    param([string]$modelName)

    # ONNX Runtime GenAI models (what Foundry Local ships) use `genai_config.json`,
    # NOT the HuggingFace-style `config.json`. Schema is different:
    #   {
    #     "model": {
    #       "context_length": 32768,
    #       "decoder": {
    #         "num_attention_heads": 28, "num_key_value_heads": 4,
    #         "num_hidden_layers": 28,   "hidden_size": 3584,
    #         "head_size": 128, ...
    #       }
    #     }
    #   }
    # HF-style config.json (used by pure HF or OpenVINO-IR exports) has the fields
    # at the top level with `head_dim`, `max_position_embeddings`, `torch_dtype`.
    # We look for either file and parse whichever schema is present.

    # 1) Preferred: ask the Foundry CLI for its actual cache root.
    $cachePaths = New-Object System.Collections.Generic.List[string]
    try {
        $cliOut = & foundry cache location 2>$null | Out-String
        # Match a Windows path in the CLI output
        if ($cliOut -match '([A-Za-z]:\\[^\r\n"]+)') {
            $cachePaths.Add($matches[1].TrimEnd('\', ' ', "`t")) | Out-Null
        }
    } catch { }

    # 2) Hardcoded fallbacks for older/broken installs and for HF-style caches
    #    (OpenVINO exports sometimes live alongside a HuggingFace snapshot dir).
    foreach ($p in @(
        "$env:LOCALAPPDATA\Microsoft\FoundryCache",
        "$env:LOCALAPPDATA\FoundryCache",
        "$env:USERPROFILE\.cache\foundry\models",
        "$env:USERPROFILE\.foundry\models",
        "$env:LOCALAPPDATA\foundry\models",
        "$env:USERPROFILE\.cache\huggingface\hub",
        "$env:LOCALAPPDATA\huggingface\hub"
    )) {
        if ($p) { $cachePaths.Add($p) | Out-Null }
    }
    $cachePaths = $cachePaths | Select-Object -Unique

    # Split the model name into tokens for path scoring. Anything >=3 chars counts.
    $tokens = @($modelName -split '[-_.:]+' | Where-Object { $_.Length -ge 3 })

    foreach ($cachePath in $cachePaths) {
        $expandedPath = [System.Environment]::ExpandEnvironmentVariables($cachePath)
        if (-not (Test-Path $expandedPath -ErrorAction SilentlyContinue)) { continue }

        try {
            # Look for either schema, recursively.
            $configFiles = @(Get-ChildItem $expandedPath -Include 'genai_config.json','config.json' `
                             -File -Recurse -ErrorAction SilentlyContinue)
            if ($configFiles.Count -eq 0) { continue }

            # Score by how many model-name tokens the file's path contains.
            # Prefer genai_config.json (matches ORT GenAI runtime exactly) over
            # a stray HF config.json when both exist for the same model.
            $best      = $null
            $bestScore = -1
            foreach ($cf in $configFiles) {
                $lower = $cf.FullName.ToLowerInvariant()
                $score = 0
                foreach ($t in $tokens) {
                    if ($lower.Contains($t.ToLowerInvariant())) { $score++ }
                }
                if ($cf.Name -eq 'genai_config.json') { $score += 1 }  # tiebreaker
                if ($score -gt $bestScore) { $bestScore = $score; $best = $cf }
            }
            # Require at least one token match to avoid picking a random model's config.
            if (-not $best -or $bestScore -le 0) { continue }

            $json = Get-Content $best.FullName -Raw | ConvertFrom-Json

            if ($best.Name -eq 'genai_config.json') {
                # --- ORT GenAI schema ------------------------------------------------
                $dec = $json.model.decoder
                if (-not $dec) { continue }
                $numAttn = $dec.num_attention_heads
                $numKv   = if ($dec.PSObject.Properties['num_key_value_heads']) { $dec.num_key_value_heads } else { $numAttn }
                $headDim = if ($dec.PSObject.Properties['head_size'] -and $dec.head_size) {
                                $dec.head_size
                           } elseif ($dec.PSObject.Properties['head_dim'] -and $dec.head_dim) {
                                $dec.head_dim
                           } else {
                                [int]($dec.hidden_size / $numAttn)
                           }
                $maxPos  = if ($json.model.PSObject.Properties['context_length']) {
                                $json.model.context_length
                           } elseif ($dec.PSObject.Properties['max_position_embeddings']) {
                                $dec.max_position_embeddings
                           } else { $null }
                # genai_config doesn't record weight dtype directly. KV cache uses
                # fp16 by default in ORT GenAI's GPU/DML/WebGPU paths. Runtime can
                # override via session options; fp16 is the right default here.
                $dtype = 'float16'
                return @{
                    num_attention_heads     = $numAttn
                    num_key_value_heads     = $numKv
                    num_hidden_layers       = $dec.num_hidden_layers
                    hidden_size             = $dec.hidden_size
                    head_dim                = $headDim
                    dtype                   = $dtype
                    max_position_embeddings = $maxPos
                    source_file             = $best.FullName
                    source_schema           = 'genai_config'
                }
            } else {
                # --- HuggingFace-style config.json schema ---------------------------
                $config  = $json
                $numAttn = $config.num_attention_heads
                $numKv   = if ($config.PSObject.Properties['num_key_value_heads']) { $config.num_key_value_heads } else { $numAttn }
                $headDim = if ($config.PSObject.Properties['head_dim'] -and $config.head_dim) {
                                $config.head_dim
                           } else {
                                [int]($config.hidden_size / $numAttn)
                           }
                $dtype   = if ($config.PSObject.Properties['torch_dtype']) { $config.torch_dtype }
                           elseif ($config.PSObject.Properties['dtype']) { $config.dtype }
                           else { "float16" }
                $maxPos  = if ($config.PSObject.Properties['max_position_embeddings']) {
                                $config.max_position_embeddings
                           } else { $null }
                return @{
                    num_attention_heads     = $numAttn
                    num_key_value_heads     = $numKv
                    num_hidden_layers       = $config.num_hidden_layers
                    hidden_size             = $config.hidden_size
                    head_dim                = $headDim
                    dtype                   = $dtype
                    max_position_embeddings = $maxPos
                    source_file             = $best.FullName
                    source_schema           = 'config'
                }
            }
        } catch {
            continue
        }
    }

    return $null
}

function Get-DtypeBytes([string]$dtype) {
    switch -Regex ($dtype) {
        'float32|fp32'  { return 4 }
        'bfloat16|bf16' { return 2 }
        'float16|fp16'  { return 2 }
        'int8|uint8'    { return 1 }
        default         { return 2 }
    }
}

function Get-KvCacheMb {
    # KV cache size for a given number of tokens.
    # Formula (GQA-aware):
    #   tokens x num_KV_heads x head_dim x num_layers x 2 (K+V) x dtype_bytes
    # For qwen3-4b (GQA 32:8), using num_attention_heads would overestimate 4x.
    param([int]$tokens, $config)
    if (-not $config) { return $null }
    $bytesPerElem = Get-DtypeBytes $config.dtype
    $bytes = [int64]$tokens * $config.num_key_value_heads * $config.head_dim * $config.num_hidden_layers * 2 * $bytesPerElem
    return [int]($bytes / 1MB)
}

function Get-AttentionScoresMb {
    # Materialized QK^T attention scores matrix per layer.
    # Shape [batch=1, num_attention_heads, seq, seq], stored in fp32 for
    # softmax numerical stability (this is the O(N^2) term that drives OOM).
    # Only one layer's scores are alive at a time, so we do NOT multiply by num_layers.
    param([int]$tokens, $config)
    if (-not $config) { return $null }
    $bytes = [int64]$config.num_attention_heads * $tokens * $tokens * 4
    return [int]($bytes / 1MB)
}

# ----------------------------------------------------------------------------
# Load / unload cycle test
# ----------------------------------------------------------------------------
# Isolates the "process grows across script runs" bug by repeating the
# unload -> load cycle N times in-process. For each cycle it reports:
#   * post-unload:  how much memory unload actually freed
#   * post-load:    how much memory the next load added
#   * net drift:    (post-load[n]) - (post-load[n-1])  -- should be ~0 if healthy
#
# A healthy service:  load adds X, unload frees ~X, drift per cycle ~= 0.
# A leaky service  :  load adds X, unload frees ~0, drift per cycle ~= X.
function Invoke-LoadCycleTest {
    param(
        [string]$ModelAlias,   # alias passed to `foundry model load`
        [string]$ApiId,        # id returned by /v1/models, used by `foundry model unload`
        [int]$Cycles,
        [int]$InitialProc,     # baseline process MB (before initial load)
        [int]$InitialSys,      # baseline system  MB
        [int]$AfterLoadProc,   # process MB after the initial load (== cycle 1 post-load)
        [int]$AfterLoadSys,    # system  MB after the initial load
        $ModelConfig = $null   # optional hashtable from Get-ModelConfig for theoretical KV/Attn
    )

    Write-Host ""
    Write-Step "Load/unload cycle test: $Cycles cycles (isolates the inter-run memory retention bug)"
    Write-Host "    Model : $ModelAlias   (unload id: $ApiId)"
    Write-Host "    Each cycle: unload -> sample -> load -> sample."

    # Per-request headroom is invariant across cycles (depends only on model + ctx),
    # so print it once here as a reference rather than per-row.
    if ($ModelConfig -and $ModelConfig.max_position_embeddings) {
        $kvPre  = Get-KvCacheMb         $ModelConfig.max_position_embeddings $ModelConfig
        $attnPk = Get-AttentionScoresMb $ModelConfig.max_position_embeddings $ModelConfig
        Write-Host ("    Per-request headroom @ ctx={0}: KV pre-alloc {1} MB (in load totals) + Attn scratch ~{2} MB (O(N^2), per request)." -f `
            $ModelConfig.max_position_embeddings, $kvPre, $attnPk) -ForegroundColor DarkGray
    }
    Write-Host ""

    $fmt  = "{0,-6}{1,-13}{2,-10}{3,-9}{4,-10}{5,-9}{6}"
    $sfmt = "{0:+#;-#;+0}"
    Write-Host ($fmt -f "Cycle","Action","Proc(MB)","[d]","Sys(MB)","[d]","Notes") -ForegroundColor Yellow
    Write-Host ("-" * 92)

    # Print cycle 1 (the initial load that already happened) for continuity.
    Write-Host ($fmt -f 1,"pre-load",$InitialProc,"-",$InitialSys,"-","from baseline") -ForegroundColor DarkCyan
    Write-Host ($fmt -f 1,"post-load",$AfterLoadProc,($sfmt -f ($AfterLoadProc - $InitialProc)), `
        $AfterLoadSys,($sfmt -f ($AfterLoadSys - $InitialSys)),"initial load") -ForegroundColor Cyan

    $prevPostLoadProc = $AfterLoadProc
    $prevPostLoadSys  = $AfterLoadSys

    $loadDeltas   = @()
    $unloadFreed  = @()
    $cycleDrift   = @()

    for ($c = 2; $c -le $Cycles; $c++) {
        # --- Unload ---
        & foundry model unload $ApiId 2>&1 | Out-Null
        Start-Sleep -Seconds 1

        $procU = Get-VramUsedMb
        $sysU  = Get-SystemUsedMb
        if ($null -eq $procU -or $null -eq $sysU) {
            Write-Host "    Cycle ${c}: could not sample memory after unload, skipping." -ForegroundColor Yellow
            continue
        }

        $uDeltaProc = $procU - $prevPostLoadProc   # negative if unload actually freed anything
        $uDeltaSys  = $sysU  - $prevPostLoadSys
        $freed      = [Math]::Max(0, -$uDeltaProc)
        $unloadFreed += $freed

        $colorU = if ($freed -lt 500) { "Red" } else { "DarkCyan" }
        Write-Host ($fmt -f $c,"post-unload",$procU,($sfmt -f $uDeltaProc), `
            $sysU,($sfmt -f $uDeltaSys),"unload freed $freed MB") -ForegroundColor $colorU

        # --- Load ---
        & foundry model load $ModelAlias 2>&1 | Out-Null
        Start-Sleep -Seconds 1

        $procL = Get-VramUsedMb
        $sysL  = Get-SystemUsedMb
        if ($null -eq $procL -or $null -eq $sysL) {
            Write-Host "    Cycle ${c}: could not sample memory after load, skipping." -ForegroundColor Yellow
            continue
        }

        $lDeltaProc = $procL - $procU
        $lDeltaSys  = $sysL  - $sysU
        $loadDeltas += $lDeltaProc

        Write-Host ($fmt -f $c,"post-load",$procL,($sfmt -f $lDeltaProc), `
            $sysL,($sfmt -f $lDeltaSys),"") -ForegroundColor Cyan

        # Net drift for this cycle = post-load[c] - post-load[c-1]
        $drift = $procL - $prevPostLoadProc
        $cycleDrift += $drift

        $prevPostLoadProc = $procL
        $prevPostLoadSys  = $sysL
    }

    if ($cycleDrift.Count -eq 0) {
        Write-Host ""
        Write-Host "No cycles completed." -ForegroundColor Yellow
        return
    }

    $meanLoad   = [int](($loadDeltas  | Measure-Object -Average).Average)
    $meanFreed  = [int](($unloadFreed | Measure-Object -Average).Average)
    $meanDrift  = [int](($cycleDrift  | Measure-Object -Average).Average)
    $minLoad    = ($loadDeltas  | Measure-Object -Minimum).Minimum
    $maxLoad    = ($loadDeltas  | Measure-Object -Maximum).Maximum
    $minFreed   = ($unloadFreed | Measure-Object -Minimum).Minimum
    $maxFreed   = ($unloadFreed | Measure-Object -Maximum).Maximum
    $minDrift   = ($cycleDrift  | Measure-Object -Minimum).Minimum
    $maxDrift   = ($cycleDrift  | Measure-Object -Maximum).Maximum

    Write-Host ""
    Write-Host "Cycle summary (n=$($cycleDrift.Count) cycles measured):" -ForegroundColor Yellow
    Write-Host ("  Load added      : mean {0,6} MB   range [{1}, {2}]" -f $meanLoad,  $minLoad,  $maxLoad)
    Write-Host ("  Unload freed    : mean {0,6} MB   range [{1}, {2}]" -f $meanFreed, $minFreed, $maxFreed)
    Write-Host ("  Net drift/cycle : mean {0,6} MB   range [{1}, {2}]" -f $meanDrift, $minDrift, $maxDrift)
    Write-Host ""

    # Verdict + extrapolation to OOM
    if ($meanDrift -ge 500) {
        Write-Host "VERDICT: LEAK in the unload path." -ForegroundColor Red
        Write-Host ("  Load costs ~{0} MB but unload only frees ~{1} MB." -f $meanLoad, $meanFreed) -ForegroundColor Red
        Write-Host ("  Every unload+reload retains ~{0} MB in foundrylocald (CPU-side)." -f $meanDrift) -ForegroundColor Red
        $sysNow = Get-SystemMemoryMb
        if ($sysNow -and $meanDrift -gt 0) {
            $cyclesLeft = [int]($sysNow.AvailableMb / $meanDrift)
            Write-Host ("  Extrapolation: ~{0} more cycles before OOM (available {1} MB / drift {2} MB)." -f `
                $cyclesLeft, $sysNow.AvailableMb, $meanDrift) -ForegroundColor Red
        }
        Write-Host "  Workaround: run 'foundry server stop' between model reloads for a clean process." -ForegroundColor Red
    } elseif ($meanDrift -ge 100) {
        Write-Host "VERDICT: SUSPICIOUS. ~$meanDrift MB retained per unload+reload cycle." -ForegroundColor DarkYellow
        Write-Host "  Could be caching (kernel/JIT/staging) that plateaus, or a slow leak. Run more cycles to distinguish." -ForegroundColor DarkYellow
    } else {
        Write-Host "VERDICT: OK. Unload releases memory equivalent to load; no inter-run leak detected." -ForegroundColor Green
    }
}

# ----------------------------------------------------------------------------
# Multi-turn conversation test
# ----------------------------------------------------------------------------
# Simulates a real chat: each turn appends a new user message + the previous
# assistant reply to the running history, then re-sends the full transcript.
# This is what production chat apps do -- input tokens grow monotonically
# each turn, exercising KV cache growth and the O(N^2) attention scores
# allocation at progressively larger sequence lengths.
#
# For each turn we report:
#   * input tokens (approximate: prior context + this turn's user filler)
#   * process + system memory before/after the turn
#   * TTFT and total wall time
#
# Contrast with a single-turn size sweep: here the request payload contains
# the full history, so we see how memory scales with conversation length in
# realistic usage. Stops before exceeding the context window if known.
function Invoke-MultiTurnTest {
    param(
        [int]$Turns,
        [int]$TokensPerTurn,   # user filler tokens added per turn (from -Sizes[0])
        [int]$OutputTokens,    # -MaxTokens (assistant cap per turn)
        [int]$ContextLength,   # 0 = unknown; otherwise cap for the OOM guard
        $ModelConfig           # optional hashtable from Get-ModelConfig for theoretical KV/Attn
    )

    Write-Host ""
    Write-Step "Multi-turn conversation test: $Turns turns of ~$TokensPerTurn user tokens each"
    Write-Host "    Each turn appends a new user message + prior assistant reply to the history,"
    Write-Host "    then sends the whole transcript. Context grows monotonically."
    if ($ContextLength -gt 0) {
        Write-Host "    Context cap: $ContextLength tokens (turns that would exceed it are skipped)."
    } else {
        Write-Host "    Context cap: unknown - will run until failure or -MultiTurn N reached." -ForegroundColor DarkYellow
    }
    Write-Host ""

    $fmt = "{0,-6}{1,-11}{2,-22}{3,-22}{4,-7}{5,-9}{6,-9}{7,-8}"
    Write-Host ($fmt -f "Turn","InputTok","Proc(MB)[d]","Sys(MB)[d]","%Ctx","TTFT(s)","Total(s)","Result") -ForegroundColor Yellow
    Write-Host ("-" * 98)

    # Baseline row (memory just before the first turn) for continuity with the size-sweep table.
    $baseProc = Get-VramUsedMb
    $baseSys  = Get-SystemUsedMb
    $bpStr    = if ($null -ne $baseProc) { "$baseProc" } else { "n/a" }
    $bsStr    = if ($null -ne $baseSys)  { "$baseSys"  } else { "n/a" }
    Write-Host ($fmt -f "pre", "-", $bpStr, $bsStr, "-", "-", "-", "BASELINE") -ForegroundColor DarkCyan

    $messages          = @()
    $cumulativeContext = 0     # running estimate of tokens the model sees BEFORE this turn's user message is added
    $turnDeltas        = @()
    $turnsCompleted    = 0
    $lastProcAfter     = $null
    $lastSysAfter      = $null

    for ($t = 1; $t -le $Turns; $t++) {
        # Build the new user message for this turn (honours -Prompt / -PromptLength).
        $newUserPrompt = New-Prompt $TokensPerTurn
        $projectedInputTokens = $cumulativeContext + $TokensPerTurn

        if ($ContextLength -gt 0 -and ($projectedInputTokens + $OutputTokens) -gt $ContextLength) {
            $pctStr = "{0}%" -f [int](($projectedInputTokens / $ContextLength) * 100)
            Write-Host ($fmt -f $t, $projectedInputTokens, "skip", "skip", $pctStr, "-", "-", "CTX-CAP") -ForegroundColor DarkGray
            Write-Host "    Next turn would exceed the context window - stopping." -ForegroundColor DarkGray
            break
        }

        # Append user turn using unary comma so a single hashtable is added as one element.
        $messages += ,@{ role = "user"; content = $newUserPrompt }

        $memBefore = Get-VramUsedMb
        $sysBefore = Get-SystemUsedMb

        $r = Invoke-Chat $messages

        $memAfter = Get-VramUsedMb
        $sysAfter = Get-SystemUsedMb
        $memDelta = if (($null -ne $memBefore) -and ($null -ne $memAfter)) { [int]($memAfter - $memBefore) } else { $null }
        $sysDelta = if (($null -ne $sysBefore) -and ($null -ne $sysAfter)) { [int]($sysAfter - $sysBefore) } else { $null }

        $resultStr = if ($r.ok) { "OK" } else { "FAIL" }
        $color     = if ($r.ok) { "Green" } else { "Red" }
        $pctStr    = if ($ContextLength -gt 0) { "{0}%" -f [int](($projectedInputTokens / $ContextLength) * 100) } else { "n/a" }
        $procStr   = if ($null -ne $memAfter) { "{0} ({1:+#;-#;+0})" -f $memAfter, $memDelta } else { "n/a" }
        $sysStr    = if ($null -ne $sysAfter) { "{0} ({1:+#;-#;+0})" -f $sysAfter, $sysDelta } else { "n/a" }
        $timeStr   = [math]::Round($r.ms / 1000, 1)
        $ttftStr   = if ($null -ne $r.ttfbMs) { [math]::Round($r.ttfbMs / 1000, 2) } else { "-" }

        Write-Host ($fmt -f $t, $projectedInputTokens, $procStr, $sysStr, $pctStr, $ttftStr, $timeStr, $resultStr) -ForegroundColor $color

        # Theoretical reference every 5 turns (and on turn 1). Input grows every turn,
        # so a fixed reference like the size-sweep prints doesn't apply here.
        if ($ModelConfig -and ($t -eq 1 -or ($t % 5) -eq 0)) {
            $kvMb   = Get-KvCacheMb        $projectedInputTokens $ModelConfig
            $attnMb = Get-AttentionScoresMb $projectedInputTokens $ModelConfig
            if ($null -ne $kvMb) {
                Write-Host ("             theoretical @ input={0}: KV={1} MB, Attn(O(N^2))={2} MB" -f $projectedInputTokens, $kvMb, $attnMb) -ForegroundColor DarkGray
            }
        }

        if (-not $r.ok) {
            $errDetail = if ([string]::IsNullOrWhiteSpace($r.detail)) { "(empty error)" } else { $r.detail }
            Write-Host "    Full error: $errDetail" -ForegroundColor DarkRed
            break
        }

        # Append the assistant reply so the next turn's request carries it.
        $asstText = if ($r.PSObject.Properties['content'] -and $null -ne $r.content) { [string]$r.content } else { "" }
        $messages += ,@{ role = "assistant"; content = $asstText }

        # Update running context. Assistant contribution is approximated at $OutputTokens
        # (the cap); real replies may be shorter, making this slightly pessimistic vs. actual,
        # which is fine for the context-window guard.
        $cumulativeContext = $projectedInputTokens + $OutputTokens

        if ($null -ne $memDelta) { $turnDeltas += $memDelta }
        $turnsCompleted = $t
        $lastProcAfter  = $memAfter
        $lastSysAfter   = $sysAfter
    }

    Write-Host ""
    Write-Host "Multi-turn summary:" -ForegroundColor Yellow
    Write-Host ("  Turns completed             : {0} / {1}" -f $turnsCompleted, $Turns)
    Write-Host ("  Estimated final context     : ~{0} tokens" -f $cumulativeContext)
    if ($turnDeltas.Count -gt 0) {
        $meanDelta = [int](($turnDeltas | Measure-Object -Average).Average)
        $maxDelta  = ($turnDeltas | Measure-Object -Maximum).Maximum
        $minDelta  = ($turnDeltas | Measure-Object -Minimum).Minimum
        Write-Host ("  Process delta per turn      : mean {0:+#;-#;0} MB   range [{1:+#;-#;0}, {2:+#;-#;0}]" -f $meanDelta, $minDelta, $maxDelta)
    }
    if ($null -ne $lastProcAfter) {
        Write-Host ("  Final process memory        : {0} MB" -f $lastProcAfter)
    }
    if ($null -ne $lastSysAfter) {
        Write-Host ("  Final system memory used    : {0} MB" -f $lastSysAfter)
    }
}

# -- 0) Prerequisite -----------------------------------------------------------
if (-not (Get-Command foundry -ErrorAction SilentlyContinue)) {
    Fail "Foundry Local CLI not found on PATH."
}

# -- 1) Ensure service is running -----------------------------------------------
function Get-ServiceBase {
    $out = (& foundry server status 2>&1 | Out-String)
    $m = [regex]::Match($out, 'http://[^/\s]+')
    if ($m.Success) { return $m.Value.TrimEnd('/') }
    return $null
}

# If -RestartService is set, kill the daemon first so we get a fresh process.
# This works around the known unload-path leak in Foundry Local, where every
# `foundry model unload` retains most of the model's CPU-side memory. Without
# this, running the script repeatedly leaves foundrylocald accumulating GB of
# retained memory across invocations.
if ($RestartService) {
    Write-Step "Stopping Foundry Local service (for a clean process baseline)..."
    & foundry server stop 2>&1 | Out-Null
    # Give the OS a moment to reap the process and release the port.
    Start-Sleep -Seconds 3
    # Belt and braces: kill any lingering foundrylocald process by name.
    Get-Process -Name foundrylocald -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "      Force-killing residual foundrylocald pid=$($_.Id)" -ForegroundColor DarkGray
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

Write-Step "Ensuring the Foundry Local service is running..."
$base = Get-ServiceBase
if (-not $base) {
    Start-Process -FilePath "foundry" -ArgumentList "server", "start" -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 30 -and -not $base; $i++) {
        Start-Sleep -Seconds 2
        $base = Get-ServiceBase
    }
}
if (-not $base) { Fail "The Foundry service did not come up." }

$chatUrl = "$base/v1/chat/completions"
$modelsUrl = "$base/v1/models"

# Unload any previously loaded models to get a clean baseline.
# We measure memory BEFORE and AFTER the unload so we can directly report
# how much CPU-side memory `foundry model unload` actually freed. This is
# the primary signal for the inter-run leak: if unload returns success but
# frees essentially nothing, that's the bug.
Write-Host ""
Write-Host "    Clearing any previously loaded models..." -ForegroundColor DarkGray
$preUnloadProc = Get-VramUsedMb
$preUnloadSys  = Get-SystemUsedMb
$unloadedCount = 0
try {
    $models = Invoke-RestMethod -Uri $modelsUrl -TimeoutSec 10
    if ($models.data -and $models.data.Count -gt 0) {
        foreach ($m in $models.data) {
            Write-Host "      Unloading $($m.id)..." -ForegroundColor DarkGray
            & foundry model unload $($m.id) 2>&1 | Out-Null
            $unloadedCount++
        }
    }
} catch { }
Start-Sleep -Seconds 1

# BASELINE: Measure memory before loading any model
$memBaseline    = Get-VramUsedMb
$sysBaseline    = Get-SystemMemoryMb

# Compute what the unload step actually freed on the CPU side.
$unloadFreedProc = $null
$unloadFreedSys  = $null
if ($unloadedCount -gt 0 -and $null -ne $preUnloadProc -and $null -ne $memBaseline) {
    $unloadFreedProc = $preUnloadProc - $memBaseline
}
if ($unloadedCount -gt 0 -and $null -ne $preUnloadSys -and $sysBaseline) {
    $unloadFreedSys = $preUnloadSys - $sysBaseline.UsedMb
}

Write-Host ""
Write-Host "    MEMORY BASELINE (service only, no model loaded):" -ForegroundColor Yellow
Write-Host "      foundrylocald process memory: $(if ($null -ne $memBaseline) { "$memBaseline MB" } else { "n/a (process not found)" })" -ForegroundColor Yellow
if ($sysBaseline) {
    Write-Host ("      system memory used         : {0} MB / {1} MB total ({2} MB available)" -f `
        $sysBaseline.UsedMb, $sysBaseline.TotalMb, $sysBaseline.AvailableMb) -ForegroundColor Yellow
    if ($null -ne $sysBaseline.CommittedMb) {
        Write-Host ("      committed (RAM + pagefile) : {0} MB" -f $sysBaseline.CommittedMb) -ForegroundColor DarkYellow
    }
}

# Direct unload-effectiveness report. This is the strongest single signal
# for the inter-run leak.
if ($unloadedCount -gt 0 -and $null -ne $unloadFreedProc) {
    Write-Host ""
    Write-Host ("    UNLOAD EFFECTIVENESS ({0} model(s) unloaded):" -f $unloadedCount) -ForegroundColor Yellow
    Write-Host ("      Process memory before unload: {0,6} MB" -f $preUnloadProc)
    Write-Host ("      Process memory after  unload: {0,6} MB" -f $memBaseline)
    $sign = if ($unloadFreedProc -ge 0) { "" } else { "-" }
    $absFreed = [Math]::Abs($unloadFreedProc)
    Write-Host ("      => unload freed             : {0}{1,5} MB" -f $sign, $absFreed)

    # A healthy unload should free most of what the load added (typically
    # ~90%+ of the model file size). If unload freed less than 500 MB while
    # the process was holding > 1 GB, that's the leak.
    if ($preUnloadProc -gt 1000 -and $unloadFreedProc -lt 500) {
        Write-Host ""
        Write-Host "    !! ALERT: foundry model unload appears to NOT be freeing CPU-side memory. !!" -ForegroundColor Red
        Write-Host ("    !! Held {0} MB before unload, still holding {1} MB after (freed only {2} MB). !!" -f `
            $preUnloadProc, $memBaseline, $unloadFreedProc) -ForegroundColor Red
        Write-Host "    !! This is the inter-run leak. Every load+unload cycle retains ~a model's" -ForegroundColor Red
        Write-Host "    !! worth of RAM in foundrylocald. Run './foundry-oom-repro.ps1 -RestartService ...'" -ForegroundColor Red
        Write-Host "    !! for a clean process, or './foundry-oom-repro.ps1 -LoadCycleTest 5' to quantify." -ForegroundColor Red
    } elseif ($preUnloadProc -gt 1000 -and $unloadFreedProc -lt ($preUnloadProc * 0.5)) {
        Write-Host ""
        Write-Host "    !! WARNING: unload freed less than half of what was in memory." -ForegroundColor DarkYellow
        Write-Host ("    !! ({0} MB freed out of {1} MB held). Partial retention." -f `
            $unloadFreedProc, $preUnloadProc) -ForegroundColor DarkYellow
    }
}

# Inter-run leak detector: a fresh foundrylocald with no model loaded is
# typically ~150-300 MB. If the baseline is much higher than that AFTER
# the unload step above, it means one or more previous loads left CPU-side
# memory in the process (the inter-run retention bug). We fire this
# whenever baseline is high, regardless of whether we just unloaded
# anything -- the script always tries to unload first, so a high baseline
# here is by definition post-unload.
if ($null -ne $memBaseline -and $memBaseline -gt 1000) {
    Write-Host ""
    Write-Host "    WARNING: baseline is $memBaseline MB even after unloading all models." -ForegroundColor Red
    Write-Host "    A freshly-started foundrylocald with no model is typically ~150-300 MB." -ForegroundColor Red
    Write-Host "    This means a previous load left CPU-side memory in the process (inter-run" -ForegroundColor Red
    Write-Host "    retention). Numbers below include that carry-over. To isolate it, use:" -ForegroundColor Red
    Write-Host "        ./foundry-oom-repro.ps1 -RestartService ...   # stop+restart for a clean run" -ForegroundColor Red
    Write-Host "        ./foundry-oom-repro.ps1 -LoadCycleTest 5 -Model $Model   # quantify the leak" -ForegroundColor Red
}
Write-Host "    Service base : $base"
Write-Host "    Chat endpoint: $chatUrl"

# -- 2) Detect available execution providers ----------------------------------
function Get-AvailableEPs {
    $eps = [System.Collections.Generic.List[string]]::new()
    try {
        $text = (& foundry model list 2>&1 | Out-String)
        if ($text -match 'cuda-gpu')    { $eps.Add('cuda-gpu') }
        if ($text -match 'openvino')    { $eps.Add('openvino') }
        if ($text -match 'generic-gpu') { $eps.Add('webgpu') }
        if ($text -match '\bqnn\b')     { $eps.Add('qnn') }
    } catch { }
    $eps.Add('cpu')
    return $eps.ToArray()
}

function Get-AllModels {
    $text = (& foundry model list 2>&1 | Out-String)
    $models = [System.Collections.Generic.List[PSCustomObject]]::new()
    $lastModel = $null
    foreach ($line in ($text -split "`r?`n")) {
        if ($line -match '\|\s+(.+?)\s+\|\s+(Chat|Multimodal|Speech|Embedding)\s+\|.+\|\s+(GPU|CPU|NPU)\s+\|') {
            $lastModel = [PSCustomObject]@{
                Name   = $Matches[1].Trim()
                Type   = $Matches[2].Trim()
                Device = $Matches[3].Trim()
            }
            $models.Add($lastModel)
        } elseif ($lastModel -and $line -match '\|\s+(\S+)\s+\|\s+\|') {
            $lastModel.Name = $lastModel.Name + $Matches[1].Trim()
            $lastModel = $null
        } else {
            $lastModel = $null
        }
    }
    return $models.ToArray()
}

if ([string]::IsNullOrWhiteSpace($Model)) {
    $availableEPs = @(Get-AvailableEPs)
    Write-Step "Detected execution providers: $($availableEPs -join ', ')"

    Write-Step "Selecting a small Chat model..."
    $allModels = @(Get-AllModels | Where-Object { $_.Type -eq 'Chat' })

    if ($allModels.Count -eq 0) {
        Fail "No Chat models found in 'foundry model list'."
    }

    $gpuEPs = @('cuda-gpu', 'openvino', 'webgpu', 'qnn')
    $hasGpuEP = ($availableEPs | Where-Object { $gpuEPs -contains $_ }).Count -gt 0
    $preferredDevice = if ($hasGpuEP) { 'GPU' } else { 'CPU' }
    Write-Host "    Preferred device: $preferredDevice"

    $filteredByDevice = @($allModels | Where-Object { $_.Device -eq $preferredDevice })
    if ($filteredByDevice.Count -eq 0) { $filteredByDevice = $allModels }

    $candidates = @()
    foreach ($p in @('0.5b', '0.6b', '1.5b', '1.7b', '1b', '2b', '3b', 'qwen2.5-0.5b', 'qwen2.5-1.5b', 'phi-3.5-mini', 'phi-4-mini')) {
        foreach ($m in ($filteredByDevice | Where-Object { $_.Name -like "*$p*" })) {
            if ($candidates -notcontains $m.Name) { $candidates += $m.Name }
        }
    }
    if ($candidates.Count -eq 0) { $candidates = @($filteredByDevice | Select-Object -ExpandProperty Name) }
} else {
    $candidates = @($Model)
}
Write-Host ("    Candidates: {0}" -f ($candidates -join ", "))

# -- 3) Download and load model -----------------------------------------------
$apiModel = $null
foreach ($cand in $candidates) {
    Write-Step "Downloading '$cand' if needed..."
    & foundry model download $cand

    Write-Step "Loading '$cand' into the service..."
    $loadLines = @()
    & foundry model load $cand 2>&1 | Tee-Object -Variable loadLines
    $loadOut = ($loadLines | Out-String)

    if ($loadOut -match '(?im)Failed loading|Internal Server Error|JSON Error|Exception:') {
        Write-Host "    '$cand' failed to LOAD - trying next." -ForegroundColor Yellow
        continue
    }

    $Model = $cand
    $apiModel = $Model
    try {
        $models = Invoke-RestMethod -Uri $modelsUrl -TimeoutSec 30
        if ($models.data -and $models.data.Count -gt 0) {
            $match = $models.data | Where-Object { $_.id -match [regex]::Escape($Model) } | Select-Object -First 1
            if (-not $match) { $match = $models.data | Select-Object -First 1 }
            if ($match -and $match.id) { $apiModel = $match.id }
        }
    } catch { }
    break
}

if (-not $apiModel) {
    Fail "Could not load any model."
}
Write-Host "    Loaded model: $Model (API id: $apiModel)"

# Extract model file size from load output (e.g., "Loading model (qwen3-4b-generic-cpu:3, 2.7 GB)")
$modelFileSizeMb = $null
if ($loadOut -match 'Loading model.*,\s*([\d.]+)\s*GB') {
    $sizeGb = [float]$matches[1]
    $modelFileSizeMb = [int]($sizeGb * 1024)
}

# Measure memory immediately after model load
$memAfterLoad  = Get-VramUsedMb
$sysAfterLoad  = Get-SystemMemoryMb
Write-Host "    Memory after load: $(if ($null -ne $memAfterLoad) { "$memAfterLoad MB (process)" } else { "n/a" })" -ForegroundColor Cyan
if ($sysAfterLoad) {
    $sysDelta = if ($sysBaseline) { $sysAfterLoad.UsedMb - $sysBaseline.UsedMb } else { $null }
    $deltaStr = if ($null -ne $sysDelta) { " (+$sysDelta MB system)" } else { "" }
    Write-Host ("                       {0} MB system used / {1} MB available{2}" -f `
        $sysAfterLoad.UsedMb, $sysAfterLoad.AvailableMb, $deltaStr) -ForegroundColor Cyan
}

# Try to get model config for KV cache calculations
$modelConfig = Get-ModelConfig $Model
if ($modelConfig) {
    $gqaNote = if ($modelConfig.num_key_value_heads -lt $modelConfig.num_attention_heads) {
        " (GQA $($modelConfig.num_attention_heads):$($modelConfig.num_key_value_heads))"
    } else { "" }
    Write-Host ("    Model architecture: {0} attn heads / {1} KV heads{2}, {3} layers, head_dim={4}, dtype={5}" -f `
        $modelConfig.num_attention_heads, $modelConfig.num_key_value_heads, $gqaNote, `
        $modelConfig.num_hidden_layers, $modelConfig.head_dim, $modelConfig.dtype) -ForegroundColor DarkGray
    if ($modelConfig.source_file) {
        Write-Host ("    Model config source: {0} ({1})" -f $modelConfig.source_file, $modelConfig.source_schema) -ForegroundColor DarkGray
    }
} else {
    Write-Host "    Model config: not found (looked for genai_config.json / config.json in Foundry cache + HF caches)" -ForegroundColor DarkGray
    Write-Host "    -> theoretical KV/Attn memory estimates will be skipped." -ForegroundColor DarkGray
}

# Check model file size and calculate overhead
if ($null -ne $modelFileSizeMb -and $null -ne $memAfterLoad) {
    Write-Host ""
    Write-Host "    MEMORY BREAKDOWN AFTER MODEL LOAD:" -ForegroundColor Cyan
    Write-Host "      Baseline (service only):        $memBaseline MB"
    Write-Host "      Model file size (on disk):      $modelFileSizeMb MB"
    Write-Host "      Total after load:              $memAfterLoad MB"
    Write-Host ""
    
    # Calculate unknowns
    $serviceOverhead     = $memBaseline
    $modelLoadedInMemory = $memAfterLoad - $memBaseline
    $unexplainedOverhead = $modelLoadedInMemory - $modelFileSizeMb

    # If we know the config, we can quantify KV cache pre-allocation.
    # With past_present_share_buffer=true, the full-context KV buffer is
    # allocated at load time (once, not per-request).
    $kvPreallocMb = $null
    if ($modelConfig -and $modelConfig.max_position_embeddings) {
        $kvPreallocMb = Get-KvCacheMb $modelConfig.max_position_embeddings $modelConfig
    }

    # Attention scratch peak at max context - NOT in the load-time totals above.
    # This is the O(N^2) softmax-input tensor per layer (fp32, only 1 layer alive).
    # It's the real OOM driver at large inputs; users need this as headroom on top of load.
    $attnPeakMb = $null
    if ($modelConfig -and $modelConfig.max_position_embeddings) {
        $attnPeakMb = Get-AttentionScoresMb $modelConfig.max_position_embeddings $modelConfig
    }

    Write-Host "    ANALYSIS:" -ForegroundColor DarkGray
    Write-Host "      Service infrastructure:        $serviceOverhead MB" -ForegroundColor DarkGray
    Write-Host "      Model + overhead combined:     $modelLoadedInMemory MB" -ForegroundColor DarkGray
    Write-Host "      |--- Model weights (from disk):  $modelFileSizeMb MB" -ForegroundColor DarkGray
    if ($null -ne $kvPreallocMb) {
        $residual = $unexplainedOverhead - $kvPreallocMb
        Write-Host ("      |--- KV cache pre-alloc @ ctx={0}: {1} MB ({2} dtype, GQA-aware)" -f `
            $modelConfig.max_position_embeddings, $kvPreallocMb, $modelConfig.dtype) -ForegroundColor DarkGray
        Write-Host ("      \--- Residual (runtime/allocator): {0} MB" -f $residual) -ForegroundColor DarkGray
    } else {
        Write-Host "      \--- Unexplained growth:         $unexplainedOverhead MB" -ForegroundColor DarkGray
    }
    if ($null -ne $attnPeakMb) {
        Write-Host "" 
        Write-Host "    PREDICTED PEAK REQUEST HEADROOM (not in load totals above):" -ForegroundColor DarkGray
        Write-Host ("      Attention scratch @ ctx={0}: ~{1} MB   (fp32, single layer alive, O(N^2))" -f `
            $modelConfig.max_position_embeddings, $attnPeakMb) -ForegroundColor DarkGray
        Write-Host ("      => Total needed for a full-context request: ~{0} MB process" -f `
            ($memAfterLoad + $attnPeakMb)) -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "    Notes:" -ForegroundColor DarkGray
    Write-Host "      * KV pre-alloc assumes past_present_share_buffer=true (full-context buffer at load)." -ForegroundColor DarkGray
    Write-Host "      * Residual = ONNX Runtime session state + graph compilation + allocator overhead." -ForegroundColor DarkGray
    Write-Host "      * Attention scratch is NOT allocated at load time; it grows per-request as O(N^2)" -ForegroundColor DarkGray
    Write-Host "        of input length. The value above is the ceiling at max context." -ForegroundColor DarkGray
} else {
    if ($null -eq $modelFileSizeMb) {
        Write-Host ""
        Write-Host "    Memory breakdown: (unable to parse model size from load output)" -ForegroundColor DarkGray
    }
}

# -- 3a) Optional: load/unload cycle test -------------------------------------
# When -LoadCycleTest N is given, skip the size sweep entirely and instead run
# N unload+load cycles to isolate the inter-run memory retention bug.
if ($LoadCycleTest -gt 0) {
    if ($null -eq $memBaseline -or $null -eq $memAfterLoad -or -not $sysBaseline -or -not $sysAfterLoad) {
        Fail "LoadCycleTest requires baseline + after-load memory samples, but one is missing."
    }
    Invoke-LoadCycleTest `
        -ModelAlias    $Model `
        -ApiId         $apiModel `
        -Cycles        $LoadCycleTest `
        -InitialProc   $memBaseline `
        -InitialSys    $sysBaseline.UsedMb `
        -AfterLoadProc $memAfterLoad `
        -AfterLoadSys  $sysAfterLoad.UsedMb `
        -ModelConfig   $modelConfig
    exit 0
}

# -- 3b) Get context length ---------------------------------------------------
function Get-ContextLength([string]$alias) {
    try {
        $info = & foundry model info $alias 2>&1 | Out-String
        if ($info -match "Context Length\s*\|\s*(\d+)") {
            return [int]$matches[1]
        }
    } catch { }
    return $null
}

$contextLength = Get-ContextLength $Model
if ($contextLength) {
    Write-Host "    Context window : $contextLength tokens"
} else {
    Write-Host "    Context window : unknown" -ForegroundColor DarkGray
}

# -- 3bb) Optional: multi-turn conversation test ------------------------------
# When -MultiTurn N is given, skip the size sweep and instead run an N-turn
# simulated conversation that grows the message history each turn. Uses
# -Sizes[0] as the number of filler tokens each new user message adds.
if ($MultiTurn -gt 0) {
    $tokensPerTurn = if ($Sizes.Count -gt 0) { $Sizes[0] } else { 128 }
    $ctxArg        = if ($null -ne $contextLength -and $contextLength -gt 0) { $contextLength } else { 0 }
    Invoke-MultiTurnTest `
        -Turns          $MultiTurn `
        -TokensPerTurn  $tokensPerTurn `
        -OutputTokens   $MaxTokens `
        -ContextLength  $ctxArg `
        -ModelConfig    $modelConfig
    exit 0
}

# -- 3c) Display available Chat models ----------------------------------------
Write-Host ""
Write-Step "Available Chat models (for this device):"
$allModels = @(Get-AllModels | Where-Object { $_.Type -eq 'Chat' })
$availableEPs = @(Get-AvailableEPs)
Write-Host "    Supported execution providers: $($availableEPs -join ', ')" -ForegroundColor Cyan

$gpuEPs = @('cuda-gpu', 'openvino', 'webgpu', 'qnn')
$hasGpuEP = ($availableEPs | Where-Object { $gpuEPs -contains $_ }).Count -gt 0
$preferredDevice = if ($hasGpuEP) { 'GPU' } else { 'CPU' }

$deviceModels = @($allModels | Where-Object { $_.Device -eq $preferredDevice })
if ($deviceModels.Count -gt 0) {
    Write-Host ""
    Write-Host ("{0,-45} {1}" -f "Model", "Context Length") -ForegroundColor Yellow
    Write-Host ("-" * 60)
    foreach ($m in $deviceModels) {
        $ctx = Get-ContextLength $m.Name
        $ctxStr = if ($ctx) { "{0:N0}" -f $ctx } else { "unknown" }
        $marker = if ($m.Name -eq $Model) { "-> " } else { "  " }
        Write-Host ("{0}{1,-43} {2}" -f $marker, $m.Name, $ctxStr)
    }
    Write-Host ""
} else {
    Write-Host "    (No Chat models found for this device)" -ForegroundColor DarkGray
}

# -- 4) Step up the input size -------------------------------------------------
Write-Step "Stepping up input size (output capped at $MaxTokens tokens to isolate input handling)..."
if ($PromptLength -gt 0) {
    Write-Host "    Prompt: $PromptLength-token base + filler to reach each size"
} elseif ([string]::IsNullOrWhiteSpace($Prompt)) {
    Write-Host "    Prompt: filler text (data data data...)"
} else {
    Write-Host "    Prompt: custom base + filler (base ~$($Prompt.Split().Count) tokens, then filler to reach size)"
}
if ($contextLength) { 
    Write-Host "    Skipping any size where input+output would exceed the $contextLength-token context window." 
} else {
    Write-Host "    Context window not detected - all test sizes will be attempted (Ctx% will be incorrect)." -ForegroundColor DarkYellow
}
Write-Host ""

# System Configuration Summary
Write-Host "Starting memory profiling..." -ForegroundColor Cyan
Write-Host ""

# Take a single "pre-input" reading now, before any request is sent.
# This gives us:
#   - a baseline the first iteration's delta is measured against
#   - a row printed at the top of the table so the reader can see the starting point
$preMemProc = Get-VramUsedMb
$preMemSys  = Get-SystemUsedMb

Write-Host "Memory stages:" -ForegroundColor Cyan
Write-Host ("  Baseline (service only)     : {0} MB process / {1} MB system" -f `
    $(if ($null -ne $memBaseline) { $memBaseline } else { 'n/a' }), `
    $(if ($sysBaseline)          { $sysBaseline.UsedMb } else { 'n/a' })) -ForegroundColor DarkGray
Write-Host ("  After model load            : {0} MB process / {1} MB system" -f `
    $(if ($null -ne $memAfterLoad) { $memAfterLoad } else { 'n/a' }), `
    $(if ($sysAfterLoad)          { $sysAfterLoad.UsedMb } else { 'n/a' })) -ForegroundColor DarkGray
Write-Host ("  Before first inference      : {0} MB process / {1} MB system" -f `
    $(if ($null -ne $preMemProc) { $preMemProc } else { 'n/a' }), `
    $(if ($null -ne $preMemSys)  { $preMemSys  } else { 'n/a' })) -ForegroundColor DarkGray
Write-Host "  (KV cache is allocated during graph compilation on the first run.)" -ForegroundColor DarkGray
Write-Host ""

# Columns:
#   Proc(MB)[d] = foundrylocald PrivateMemorySize + delta (per-process working set)
#   Sys(MB)[d]  = OS-wide used physical RAM + delta (TotalVisible - FreePhysical)
#   Attn(MB)    = theoretical attention-scratch peak for this input size
#                 (num_attention_heads * N^2 * 4 bytes, single layer alive, fp32).
#                 Requires model config; shows "-" if config wasn't discoverable.
#   Sys is the real OOM signal - BFCArena fails when RAM+pagefile is exhausted.
Write-Host ("{0,-11}{1,-10}{2,-22}{3,-22}{4,-10}{5,-7}{6,-9}{7,-9}{8,-8}" -f `
    "InputTok", "MaxOut", "Proc(MB)[d]", "Sys(MB)[d]", "Attn(MB)", "%Ctx", "TTFT(s)", "Total(s)", "Result") -ForegroundColor Yellow
Write-Host ("-" * 108)

# Pre-input baseline row (no delta - this IS the starting point).
$preProcStr = if ($null -ne $preMemProc) { "$preMemProc" } else { "n/a" }
$preSysStr  = if ($null -ne $preMemSys)  { "$preMemSys"  } else { "n/a" }
Write-Host ("{0,-11}{1,-10}{2,-22}{3,-22}{4,-10}{5,-7}{6,-9}{7,-9}{8,-8}" -f `
    "pre-input", "-", $preProcStr, $preSysStr, "-", "-", "-", "-", "BASELINE") -ForegroundColor DarkCyan

$firstFailure = $null
$measurements = [System.Collections.Generic.List[PSCustomObject]]::new()

try {
    foreach ($n in $Sizes) {
        if ($contextLength -and ($n + $MaxTokens) -gt $contextLength) {
            $pctStr = "{0}%" -f [int](($n / $contextLength) * 100)
            Write-Host ("{0,-11}{1,-10}{2,-22}{3,-22}{4,-10}{5,-7}{6,-9}{7,-9}{8,-8}" -f $n, $MaxTokens, "skip", "skip", "-", $pctStr, "-", "-", "SKIP") -ForegroundColor DarkGray
            continue
        }

        for ($iter = 1; $iter -le $Iterations; $iter++) {
            [Console]::Out.Flush()
            [Console]::Error.Flush()

            # Measure memory before inference (process + system)
            $memBefore = Get-VramUsedMb
            $sysBefore = Get-SystemUsedMb
            
            $prompt = New-Prompt $n
            $r = Invoke-Chat $prompt
            
            # Measure memory after inference (process + system)
            $memAfter = Get-VramUsedMb
            $sysAfter = Get-SystemUsedMb
            $memDelta = if (($null -ne $memBefore) -and ($null -ne $memAfter)) { [int]($memAfter - $memBefore) } else { $null }
            $sysDelta = if (($null -ne $sysBefore) -and ($null -ne $sysAfter)) { [int]($sysAfter - $sysBefore) } else { $null }
            
            # Calculate theoretical per-request memory for comparison.
            # KV cache grows linearly in N. Attention scores (QK^T) grow O(N^2)
            # and are the actual driver of OOM at large input sizes.
            # Compute EVERY iteration so the Attn(MB) column is populated even
            # when -Iterations > 1 keeps N the same (e.g. -LeakTest mode).
            $kvMb   = $null
            $attnMb = $null
            if ($modelConfig) {
                $kvMb   = Get-KvCacheMb        $n $modelConfig
                $attnMb = Get-AttentionScoresMb $n $modelConfig
            }

            $result = if ($r.ok) { "OK" } else { "FAIL" }
            $color  = if ($r.ok) { "Green" } else { "Red" }
            $pctStr = if ($contextLength) { "{0}%" -f [int](($n / $contextLength) * 100) } else { "n/a" }
            $procStr = if ($null -ne $memAfter) { ("{0} ({1:+#;-#;+0})" -f $memAfter, $memDelta) } else { "n/a" }
            $sysStr  = if ($null -ne $sysAfter) { ("{0} ({1:+#;-#;+0})" -f $sysAfter, $sysDelta) } else { "n/a" }
            $attnStr = if ($null -ne $attnMb) { "$attnMb" } else { "-" }

            $timeStr = [math]::Round($r.ms / 1000, 1)
            $ttftStr = if ($null -ne $r.ttfbMs) { [math]::Round($r.ttfbMs / 1000, 2) } else { "-" }
            Write-Host ("{0,-11}{1,-10}{2,-22}{3,-22}{4,-10}{5,-7}{6,-9}{7,-9}{8,-8}" -f $n, $MaxTokens, $procStr, $sysStr, $attnStr, $pctStr, $ttftStr, $timeStr, $result) -ForegroundColor $color

            # Record measurement for leak analysis (process + system)
            $measurements.Add([PSCustomObject]@{
                Size      = $n
                Iter      = $iter
                MemBefore = $memBefore
                MemAfter  = $memAfter
                SysBefore = $sysBefore
                SysAfter  = $sysAfter
                OK        = $r.ok
                TimeMs    = $r.ms
            })

            [Console]::Out.Flush()
            [Console]::Error.Flush()
            
            if (-not $r.ok) {
                if (-not $firstFailure) {
                    $firstFailure = $n
                    $errDetail = if ([string]::IsNullOrWhiteSpace($r.detail)) { "(empty error)" } else { $r.detail }
                    Write-Host "    Full error: $errDetail" -ForegroundColor DarkRed
                    Write-Host "    First 100 chars of prompt: $($prompt.Substring(0, [Math]::Min(100, $prompt.Length)))" -ForegroundColor DarkGray
                }
                if (-not $KeepGoing) { break }
            }
            
            [Console]::Out.Flush()
            [Console]::Error.Flush()
            
            # Clear variables to prevent scope corruption
            $r = $null
            $prompt = $null
            $result = $null
            $pctStr = $null
            $errDetail = $null
        }
        
        if ($firstFailure) { break }
    }
} catch {
    Write-Host "ERROR during test loop: $_" -ForegroundColor DarkRed
    Write-Host "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor DarkRed
    exit 1
}

# -- 5) Summary ----------------------------------------------------------------
Write-Host ""
if ($firstFailure) {
    Write-Host "First failure at ~$firstFailure input tokens." -ForegroundColor Red
} else {
    Write-Host "No failure across the tested sizes on this machine. Try larger -Sizes." -ForegroundColor Green
}

# -- 6) Leak analysis ---------------------------------------------------------
# Only run when we have enough measurements for a meaningful regression.
# We analyze BOTH signals:
#   - Process memory (MemAfter): what foundrylocald holds. Retention-sensitive.
#   - System memory (SysAfter): OS-wide RAM used. If this grows without bound,
#     something is leaking somewhere (not necessarily foundrylocald).
$successful = @($measurements | Where-Object { $_.OK -and $null -ne $_.MemAfter })
if ($LeakTest -or $successful.Count -ge 5) {
    Write-Host ""
    Write-Step "Leak analysis (post-inference memory over iterations)"

    function Get-LeakStats {
        param([double[]]$series)
        $n = $series.Count
        if ($n -lt 2) { return $null }
        $xs = @(0..($n - 1))
        $sumX  = ($xs     | Measure-Object -Sum).Sum
        $sumY  = ($series | Measure-Object -Sum).Sum
        $sumXY = 0.0
        $sumX2 = 0.0
        for ($i = 0; $i -lt $n; $i++) {
            $sumXY += $xs[$i] * $series[$i]
            $sumX2 += $xs[$i] * $xs[$i]
        }
        $denom = ($n * $sumX2) - ($sumX * $sumX)
        $slope = if ($denom -ne 0) { (($n * $sumXY) - ($sumX * $sumY)) / $denom } else { 0.0 }
        return [PSCustomObject]@{
            N      = $n
            Slope  = $slope
            First  = $series[0]
            Last   = $series[-1]
            Peak   = ($series | Measure-Object -Maximum).Maximum
            Trough = ($series | Measure-Object -Minimum).Minimum
            Delta  = $series[-1] - $series[0]
        }
    }

    function Write-LeakVerdict {
        param([string]$label, [double]$slope, [int]$threshold)
        $absSlope = [Math]::Abs($slope)
        if ($absSlope -lt $threshold) {
            Write-Host ("    {0} verdict : NO LEAK - drift within allocator noise (< {1} MB/iter)" -f $label, $threshold) -ForegroundColor Green
        } elseif ($absSlope -lt ($threshold * 10)) {
            Write-Host ("    {0} verdict : SUSPICIOUS - {1:F1} MB/iter growth" -f $label, $slope) -ForegroundColor Yellow
        } else {
            $projected = [int]($slope * 1000)
            Write-Host ("    {0} verdict : PROBABLE LEAK - {1:F1} MB/iter (~{2} MB / 1000 iters)" -f $label, $slope, $projected) -ForegroundColor Red
        }
    }

    # Group by size, analyze each size independently.
    $groups = $successful | Group-Object -Property Size
    foreach ($g in $groups) {
        $samples = @($g.Group)
        if ($samples.Count -lt 3) { continue }

        # Drop the first iteration for each size (warm-up allocates the biggest working set).
        $warm = @($samples | Select-Object -Skip 1)
        if ($warm.Count -lt 3) { $warm = $samples }

        $procSeries = @($warm | ForEach-Object { [double]$_.MemAfter })
        $procStats  = Get-LeakStats $procSeries

        $sysStats = $null
        if ($warm[0].SysAfter -ne $null) {
            $sysSeries = @($warm | Where-Object { $null -ne $_.SysAfter } | ForEach-Object { [double]$_.SysAfter })
            if ($sysSeries.Count -ge 2) { $sysStats = Get-LeakStats $sysSeries }
        }

        Write-Host ""
        Write-Host ("  Size {0} tokens x {1} iterations (post-warmup n={2})" -f $g.Name, $samples.Count, $procStats.N) -ForegroundColor Cyan
        Write-Host ("    Process : first={0} MB, last={1} MB, trough/peak={2}/{3} MB, drift={4:+0;-0;0} MB, slope={5:F2} MB/iter" -f `
            $procStats.First, $procStats.Last, $procStats.Trough, $procStats.Peak, $procStats.Delta, $procStats.Slope)
        Write-LeakVerdict "Process" $procStats.Slope $LeakThresholdMb

        if ($sysStats) {
            Write-Host ("    System  : first={0} MB, last={1} MB, trough/peak={2}/{3} MB, drift={4:+0;-0;0} MB, slope={5:F2} MB/iter" -f `
                $sysStats.First, $sysStats.Last, $sysStats.Trough, $sysStats.Peak, $sysStats.Delta, $sysStats.Slope)
            Write-LeakVerdict "System " $sysStats.Slope $LeakThresholdMb
        }

        # Distinguish retention vs. real leak on the PROCESS series:
        # retention plateaus; a real leak keeps growing across both halves.
        if ($samples.Count -ge 10 -and [Math]::Abs($procStats.Slope) -ge $LeakThresholdMb) {
            $firstHalf  = @($samples | Select-Object -First ([int]($samples.Count / 2)))
            $secondHalf = @($samples | Select-Object -Last  ([int]($samples.Count / 2)))
            $fhAvg = ($firstHalf  | Measure-Object -Property MemAfter -Average).Average
            $shAvg = ($secondHalf | Measure-Object -Property MemAfter -Average).Average
            $halfDelta = $shAvg - $fhAvg
            Write-Host ("    First-half avg : {0:F0} MB    Second-half avg: {1:F0} MB    Delta: {2:+0;-0;0} MB" -f `
                $fhAvg, $shAvg, $halfDelta) -ForegroundColor DarkGray
            if ([Math]::Abs($halfDelta) -lt $LeakThresholdMb) {
                Write-Host "    -> Process growth plateaued: likely BFCArena retention, NOT a leak." -ForegroundColor Green
            } else {
                Write-Host "    -> Process growth continues across halves: LEAK CONFIRMED." -ForegroundColor Red
            }
        }

        # Cross-check: if system grows but process doesn't, the leak is elsewhere.
        if ($sysStats -and [Math]::Abs($sysStats.Slope) -ge $LeakThresholdMb -and [Math]::Abs($procStats.Slope) -lt $LeakThresholdMb) {
            Write-Host "    -> System growing but process flat: leak is in another process (or kernel/driver)." -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Leak detection guide:" -ForegroundColor DarkGray
    Write-Host "  < $LeakThresholdMb MB/iter          -> noise / allocator internal state" -ForegroundColor DarkGray
    Write-Host "  $LeakThresholdMb - $($LeakThresholdMb * 10) MB/iter  -> suspicious (could be retention)" -ForegroundColor DarkGray
    Write-Host "  > $($LeakThresholdMb * 10) MB/iter         -> probable leak" -ForegroundColor DarkGray
    Write-Host "  Plateau in 2nd half -> BFCArena retention (not a leak)" -ForegroundColor DarkGray
    Write-Host "  Continued growth    -> real leak" -ForegroundColor DarkGray
    Write-Host "  System>0 & Process=0 -> leak in another process" -ForegroundColor DarkGray
}

# -- 7) Final unload ----------------------------------------------------------
# Unload the model we loaded, and report how much memory the unload actually
# freed. This is the same call the next script run would make at its baseline
# cleanup step -- doing it here just moves the measurement earlier and leaves
# the daemon in a clean state on exit. It does NOT stop the daemon; use
# 'foundry server stop' or -RestartService for that.
if ($apiModel) {
    Write-Host ""
    Write-Step "Unloading '$apiModel' before exit..."
    $procBeforeUnload = Get-VramUsedMb
    $sysBeforeUnload  = Get-SystemUsedMb
    & foundry model unload $apiModel 2>&1 | Out-Null
    Start-Sleep -Seconds 1
    $procAfterUnload = Get-VramUsedMb
    $sysAfterUnload  = Get-SystemUsedMb

    if ($null -ne $procBeforeUnload -and $null -ne $procAfterUnload) {
        $procFreed = $procBeforeUnload - $procAfterUnload
        $sysFreed  = if ($null -ne $sysBeforeUnload -and $null -ne $sysAfterUnload) { $sysBeforeUnload - $sysAfterUnload } else { $null }
        $sysStr    = if ($null -ne $sysFreed) { " ({0} MB system)" -f $sysFreed } else { "" }
        $color     = if ($procBeforeUnload -gt 1000 -and $procFreed -lt 500) { "Red" } else { "DarkGray" }
        Write-Host ("    Process: {0} MB -> {1} MB  (freed {2} MB{3})" -f `
            $procBeforeUnload, $procAfterUnload, $procFreed, $sysStr) -ForegroundColor $color
        if ($procBeforeUnload -gt 1000 -and $procFreed -lt 500) {
            Write-Host "    Unload freed <500 MB despite >1 GB loaded -- inter-run retention bug is active." -ForegroundColor Red
            Write-Host "    Run 'foundry server stop' to fully release, or use -RestartService next run." -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Stop the service to free everything: foundry server stop" -ForegroundColor DarkGray
