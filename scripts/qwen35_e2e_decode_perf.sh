#!/usr/bin/env bash
set -euo pipefail

RUNS=7
PORT=50061
BUILD=1
PROMPT="你好，北京有什么好玩的景点？香山如何？早上去晚上去？单纯爬山么？还有什么可以在香山玩的？"
MODEL_ROOT="$HOME/Workspace/models"
TOKENIZER=""
MODEL_27B=""
MTP_27B=""
DSPARK_27B=""
MODEL_35B=""
MTP_35B=""
DSPARK_35B=""
MODEL_27B_REPO="mlx-community/Qwen3.8-27B-4bit"
MTP_27B_REPO="mlx-community/Qwen3.8-27B-MTP-4bit"
DSPARK_27B_REPO=""
MODEL_35B_REPO="mlx-community/Qwen3.6-35B-A3B-4bit"
MTP_35B_REPO="mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
DSPARK_35B_REPO=""
CASES="27b_off,27b_mtp,27b_dspark,35b_off,35b_mtp,35b_dspark"
BASELINE=1
CASE_COOLDOWN_SECS=8
LOGGING=info
SEED=42
NUM_CACHE_PAGES=393216
MAX_REQUESTS=4
MAX_TOKENS=128
MAX_TOKENS_PER_REQUEST=64
BASELINE_MACHINE="apple_m3_max_40_gpu_cores"
BASELINE_DATE="2026-07-21"
BASELINE_COMMIT="132c507380754302bd2e78a2d1a0abd4d1094d58"
BASELINE_OS_VERSION="27.0"
BASELINE_ARCH="arm64"
BASELINE_NUM_CACHE_PAGES=393216
BASELINE_CACHE_BLOCK_TOKENS=2048
BASELINE_MAX_REQUESTS=4
BASELINE_MAX_TOKENS=128
BASELINE_MAX_TOKENS_PER_REQUEST=64
BASELINE_CASE_COOLDOWN_SECS=8
BASELINE_LOGGING="info"
BASELINE_SEED=42
BASELINE_MIN_RUNS=3
BASELINE_PROMPT_SHA256="a4f0a10af5d122c667bfce70079dfc873cd384a28a879022b4c106b71feca78f"
BASELINE_TOKENIZER_DIR_NAME="Qwen3.6-35B-A3B-4bit"
BASELINE_MODEL_27B_DIR_NAME="Qwen3.6-27B-4bit"
BASELINE_MTP_27B_DIR_NAME="Qwen3.6-27B-MTP-4bit"
BASELINE_MODEL_35B_DIR_NAME="Qwen3.6-35B-A3B-4bit"
BASELINE_MTP_35B_DIR_NAME="Qwen3.6-35B-A3B-MTP-4bit"
CACHE_BLOCK_TOKENS=2048
ACTIVE_SERVER_PID=""

cleanup_active_server() {
    if [[ -n "$ACTIVE_SERVER_PID" ]]; then
        kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        wait "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        ACTIVE_SERVER_PID=""
    fi
}

trap cleanup_active_server EXIT
trap 'cleanup_active_server; exit 130' INT
trap 'cleanup_active_server; exit 143' TERM

usage() {
    cat <<'EOF'
Usage: scripts/qwen35_e2e_decode_perf.sh [options]

Runs Qwen3.5/3.6/3.8 replay e2e decode perf one server at a time and reports
decode throughput, TTFT, inter-chunk latency, and speculative acceptance.

Options:
  --runs N              Runs per token-count case. Default: 7
  --cases LIST          Comma-separated cases.
                        Default: 27b_off,27b_mtp,27b_dspark,35b_off,35b_mtp,35b_dspark
                        Available: 27b_off,27b_mtp,27b_dspark,35b_off,35b_mtp,35b_dspark
                        Each *_mtp or *_dspark case runs 1 and then 2 speculative tokens.
                        Group cases: 27b_on and 35b_on run both MTP and DSpark.
                        Missing Spec checkpoints without a repo are warned and skipped.
  --port N              Server port. Default: 50061
  --prompt TEXT         Prompt string.
  --seed N              Fixed request seed. Default: 42
  --num-cache-pages N   Shared cache pages. Default: 393216 (baseline workload)
  --max-requests N      Scheduler request capacity. Default: 4
  --max-tokens N        Scheduler flattened-token capacity. Default: 128
  --max-tokens-per-request N
                        Scheduler per-request token capacity. Default: 64
  --model-root DIR      Default checkpoint root. Default: $HOME/Workspace/models
  --tokenizer DIR       Tokenizer/chat-template override. Default: each case's Main model.
  --model-27b DIR       27B Main directory. Default: MODEL_ROOT/Qwen3.8-27B-4bit
  --mtp-27b DIR         27B MTP directory. Default: MODEL_ROOT/Qwen3.8-27B-MTP-4bit
  --dspark-27b DIR      27B DSpark directory. Default: MODEL_ROOT/Qwen3.8-27B-DSpark-affine
  --model-35b DIR       35B Main directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-4bit
  --mtp-35b DIR         35B MTP directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-MTP-4bit
  --dspark-35b DIR      35B DSpark directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-DSpark-affine
  --model-27b-repo REPO Hugging Face repo used if the 27B Main model is missing.
  --mtp-27b-repo REPO   Hugging Face repo used if the 27B MTP model is missing.
  --dspark-27b-repo REPO
                        Hugging Face repo used if the 27B affine DSpark model is missing.
  --model-35b-repo REPO Hugging Face repo used if the 35B Main model is missing.
  --mtp-35b-repo REPO   Hugging Face repo used if the 35B MTP model is missing.
  --dspark-35b-repo REPO
                        Hugging Face repo used if the 35B affine DSpark model is missing.
  --no-build            Skip release build.
  --no-baseline         Do not compare summary rows with the checked-in M3 Max baseline.
  --case-cooldown-secs N
                        Idle time between model cases. Default: 8
                        Pass 0 for an intentional sustained-load run.
  --logging LEVEL       Server logging: info or debug. Default: info.
                        Debug adds request/response and replay-stage details.
  -h, --help            Show this help.

Examples:
  scripts/qwen35_e2e_decode_perf.sh \
    --model-root "$HOME/Workspace/models" \
    --cases 27b_off,27b_mtp,27b_dspark,35b_off,35b_mtp,35b_dspark --runs 3

  scripts/qwen35_e2e_decode_perf.sh \
    --model-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-4bit" \
    --mtp-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-MTP-4bit" \
    --dspark-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-DSpark-affine" \
    --cases 35b_mtp,35b_dspark --runs 3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --runs)
        [[ $# -ge 2 ]] || {
            echo "--runs requires a value" >&2
            exit 2
        }
        RUNS="$2"
        shift 2
        ;;
    --cases)
        [[ $# -ge 2 ]] || {
            echo "--cases requires a value" >&2
            exit 2
        }
        CASES="$2"
        shift 2
        ;;
    --port)
        [[ $# -ge 2 ]] || {
            echo "--port requires a value" >&2
            exit 2
        }
        PORT="$2"
        shift 2
        ;;
    --prompt)
        [[ $# -ge 2 ]] || {
            echo "--prompt requires a value" >&2
            exit 2
        }
        PROMPT="$2"
        shift 2
        ;;
    --seed)
        [[ $# -ge 2 ]] || {
            echo "--seed requires a value" >&2
            exit 2
        }
        SEED="$2"
        shift 2
        ;;
    --num-cache-pages)
        [[ $# -ge 2 ]] || {
            echo "--num-cache-pages requires a value" >&2
            exit 2
        }
        NUM_CACHE_PAGES="$2"
        shift 2
        ;;
    --max-requests)
        [[ $# -ge 2 ]] || {
            echo "--max-requests requires a value" >&2
            exit 2
        }
        MAX_REQUESTS="$2"
        shift 2
        ;;
    --max-tokens)
        [[ $# -ge 2 ]] || {
            echo "--max-tokens requires a value" >&2
            exit 2
        }
        MAX_TOKENS="$2"
        shift 2
        ;;
    --max-tokens-per-request)
        [[ $# -ge 2 ]] || {
            echo "--max-tokens-per-request requires a value" >&2
            exit 2
        }
        MAX_TOKENS_PER_REQUEST="$2"
        shift 2
        ;;
    --model-root)
        [[ $# -ge 2 ]] || {
            echo "--model-root requires a value" >&2
            exit 2
        }
        MODEL_ROOT="$2"
        shift 2
        ;;
    --tokenizer)
        [[ $# -ge 2 ]] || {
            echo "--tokenizer requires a value" >&2
            exit 2
        }
        TOKENIZER="$2"
        shift 2
        ;;
    --model-27b)
        [[ $# -ge 2 ]] || {
            echo "--model-27b requires a value" >&2
            exit 2
        }
        MODEL_27B="$2"
        shift 2
        ;;
    --mtp-27b)
        [[ $# -ge 2 ]] || {
            echo "--mtp-27b requires a value" >&2
            exit 2
        }
        MTP_27B="$2"
        shift 2
        ;;
    --dspark-27b)
        [[ $# -ge 2 ]] || {
            echo "--dspark-27b requires a value" >&2
            exit 2
        }
        DSPARK_27B="$2"
        shift 2
        ;;
    --model-35b)
        [[ $# -ge 2 ]] || {
            echo "--model-35b requires a value" >&2
            exit 2
        }
        MODEL_35B="$2"
        shift 2
        ;;
    --mtp-35b)
        [[ $# -ge 2 ]] || {
            echo "--mtp-35b requires a value" >&2
            exit 2
        }
        MTP_35B="$2"
        shift 2
        ;;
    --dspark-35b)
        [[ $# -ge 2 ]] || {
            echo "--dspark-35b requires a value" >&2
            exit 2
        }
        DSPARK_35B="$2"
        shift 2
        ;;
    --model-27b-repo)
        [[ $# -ge 2 ]] || {
            echo "--model-27b-repo requires a value" >&2
            exit 2
        }
        MODEL_27B_REPO="$2"
        shift 2
        ;;
    --mtp-27b-repo)
        [[ $# -ge 2 ]] || {
            echo "--mtp-27b-repo requires a value" >&2
            exit 2
        }
        MTP_27B_REPO="$2"
        shift 2
        ;;
    --dspark-27b-repo)
        [[ $# -ge 2 ]] || {
            echo "--dspark-27b-repo requires a value" >&2
            exit 2
        }
        DSPARK_27B_REPO="$2"
        shift 2
        ;;
    --model-35b-repo)
        [[ $# -ge 2 ]] || {
            echo "--model-35b-repo requires a value" >&2
            exit 2
        }
        MODEL_35B_REPO="$2"
        shift 2
        ;;
    --mtp-35b-repo)
        [[ $# -ge 2 ]] || {
            echo "--mtp-35b-repo requires a value" >&2
            exit 2
        }
        MTP_35B_REPO="$2"
        shift 2
        ;;
    --dspark-35b-repo)
        [[ $# -ge 2 ]] || {
            echo "--dspark-35b-repo requires a value" >&2
            exit 2
        }
        DSPARK_35B_REPO="$2"
        shift 2
        ;;
    --no-build)
        BUILD=0
        shift
        ;;
    --no-baseline)
        BASELINE=0
        shift
        ;;
    --case-cooldown-secs)
        [[ $# -ge 2 ]] || {
            echo "--case-cooldown-secs requires a value" >&2
            exit 2
        }
        CASE_COOLDOWN_SECS="$2"
        shift 2
        ;;
    --logging)
        [[ $# -ge 2 ]] || {
            echo "--logging requires a value" >&2
            exit 2
        }
        LOGGING="$2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

require_positive_integer() {
    local option="$1"
    local value="$2"
    case "$value" in
    "" | *[!0-9]* | 0)
        echo "$option expects a positive integer" >&2
        exit 2
        ;;
    esac
}

require_nonnegative_integer() {
    local option="$1"
    local value="$2"
    case "$value" in
    "" | *[!0-9]*)
        echo "$option expects a non-negative integer" >&2
        exit 2
        ;;
    esac
}

require_positive_integer "--runs" "$RUNS"
require_positive_integer "--port" "$PORT"
require_positive_integer "--num-cache-pages" "$NUM_CACHE_PAGES"
require_positive_integer "--max-requests" "$MAX_REQUESTS"
require_positive_integer "--max-tokens" "$MAX_TOKENS"
require_positive_integer "--max-tokens-per-request" "$MAX_TOKENS_PER_REQUEST"
require_nonnegative_integer "--case-cooldown-secs" "$CASE_COOLDOWN_SECS"
require_nonnegative_integer "--seed" "$SEED"

case "$LOGGING" in
info | debug) ;;
*)
    echo "--logging must be info or debug" >&2
    exit 2
    ;;
esac

if ((PORT > 65535)); then
    echo "--port must be at most 65535" >&2
    exit 2
fi
if ((MAX_TOKENS_PER_REQUEST > MAX_TOKENS)); then
    echo "--max-tokens-per-request must not exceed --max-tokens" >&2
    exit 2
fi

[[ -n "$MODEL_27B" ]] || MODEL_27B="$MODEL_ROOT/Qwen3.8-27B-4bit"
[[ -n "$MTP_27B" ]] || MTP_27B="$MODEL_ROOT/Qwen3.8-27B-MTP-4bit"
[[ -n "$DSPARK_27B" ]] || DSPARK_27B="$MODEL_ROOT/Qwen3.8-27B-DSpark-affine"
[[ -n "$MODEL_35B" ]] || MODEL_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-4bit"
[[ -n "$MTP_35B" ]] || MTP_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-MTP-4bit"
[[ -n "$DSPARK_35B" ]] || DSPARK_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-DSpark-affine"

IFS=, read -r -a selected_cases <<<"$CASES"
requested_cases=("${selected_cases[@]}")
selected_cases=()

append_case() {
    local candidate="$1"
    local existing
    if ((${#selected_cases[@]})); then
        for existing in "${selected_cases[@]}"; do
            if [[ "$existing" == "$candidate" ]]; then
                return
            fi
        done
    fi
    selected_cases+=("$candidate")
}

for case_name in "${requested_cases[@]}"; do
    case "$case_name" in
    27b_on)
        append_case 27b_mtp
        append_case 27b_dspark
        ;;
    35b_on)
        append_case 35b_mtp
        append_case 35b_dspark
        ;;
    *) append_case "$case_name" ;;
    esac
done

need_27b=0
need_27b_mtp=0
need_27b_dspark=0
need_35b=0
need_35b_mtp=0
need_35b_dspark=0
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_off) need_27b=1 ;;
    27b_mtp)
        need_27b=1
        need_27b_mtp=1
        ;;
    27b_dspark)
        need_27b=1
        need_27b_dspark=1
        ;;
    35b_off) need_35b=1 ;;
    35b_mtp)
        need_35b=1
        need_35b_mtp=1
        ;;
    35b_dspark)
        need_35b=1
        need_35b_dspark=1
        ;;
    *)
        echo "unknown case: $case_name" >&2
        exit 2
        ;;
    esac
done
CASES="$(
    IFS=,
    printf '%s' "${selected_cases[*]}"
)"

if [[ ${#selected_cases[@]} -eq 0 ]]; then
    echo "--cases must include at least one case" >&2
    exit 2
fi

require_dir() {
    local option="$1"
    local dir="$2"
    if [[ -z "$dir" || ! -d "$dir" ]]; then
        echo "$option must name an existing directory" >&2
        exit 2
    fi
}

validate_affine_dspark() {
    local option="$1"
    local dir="$2"
    DSPARK_DIR="$dir" DSPARK_OPTION="$option" python3 - <<'PY'
import json
import os
from pathlib import Path

option = os.environ["DSPARK_OPTION"]
path = Path(os.environ["DSPARK_DIR"]) / "config.json"
try:
    config = json.loads(path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"unable to read {path}: {exc}") from exc

quantization = config.get("quantization_config") or config.get("quantization")
if not isinstance(quantization, dict):
    raise SystemExit(f"{option} must name an affine-quantized DSpark checkpoint")

block_size = config.get("block_size")
if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 2:
    raise SystemExit(
        f"{path} must have block_size >= 2 for the DSpark 1/2 benchmark matrix"
    )
PY
}

model_present() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    find "$dir" -type f \
        \( -name '*.safetensors' -o -name '*.npz' \) \
        -print -quit 2>/dev/null | grep -q .
}

ensure_model() {
    local repo="$1"
    local dir="$2"
    local -a download_command

    if model_present "$dir"; then
        echo "==> Found checkpoint: $dir"
        return
    fi

    if [[ -z "$repo" ]]; then
        echo "checkpoint is missing: $dir" >&2
        echo "Pass the matching checkpoint directory or configure its --*-repo option." >&2
        exit 1
    fi

    if command -v hf >/dev/null 2>&1; then
        download_command=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        download_command=(huggingface-cli download)
    else
        echo "checkpoint is missing: $dir" >&2
        echo 'Install the Hugging Face CLI with: python3 -m pip install -U "huggingface_hub[hf_xet]"' >&2
        exit 1
    fi

    echo "==> Downloading $repo -> $dir"
    mkdir -p "$dir"
    "${download_command[@]}" "$repo" --local-dir "$dir"

    if ! model_present "$dir"; then
        echo "download completed, but checkpoint is incomplete: $dir" >&2
        exit 1
    fi
}

ensure_optional_model() {
    local case_name="$1"
    local option="$2"
    local repo="$3"
    local dir="$4"

    if model_present "$dir"; then
        echo "==> Found checkpoint: $dir"
        return 0
    fi
    if [[ -z "$repo" ]]; then
        echo "WARNING: skipping $case_name because its checkpoint is missing: $dir" >&2
        echo "Pass $option DIR or ${option}-repo REPO to enable this case." >&2
        return 1
    fi
    ensure_model "$repo" "$dir"
}

if ((need_27b_mtp)); then
    ensure_optional_model 27b_mtp --mtp-27b "$MTP_27B_REPO" "$MTP_27B" || need_27b_mtp=0
fi
if ((need_27b_dspark)); then
    ensure_optional_model 27b_dspark --dspark-27b "$DSPARK_27B_REPO" "$DSPARK_27B" || need_27b_dspark=0
fi
if ((need_35b_mtp)); then
    ensure_optional_model 35b_mtp --mtp-35b "$MTP_35B_REPO" "$MTP_35B" || need_35b_mtp=0
fi
if ((need_35b_dspark)); then
    ensure_optional_model 35b_dspark --dspark-35b "$DSPARK_35B_REPO" "$DSPARK_35B" || need_35b_dspark=0
fi

runnable_cases=()
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_mtp) ((need_27b_mtp)) || continue ;;
    27b_dspark) ((need_27b_dspark)) || continue ;;
    35b_mtp) ((need_35b_mtp)) || continue ;;
    35b_dspark) ((need_35b_dspark)) || continue ;;
    esac
    runnable_cases+=("$case_name")
done
if ((${#runnable_cases[@]})); then
    selected_cases=("${runnable_cases[@]}")
else
    selected_cases=()
fi
if ((${#selected_cases[@]} == 0)); then
    echo "WARNING: no runnable cases remain after checkpoint discovery; exiting." >&2
    exit 0
fi
CASES="$(
    IFS=,
    printf '%s' "${selected_cases[*]}"
)"

need_27b=0
need_35b=0
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_*) need_27b=1 ;;
    35b_*) need_35b=1 ;;
    esac
done

if ((need_27b)); then
    ensure_model "$MODEL_27B_REPO" "$MODEL_27B"
fi
if ((need_35b)); then
    ensure_model "$MODEL_35B_REPO" "$MODEL_35B"
fi

if [[ -n "$TOKENIZER" ]]; then
    require_dir "--tokenizer" "$TOKENIZER"
fi
if ((need_27b)); then
    require_dir "--model-27b" "$MODEL_27B"
fi
if ((need_27b_mtp)); then
    require_dir "--mtp-27b" "$MTP_27B"
fi
if ((need_27b_dspark)); then
    require_dir "--dspark-27b" "$DSPARK_27B"
fi
if ((need_35b)); then
    require_dir "--model-35b" "$MODEL_35B"
fi
if ((need_35b_mtp)); then
    require_dir "--mtp-35b" "$MTP_35B"
fi
if ((need_35b_dspark)); then
    require_dir "--dspark-35b" "$DSPARK_35B"
fi
if ((need_27b_dspark)); then
    validate_affine_dspark "--dspark-27b" "$DSPARK_27B"
fi
if ((need_35b_dspark)); then
    validate_affine_dspark "--dspark-35b" "$DSPARK_35B"
fi

current_machine_id() {
    local display_info chipset_model gpu_cores normalized_chipset
    display_info="$(system_profiler SPDisplaysDataType 2>/dev/null || true)"
    chipset_model="$(printf '%s\n' "$display_info" | awk -F ': ' '/Chipset Model:/{print $2; exit}')"
    gpu_cores="$(printf '%s\n' "$display_info" | awk -F ': ' '/Total Number of Cores:/{print $2; exit}')"
    if [[ -z "$chipset_model" || -z "$gpu_cores" ]]; then
        echo "unknown"
        return
    fi
    normalized_chipset="$(printf '%s' "$chipset_model" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_')"
    echo "${normalized_chipset}_${gpu_cores}_gpu_cores"
}

baseline_config_mismatches() {
    local current_machine="$1"
    local current_os="$2"
    local current_arch="$3"
    local prompt_sha256
    local mismatches=""
    prompt_sha256="$(printf '%s' "$PROMPT" | shasum -a 256 | awk '{print $1}')"

    [[ "$current_machine" == "$BASELINE_MACHINE" ]] || mismatches="machine"
    [[ "$current_os" == "$BASELINE_OS_VERSION" ]] || mismatches="${mismatches:+$mismatches,}os"
    [[ "$current_arch" == "$BASELINE_ARCH" ]] || mismatches="${mismatches:+$mismatches,}arch"
    [[ "$NUM_CACHE_PAGES" == "$BASELINE_NUM_CACHE_PAGES" ]] || mismatches="${mismatches:+$mismatches,}num_cache_pages"
    [[ "$CACHE_BLOCK_TOKENS" == "$BASELINE_CACHE_BLOCK_TOKENS" ]] || mismatches="${mismatches:+$mismatches,}cache_block_tokens"
    [[ "$MAX_REQUESTS" == "$BASELINE_MAX_REQUESTS" ]] || mismatches="${mismatches:+$mismatches,}max_requests"
    [[ "$MAX_TOKENS" == "$BASELINE_MAX_TOKENS" ]] || mismatches="${mismatches:+$mismatches,}max_tokens"
    [[ "$MAX_TOKENS_PER_REQUEST" == "$BASELINE_MAX_TOKENS_PER_REQUEST" ]] || mismatches="${mismatches:+$mismatches,}max_tokens_per_request"
    [[ "$CASE_COOLDOWN_SECS" == "$BASELINE_CASE_COOLDOWN_SECS" ]] || mismatches="${mismatches:+$mismatches,}case_cooldown_secs"
    [[ "$LOGGING" == "$BASELINE_LOGGING" ]] || mismatches="${mismatches:+$mismatches,}logging"
    [[ "$SEED" == "$BASELINE_SEED" ]] || mismatches="${mismatches:+$mismatches,}seed"
    [[ "$prompt_sha256" == "$BASELINE_PROMPT_SHA256" ]] || mismatches="${mismatches:+$mismatches,}prompt"
    if [[ -n "$TOKENIZER" ]]; then
        [[ "$(basename "$TOKENIZER")" == "$BASELINE_TOKENIZER_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}tokenizer"
    elif ((need_35b)); then
        [[ "$(basename "$MODEL_35B")" == "$BASELINE_TOKENIZER_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}tokenizer"
    else
        [[ "$(basename "$MODEL_27B")" == "$BASELINE_TOKENIZER_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}tokenizer"
    fi
    if ((need_27b)); then
        [[ "$(basename "$MODEL_27B")" == "$BASELINE_MODEL_27B_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}model_27b"
    fi
    if ((need_27b_mtp)); then
        [[ "$(basename "$MTP_27B")" == "$BASELINE_MTP_27B_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}mtp_27b"
    fi
    if ((need_35b)); then
        [[ "$(basename "$MODEL_35B")" == "$BASELINE_MODEL_35B_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}model_35b"
    fi
    if ((need_35b_mtp)); then
        [[ "$(basename "$MTP_35B")" == "$BASELINE_MTP_35B_DIR_NAME" ]] || mismatches="${mismatches:+$mismatches,}mtp_35b"
    fi
    echo "$mismatches"
}

baseline_metrics() {
    # Medians from a controlled M3 Max run on BASELINE_DATE at BASELINE_COMMIT.
    # The run used three samples per token count and the exact workload recorded
    # by the BASELINE_* constants above. The machine is part of the lookup key
    # so another hardware baseline can be added without replacing these values.
    # Fields: decode_tok_s, ttft_ms, inter_chunk_p50_ms, inter_chunk_p95_ms.
    case "$1:$2:$3" in
    apple_m3_max_40_gpu_cores:27b_off:256) echo "22.561,335.410,44.383,45.088" ;;
    apple_m3_max_40_gpu_cores:27b_off:384) echo "22.490,339.124,44.820,45.101" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:256) echo "38.826,338.567,49.179,49.851" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:384) echo "36.921,358.611,51.069,51.663" ;;
    apple_m3_max_40_gpu_cores:35b_off:256) echo "95.597,77.732,10.509,10.910" ;;
    apple_m3_max_40_gpu_cores:35b_off:1024) echo "92.860,74.288,10.822,10.948" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:256) echo "147.397,78.127,13.030,13.383" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:1024) echo "129.709,75.905,13.585,14.221" ;;
    *) return 1 ;;
    esac
}

baseline_trajectory() {
    # Fields: input tokens, sampled tokens, chunks, proposed speculative tokens, accepted speculative tokens.
    case "$1:$2:$3" in
    apple_m3_max_40_gpu_cores:27b_off:256) echo "34,256,256,0,0" ;;
    apple_m3_max_40_gpu_cores:27b_off:384) echo "34,384,384,0,0" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:256) echo "34,256,135,134,122" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:384) echo "34,384,205,204,180" ;;
    apple_m3_max_40_gpu_cores:35b_off:256) echo "34,256,256,0,0" ;;
    apple_m3_max_40_gpu_cores:35b_off:1024) echo "34,1024,1024,0,0" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:256) echo "34,256,134,133,123" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:1024) echo "34,1024,583,582,442" ;;
    *) return 1 ;;
    esac
}

if [[ "$BUILD" -eq 1 ]]; then
    cargo build --release -p inference-runtime-service --bin qwen3_5_dense --bin qwen3_5_sparse --bin decode
fi

if pgrep -fl "qwen3_5|decode|inference-runtime-service|cargo bench|cargo run" >/dev/null 2>&1; then
    echo "refusing to run while another qwen/decode/cargo perf process is active:" >&2
    pgrep -fl "qwen3_5|decode|inference-runtime-service|cargo bench|cargo run" >&2 || true
    exit 1
fi

wait_for_port() {
    for _ in $(seq 1 240); do
        if nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_decode() {
    local label="$1"
    local tokens="$2"
    local run="$3"
    local server_log="$4"
    local tokenizer="$5"
    local out="/tmp/psi_dec_${label}_${tokens}_${run}.out"
    local server_log_offset
    server_log_offset="$(wc -c <"$server_log")"
    if ! target/release/decode \
        --server-url "http://127.0.0.1:${PORT}" \
        --max-sampled-tokens "$tokens" \
        --seed "$SEED" \
        --chat-template auto \
        --show-stats \
        --hf-model-dir "$tokenizer" \
        --prompt-str "$PROMPT" >"$out" 2>&1; then
        echo "DECODE_FAILED label=$label max_new=$tokens run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
    local json
    json=$(grep "^{" "$out" | tail -n 1 || true)
    if [[ -z "$json" ]]; then
        echo "DECODE_STATS_MISSING label=$label max_new=$tokens run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
    if ! JSON_LINE="$json" SERVER_LOG="$server_log" SERVER_LOG_OFFSET="$server_log_offset" python3 - <<'PY'; then
import json
import os
import re

j = json.loads(os.environ["JSON_LINE"])
with open(os.environ["SERVER_LOG"], "rb") as f:
    f.seek(int(os.environ["SERVER_LOG_OFFSET"]))
    server_log = f.read().decode("utf-8", errors="replace")
server_log = re.sub(r"\x1b\[[0-9;]*m", "", server_log)
proposed = 0
accepted = 0
for line in server_log.splitlines():
    if 'phase="executor.batch.perf"' not in line:
        continue
    proposed_match = re.search(r"num_spec_tokens=(\d+)", line)
    accepted_match = re.search(r"num_accepted_tokens=(\d+)", line)
    if proposed_match and accepted_match:
        proposed += int(proposed_match.group(1))
        accepted += int(accepted_match.group(1))
acceptance_rate = accepted / proposed if proposed else 0.0
tokens_per_chunk = j["sampled_tokens"] / j["chunk_count"]
print("{:.6f},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.6f},{:.6f}".format(
    j["decode_tokens_per_s"],
    j["chunk_count"],
    j["sampled_tokens"],
    j["input_tokens"],
    j["ttft_ms"],
    j["prompt_tokens_per_s"],
    j["inter_chunk_p50_ms"],
    j["inter_chunk_p95_ms"],
    proposed,
    accepted,
    acceptance_rate,
    tokens_per_chunk,
))
PY
        echo "DECODE_STATS_INVALID label=$label max_new=$tokens run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
}

run_server_case() {
    local label="$1"
    local token_list="$2"
    local tokenizer="$3"
    shift 3
    local log="/tmp/psi_dec_${label}.log"

    local server_command=("$@")
    server_command+=(--logging "$LOGGING")
    "${server_command[@]}" >"$log" 2>&1 &
    ACTIVE_SERVER_PID=$!

    if ! wait_for_port; then
        echo "SERVER_START_FAILED $label" >&2
        tail -n 80 "$log" >&2 || true
        cleanup_active_server
        exit 1
    fi

    for tokens in $token_list; do
        local vals=""
        local inputs=""
        local chunks=""
        local samples=""
        local ttfts=""
        local prompt_rates=""
        local inter_chunk_p50s=""
        local inter_chunk_p95s=""
        local proposed_specs=""
        local accepted_specs=""
        local acceptance_rates=""
        local tokens_per_chunks=""
        for run in $(seq 1 "$RUNS"); do
            local parsed tokps chunk sampled input_tokens ttft prompt_rate
            local inter_chunk_p50 inter_chunk_p95 proposed_spec accepted_spec
            local acceptance_rate tokens_per_chunk
            parsed=$(run_decode "$label" "$tokens" "$run" "$log" "$tokenizer")
            IFS=, read -r \
                tokps chunk sampled input_tokens ttft prompt_rate \
                inter_chunk_p50 inter_chunk_p95 proposed_spec accepted_spec \
                acceptance_rate tokens_per_chunk <<<"$parsed"
            vals="$vals $tokps"
            inputs="$inputs $input_tokens"
            chunks="$chunks $chunk"
            samples="$samples $sampled"
            ttfts="$ttfts $ttft"
            prompt_rates="$prompt_rates $prompt_rate"
            inter_chunk_p50s="$inter_chunk_p50s $inter_chunk_p50"
            inter_chunk_p95s="$inter_chunk_p95s $inter_chunk_p95"
            proposed_specs="$proposed_specs $proposed_spec"
            accepted_specs="$accepted_specs $accepted_spec"
            acceptance_rates="$acceptance_rates $acceptance_rate"
            tokens_per_chunks="$tokens_per_chunks $tokens_per_chunk"
            echo "RUN label=$label max_new=$tokens run=$run" \
                "input_tokens=$input_tokens sampled=$sampled chunks=$chunk" \
                "proposed_spec=$proposed_spec accepted_spec=$accepted_spec" \
                "acceptance_rate=$acceptance_rate tokens_per_chunk=$tokens_per_chunk" \
                "decode_tok_s=$tokps ttft_ms=$ttft prompt_tok_s=$prompt_rate" \
                "inter_chunk_p50_ms=$inter_chunk_p50 inter_chunk_p95_ms=$inter_chunk_p95"
        done

        local baseline_decode=""
        local baseline_ttft=""
        local baseline_inter_chunk_p50=""
        local baseline_inter_chunk_p95=""
        local baseline_input_tokens=""
        local baseline_sampled=""
        local baseline_chunks=""
        local baseline_proposed_spec=""
        local baseline_accepted_spec=""
        local baseline_status="disabled"
        local baseline_mismatch=""
        if [[ "$BASELINE" -eq 1 ]]; then
            IFS=, read -r \
                baseline_decode baseline_ttft baseline_inter_chunk_p50 baseline_inter_chunk_p95 \
                <<<"$(baseline_metrics "$MACHINE" "$label" "$tokens" || true)"
            IFS=, read -r \
                baseline_input_tokens baseline_sampled baseline_chunks \
                baseline_proposed_spec baseline_accepted_spec \
                <<<"$(baseline_trajectory "$MACHINE" "$label" "$tokens" || true)"
            if [[ -z "$baseline_decode" ]]; then
                baseline_status="no-hardware-baseline"
                baseline_mismatch="machine"
            elif [[ -n "$BASELINE_CONFIG_MISMATCHES" ]]; then
                baseline_status="config-mismatch"
                baseline_mismatch="$BASELINE_CONFIG_MISMATCHES"
            elif ((RUNS < BASELINE_MIN_RUNS)); then
                baseline_status="insufficient-runs"
                baseline_mismatch="runs"
            else
                baseline_status="comparable"
            fi
        fi
        VALS="$vals" \
            INPUTS="$inputs" \
            CHUNKS="$chunks" \
            SAMPLES="$samples" \
            TTFTS="$ttfts" \
            PROMPT_RATES="$prompt_rates" \
            INTER_CHUNK_P50S="$inter_chunk_p50s" \
            INTER_CHUNK_P95S="$inter_chunk_p95s" \
            PROPOSED_SPECS="$proposed_specs" \
            ACCEPTED_SPECS="$accepted_specs" \
            ACCEPTANCE_RATES="$acceptance_rates" \
            TOKENS_PER_CHUNKS="$tokens_per_chunks" \
            LABEL="$label" \
            TOKENS="$tokens" \
            BASELINE_DECODE="$baseline_decode" \
            BASELINE_TTFT="$baseline_ttft" \
            BASELINE_INTER_CHUNK_P50="$baseline_inter_chunk_p50" \
            BASELINE_INTER_CHUNK_P95="$baseline_inter_chunk_p95" \
            BASELINE_INPUT_TOKENS="$baseline_input_tokens" \
            BASELINE_SAMPLED="$baseline_sampled" \
            BASELINE_CHUNKS="$baseline_chunks" \
            BASELINE_PROPOSED_SPEC="$baseline_proposed_spec" \
            BASELINE_ACCEPTED_SPEC="$baseline_accepted_spec" \
            BASELINE_STATUS="$baseline_status" \
            BASELINE_MISMATCH="$baseline_mismatch" \
            python3 - <<'PY'
import os
import statistics

vals = [float(x) for x in os.environ["VALS"].split()]
inputs = os.environ["INPUTS"].split()
chunks = os.environ["CHUNKS"].split()
samples = os.environ["SAMPLES"].split()
ttfts = [float(x) for x in os.environ["TTFTS"].split()]
prompt_rates = [float(x) for x in os.environ["PROMPT_RATES"].split()]
inter_chunk_p50s = [float(x) for x in os.environ["INTER_CHUNK_P50S"].split()]
inter_chunk_p95s = [float(x) for x in os.environ["INTER_CHUNK_P95S"].split()]
proposed_specs = os.environ["PROPOSED_SPECS"].split()
accepted_specs = os.environ["ACCEPTED_SPECS"].split()
acceptance_rates = [float(x) for x in os.environ["ACCEPTANCE_RATES"].split()]
tokens_per_chunks = [float(x) for x in os.environ["TOKENS_PER_CHUNKS"].split()]
median_decode = statistics.median(vals)
median_ttft = statistics.median(ttfts)
median_prompt_rate = statistics.median(prompt_rates)
median_inter_chunk_p50 = statistics.median(inter_chunk_p50s)
median_inter_chunk_p95 = statistics.median(inter_chunk_p95s)
median_acceptance_rate = statistics.median(acceptance_rates)
median_tokens_per_chunk = statistics.median(tokens_per_chunks)
acceptance_rate_text = (
    "{:.6f}".format(median_acceptance_rate)
    if any(int(proposed) > 0 for proposed in proposed_specs)
    else "na"
)
prefix = (
    "SUMMARY label={} max_new={} median_decode_tok_s={:.3f} median_ttft_ms={:.3f} "
    "median_prompt_tok_s={:.3f} median_inter_chunk_p50_ms={:.3f} "
    "median_inter_chunk_p95_ms={:.3f} median_tokens_per_chunk={:.3f} "
    "median_acceptance_rate={}"
).format(
    os.environ["LABEL"],
    os.environ["TOKENS"],
    median_decode,
    median_ttft,
    median_prompt_rate,
    median_inter_chunk_p50,
    median_inter_chunk_p95,
    median_tokens_per_chunk,
    acceptance_rate_text,
)
baseline_decode = os.environ.get("BASELINE_DECODE", "")
baseline_status = os.environ["BASELINE_STATUS"]
baseline_input_tokens = os.environ.get("BASELINE_INPUT_TOKENS", "")
baseline_sampled = os.environ.get("BASELINE_SAMPLED", "")
baseline_chunks = os.environ.get("BASELINE_CHUNKS", "")
baseline_proposed_spec = os.environ.get("BASELINE_PROPOSED_SPEC", "")
baseline_accepted_spec = os.environ.get("BASELINE_ACCEPTED_SPEC", "")
if baseline_status == "comparable" and (
    any(input_tokens != baseline_input_tokens for input_tokens in inputs)
    or any(sample != baseline_sampled for sample in samples)
    or any(chunk != baseline_chunks for chunk in chunks)
    or any(proposed != baseline_proposed_spec for proposed in proposed_specs)
    or any(accepted != baseline_accepted_spec for accepted in accepted_specs)
):
    baseline_status = "trajectory-mismatch"
    baseline_mismatch = "trajectory"
else:
    baseline_mismatch = os.environ.get("BASELINE_MISMATCH", "")
if baseline_decode:
    def optional_float(name):
        value = os.environ.get(name, "")
        return None if value in ("", "na") else float(value)

    baseline_decode_value = float(baseline_decode)
    baseline_ttft = float(os.environ["BASELINE_TTFT"])
    baseline_inter_chunk_p50 = optional_float("BASELINE_INTER_CHUNK_P50")
    baseline_inter_chunk_p95 = optional_float("BASELINE_INTER_CHUNK_P95")
    prefix += (
        " baseline_decode_tok_s={:.3f} baseline_ttft_ms={:.3f} "
        "baseline_inter_chunk_p50_ms={} baseline_inter_chunk_p95_ms={}"
    ).format(
        baseline_decode_value,
        baseline_ttft,
        "{:.3f}".format(baseline_inter_chunk_p50) if baseline_inter_chunk_p50 is not None else "na",
        "{:.3f}".format(baseline_inter_chunk_p95) if baseline_inter_chunk_p95 is not None else "na",
    )
    if baseline_status == "comparable":
        decode_delta = 100.0 * (median_decode - baseline_decode_value) / baseline_decode_value
        ttft_delta = 100.0 * (median_ttft - baseline_ttft) / baseline_ttft
        inter_chunk_p95_delta = (
            100.0 * (median_inter_chunk_p95 - baseline_inter_chunk_p95) / baseline_inter_chunk_p95
            if baseline_inter_chunk_p95 is not None
            else None
        )
        prefix += (
            " decode_delta_pct={:+.2f} ttft_delta_pct={:+.2f} "
            "inter_chunk_p95_delta_pct={}"
        ).format(
            decode_delta,
            ttft_delta,
            "{:+.2f}".format(inter_chunk_p95_delta) if inter_chunk_p95_delta is not None else "na",
        )
prefix += " baseline_status={}".format(baseline_status)
if baseline_mismatch:
    prefix += " baseline_mismatch={}".format(baseline_mismatch)
print(
    "{} min_decode_tok_s={:.3f} max_decode_tok_s={:.3f} runs={} input_tokens={} samples={} chunks={} proposed_spec={} accepted_spec={}".format(
        prefix,
        min(vals),
        max(vals),
        ",".join("{:.3f}".format(v) for v in vals),
        ",".join(inputs),
        ",".join(samples),
        ",".join(chunks),
        ",".join(proposed_specs),
        ",".join(accepted_specs),
    )
)
PY
    done

    cleanup_active_server
}

run_spec_case_sequence() {
    local model_label="$1"
    local spec_mode="$2"
    local token_list="$3"
    local server_binary="$4"
    local model_dir="$5"
    local spec_model_dir="$6"
    local tokenizer="${TOKENIZER:-$model_dir}"
    local spec_model_option

    case "$spec_mode" in
    mtp) spec_model_option="--hf-mtp-model-dir" ;;
    dspark) spec_model_option="--hf-dspark-model-dir" ;;
    *)
        echo "unsupported speculative mode: $spec_mode" >&2
        exit 2
        ;;
    esac

    for num_spec_tokens in 1 2; do
        local label="${model_label}_${spec_mode}${num_spec_tokens}"
        if ((num_spec_tokens > 1 && CASE_COOLDOWN_SECS > 0)); then
            echo "COOLDOWN before=$label seconds=$CASE_COOLDOWN_SECS"
            sleep "$CASE_COOLDOWN_SECS"
        fi
        run_server_case "$label" "$token_list" "$tokenizer" "$server_binary" \
            --grpc-listen-addr "127.0.0.1:${PORT}" \
            --hf-model-dir "$model_dir" \
            "$spec_model_option" "$spec_model_dir" \
            --num-spec-tokens "$num_spec_tokens" \
            --num-cache-pages "$NUM_CACHE_PAGES" \
            --max-requests "$MAX_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
    done
}

run_named_case() {
    case "$1" in
    27b_off)
        run_server_case 27b_off "256 384" "${TOKENIZER:-$MODEL_27B}" target/release/qwen3_5_dense \
            --grpc-listen-addr "127.0.0.1:${PORT}" \
            --hf-model-dir "$MODEL_27B" \
            --num-cache-pages "$NUM_CACHE_PAGES" \
            --max-requests "$MAX_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
        ;;
    27b_mtp)
        run_spec_case_sequence 27b mtp "256 384" target/release/qwen3_5_dense "$MODEL_27B" "$MTP_27B"
        ;;
    27b_dspark)
        run_spec_case_sequence 27b dspark "256 384" target/release/qwen3_5_dense "$MODEL_27B" "$DSPARK_27B"
        ;;
    35b_off)
        run_server_case 35b_off "256 1024" "${TOKENIZER:-$MODEL_35B}" target/release/qwen3_5_sparse \
            --grpc-listen-addr "127.0.0.1:${PORT}" \
            --hf-model-dir "$MODEL_35B" \
            --num-cache-pages "$NUM_CACHE_PAGES" \
            --max-requests "$MAX_REQUESTS" \
            --max-tokens "$MAX_TOKENS" \
            --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
        ;;
    35b_mtp)
        run_spec_case_sequence 35b mtp "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$MTP_35B"
        ;;
    35b_dspark)
        run_spec_case_sequence 35b dspark "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$DSPARK_35B"
        ;;
    *)
        echo "unknown case: $1" >&2
        exit 2
        ;;
    esac
}

GIT_COMMIT="$(git rev-parse --verify HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=unknown
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -n "$(git status --porcelain)" ]]; then
        GIT_DIRTY=1
    else
        GIT_DIRTY=0
    fi
fi
OS_VERSION="$(sw_vers -productVersion 2>/dev/null || uname -s)"
ARCH="$(uname -m)"
MACHINE="$(current_machine_id)"
BASELINE_CONFIG_MISMATCHES="$(baseline_config_mismatches "$MACHINE" "$OS_VERSION" "$ARCH")"
echo "CONFIG commit=$GIT_COMMIT dirty=$GIT_DIRTY machine=$MACHINE os=$OS_VERSION arch=$ARCH baseline_machine=$BASELINE_MACHINE baseline_date=$BASELINE_DATE baseline_commit=$BASELINE_COMMIT baseline_os=$BASELINE_OS_VERSION baseline_arch=$BASELINE_ARCH baseline_num_cache_pages=$BASELINE_NUM_CACHE_PAGES baseline_cache_block_tokens=$BASELINE_CACHE_BLOCK_TOKENS baseline_max_requests=$BASELINE_MAX_REQUESTS baseline_max_tokens=$BASELINE_MAX_TOKENS baseline_max_tokens_per_request=$BASELINE_MAX_TOKENS_PER_REQUEST baseline_case_cooldown_secs=$BASELINE_CASE_COOLDOWN_SECS baseline_logging=$BASELINE_LOGGING baseline_seed=$BASELINE_SEED baseline_min_runs=$BASELINE_MIN_RUNS baseline_config_mismatches=${BASELINE_CONFIG_MISMATCHES:-none} num_cache_pages=$NUM_CACHE_PAGES cache_block_tokens=$CACHE_BLOCK_TOKENS max_requests=$MAX_REQUESTS max_tokens=$MAX_TOKENS max_tokens_per_request=$MAX_TOKENS_PER_REQUEST num_spec_tokens=1,2 cases=$CASES case_cooldown_secs=$CASE_COOLDOWN_SECS logging=$LOGGING seed=$SEED prompt_chars=${#PROMPT} tokenizer=${TOKENIZER:-auto-per-model} model_27b=$MODEL_27B mtp_27b=$MTP_27B dspark_27b=$DSPARK_27B model_35b=$MODEL_35B mtp_35b=$MTP_35B dspark_35b=$DSPARK_35B"
for case_index in "${!selected_cases[@]}"; do
    case_name="${selected_cases[$case_index]}"
    if [[ "$case_index" -gt 0 && "$CASE_COOLDOWN_SECS" -gt 0 ]]; then
        echo "COOLDOWN before=$case_name seconds=$CASE_COOLDOWN_SECS"
        sleep "$CASE_COOLDOWN_SECS"
    fi
    run_named_case "$case_name"
done
