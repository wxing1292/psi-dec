#!/usr/bin/env bash
set -euo pipefail

RUNS=7
BLOCK_SPEC_TOKENS=""
PORT=50061
BUILD=1
REFERENCE=1
SHOW_RUNS=0
# One fixed GSM8K prompt and the original Beijing prompt.
PROMPT_SET="representative2"
PROMPT_IDS=(
    "gsm8k_typing_average"
    "beijing_travel"
)
PROMPTS=(
    "Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next time he tests his typing speed it has increased to 52 WPM. If he continues to increase his typing speed once more by 5 words, what will be the average of the three measurements?
Please reason step by step, and put your final answer within \boxed{}."
    "你好，北京有什么好玩的景点？香山如何？早上去晚上去？单纯爬山么？还有什么可以在香山玩的？"
)
MODEL_ROOT="$HOME/Workspace/models"
TOKENIZER=""
MODEL_27B=""
MTP_27B=""
DSPARK_27B=""
DFLASH2_27B=""
MODEL_35B=""
MTP_35B=""
DSPARK_35B=""
DFLASH2_35B=""
MODEL_27B_REPO="mlx-community/Qwen3.8-27B-4bit"
MTP_27B_REPO="mlx-community/Qwen3.8-27B-MTP-4bit"
DSPARK_27B_REPO=""
MODEL_35B_REPO="mlx-community/Qwen3.6-35B-A3B-4bit"
MTP_35B_REPO="mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
DSPARK_35B_REPO=""
CASES="27b_off,35b_off,27b_mtp1,35b_mtp1,27b_dspark,35b_dspark,27b_dflash2,35b_dflash2,27b_mtp2,35b_mtp2"
CASE_COOLDOWN_SECS=8
LOGGING=info
SEED=42
TEMPERATURE=0.7
TOP_K=20
TOP_P=0.8
NUM_CACHE_PAGES=294912
MAX_REQUESTS=2
MAX_TOKENS=128
MAX_TOKENS_PER_REQUEST=64
CACHE_BLOCK_TOKENS=2048
REFERENCE_MACHINE="apple_m3_max_40_gpu_cores"
REFERENCE_DATE="2026-08-27"
REFERENCE_COMMIT="fa5016c5ce49f4ff31dadf3464587bf4e91631a5"
REFERENCE_DIRTY=0
REFERENCE_OS_VERSION="27.0"
REFERENCE_ARCH="arm64"
REFERENCE_CACHE_BLOCK_TOKENS=2048
REFERENCE_MAX_TOKENS=128
REFERENCE_MAX_TOKENS_PER_REQUEST=64
REFERENCE_CASE_COOLDOWN_SECS=8
REFERENCE_LOGGING="info"
REFERENCE_SEED=42
REFERENCE_TEMPERATURE=0.7
REFERENCE_TOP_K=20
REFERENCE_TOP_P=0.8
REFERENCE_RUNS=3
REFERENCE_CASES="27b_off,35b_off,27b_mtp1,27b_dspark,27b_dflash2,27b_mtp2,35b_mtp1,35b_mtp2"
REFERENCE_PROMPT_SET="representative2"
REFERENCE_PROMPT_SET_SHA256="1e842b94e2f61518333df4682093921e5be3b2a8909a45157e7f7fc1fd27cffc"
REFERENCE_MODEL_27B_DIR_NAME="Qwen3.8-27B-4bit"
REFERENCE_MTP_27B_DIR_NAME="Qwen3.8-27B-MTP-4bit"
REFERENCE_DSPARK_27B_DIR_NAME="Qwen3.8-27B-DSpark-affine"
REFERENCE_DFLASH2_27B_DIR_NAME="Qwen3.8-27B-DFlash2-affine"
REFERENCE_MODEL_35B_DIR_NAME="Qwen3.6-35B-A3B-4bit"
REFERENCE_MTP_35B_DIR_NAME="Qwen3.6-35B-A3B-MTP-4bit"
ACTIVE_SERVER_PID=""
REPORT_FILE=""

cleanup_active_server() {
    if [[ -n "$ACTIVE_SERVER_PID" ]]; then
        kill "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        wait "$ACTIVE_SERVER_PID" >/dev/null 2>&1 || true
        ACTIVE_SERVER_PID=""
    fi
}

cleanup() {
    cleanup_active_server
    if [[ -n "$REPORT_FILE" ]]; then
        rm -f "$REPORT_FILE"
        REPORT_FILE=""
    fi
}

trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

usage() {
    cat <<'EOF'
Usage: scripts/qwen35_e2e_decode_perf.sh [options]

Runs Qwen3.5/3.6/3.8 replay e2e decode perf one server at a time and reports
decode throughput, TTFT, inter-chunk latency, and speculative acceptance.

Options:
  --runs N              Runs per prompt and token-count case. Default: 7
  --cases LIST          Comma-separated cases.
                        Default order:
                          27b_off,35b_off,
                          27b_mtp1,35b_mtp1,
                          27b_dspark,35b_dspark,
                          27b_dflash2,35b_dflash2,
                          27b_mtp2,35b_mtp2
                        Available modes: off, mtp, dspark, dflash2.
                        The *_mtp alias runs proposal counts 1 and 2.
                        DSpark and DFlash2 use checkpoint defaults unless overridden.
  --block-spec-tokens K  Generate K tokens in each DSpark/DFlash2 proposal.
                        This changes actual draft input rows, not only verification.
                        Does not change MTP cases. Default: checkpoint geometry.
                        Group cases: 27b_on and 35b_on select all Spec modes.
                        Missing Spec checkpoints are warned and skipped.
  --port N              Server port. Default: 50061
  --prompt TEXT         Use one custom prompt instead of the default representative2 set.
  --seed N              Fixed request seed. Default: 42
  --temperature N       Sampling temperature. Default: 0.7
  --top-k N             Sampling top-k. Default: 20
  --top-p N             Sampling top-p. Default: 0.8
  --num-cache-pages N   Shared cache pages. Default: 294912
  --max-requests N      Scheduler request capacity. Default: 2
  --max-tokens N        Scheduler flattened-token capacity. Default: 128
  --max-tokens-per-request N
                        Scheduler per-request token capacity. Default: 64
  --model-root DIR      Default checkpoint root. Default: $HOME/Workspace/models
  --tokenizer DIR       Tokenizer/chat-template override. Default: each case's Main model.
  --model-27b DIR       27B Main directory. Default: MODEL_ROOT/Qwen3.8-27B-4bit
  --mtp-27b DIR         27B MTP directory. Default: MODEL_ROOT/Qwen3.8-27B-MTP-4bit
  --dspark-27b DIR      27B DSpark directory. Default: MODEL_ROOT/Qwen3.8-27B-DSpark-affine
  --dflash2-27b DIR     27B affine DFlash2 directory. Default: MODEL_ROOT/Qwen3.8-27B-DFlash2-affine
  --model-35b DIR       35B Main directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-4bit
  --mtp-35b DIR         35B MTP directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-MTP-4bit
  --dspark-35b DIR      35B DSpark directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-DSpark-affine
  --dflash2-35b DIR     35B affine DFlash2 directory. Default: MODEL_ROOT/Qwen3.6-35B-A3B-DFlash2-affine
  --model-27b-repo REPO Hugging Face repo used if the 27B Main model is missing.
  --mtp-27b-repo REPO   Hugging Face repo used if the 27B MTP model is missing.
  --dspark-27b-repo REPO
                        Hugging Face repo used if the 27B affine DSpark model is missing.
  --model-35b-repo REPO Hugging Face repo used if the 35B Main model is missing.
  --mtp-35b-repo REPO   Hugging Face repo used if the 35B MTP model is missing.
  --dspark-35b-repo REPO
                        Hugging Face repo used if the 35B affine DSpark model is missing.
  --no-build            Skip release build.
  --no-reference        Do not compare summaries with the checked-in reference run.
  --show-runs           Print each machine-readable RUN and SUMMARY row.
  --case-cooldown-secs N
                        Idle time between model cases. Default: 8
                        Pass 0 for an intentional sustained-load run.
  --logging LEVEL       Server logging: info or debug. Default: info.
                        Debug adds request/response and replay-stage details.
                        The benchmark always enables the executor perf DEBUG target.
  -h, --help            Show this help.

Examples:
  scripts/qwen35_e2e_decode_perf.sh \
    --model-root "$HOME/Workspace/models" \
    --cases 27b_off,35b_off,27b_mtp1,35b_mtp1,27b_dspark,35b_dspark \
    --runs 3

  scripts/qwen35_e2e_decode_perf.sh \
    --model-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-4bit" \
    --mtp-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-MTP-4bit" \
    --dspark-35b "$HOME/Workspace/models/Qwen3.6-35B-A3B-DSpark-affine" \
    --cases 35b_mtp1,35b_dspark,35b_mtp2 --runs 3
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
    --block-spec-tokens)
        [[ $# -ge 2 ]] || {
            echo "--block-spec-tokens requires a value" >&2
            exit 2
        }
        BLOCK_SPEC_TOKENS="$2"
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
        PROMPT_SET="custom"
        PROMPT_IDS=("custom")
        PROMPTS=("$2")
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
    --temperature)
        [[ $# -ge 2 ]] || {
            echo "--temperature requires a value" >&2
            exit 2
        }
        TEMPERATURE="$2"
        shift 2
        ;;
    --top-k)
        [[ $# -ge 2 ]] || {
            echo "--top-k requires a value" >&2
            exit 2
        }
        TOP_K="$2"
        shift 2
        ;;
    --top-p)
        [[ $# -ge 2 ]] || {
            echo "--top-p requires a value" >&2
            exit 2
        }
        TOP_P="$2"
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
    --dflash2-27b)
        [[ $# -ge 2 ]] || {
            echo "--dflash2-27b requires a value" >&2
            exit 2
        }
        DFLASH2_27B="$2"
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
    --dflash2-35b)
        [[ $# -ge 2 ]] || {
            echo "--dflash2-35b requires a value" >&2
            exit 2
        }
        DFLASH2_35B="$2"
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
    --no-reference)
        REFERENCE=0
        shift
        ;;
    --show-runs)
        SHOW_RUNS=1
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
if [[ -n "$BLOCK_SPEC_TOKENS" ]]; then
    require_positive_integer "--block-spec-tokens" "$BLOCK_SPEC_TOKENS"
fi
require_positive_integer "--port" "$PORT"
require_positive_integer "--num-cache-pages" "$NUM_CACHE_PAGES"
require_positive_integer "--max-requests" "$MAX_REQUESTS"
require_positive_integer "--max-tokens" "$MAX_TOKENS"
require_positive_integer "--max-tokens-per-request" "$MAX_TOKENS_PER_REQUEST"
require_nonnegative_integer "--case-cooldown-secs" "$CASE_COOLDOWN_SECS"
require_nonnegative_integer "--seed" "$SEED"
require_positive_integer "--top-k" "$TOP_K"

if ! TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" python3 - <<'PY'
import math
import os

for name in ("TEMPERATURE", "TOP_P"):
    try:
        value = float(os.environ[name])
    except ValueError as error:
        raise SystemExit(f"--{name.lower().replace('_', '-')} expects a number") from error
    if not math.isfinite(value):
        raise SystemExit(f"--{name.lower().replace('_', '-')} must be finite")
temperature = float(os.environ["TEMPERATURE"])
top_p = float(os.environ["TOP_P"])
if temperature < 0:
    raise SystemExit("--temperature must be non-negative")
if not 0 <= top_p <= 1:
    raise SystemExit("--top-p must be in [0, 1]")
PY
then
    exit 2
fi

if ((${#PROMPTS[@]} == 0 || ${#PROMPT_IDS[@]} != ${#PROMPTS[@]})); then
    echo "prompt IDs and prompts must define the same nonzero workload set" >&2
    exit 2
fi
for prompt_index in "${!PROMPTS[@]}"; do
    if [[ -z "${PROMPTS[$prompt_index]}" ]]; then
        echo "prompt ${PROMPT_IDS[$prompt_index]} must not be empty" >&2
        exit 2
    fi
done

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
[[ -n "$DFLASH2_27B" ]] || DFLASH2_27B="$MODEL_ROOT/Qwen3.8-27B-DFlash2-affine"
[[ -n "$MODEL_35B" ]] || MODEL_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-4bit"
[[ -n "$MTP_35B" ]] || MTP_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-MTP-4bit"
[[ -n "$DSPARK_35B" ]] || DSPARK_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-DSpark-affine"
[[ -n "$DFLASH2_35B" ]] || DFLASH2_35B="$MODEL_ROOT/Qwen3.6-35B-A3B-DFlash2-affine"

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
    27b_mtp)
        append_case 27b_mtp1
        append_case 27b_mtp2
        ;;
    27b_dspark)
        append_case 27b_dspark
        ;;
    27b_dflash2)
        append_case 27b_dflash2
        ;;
    27b_on)
        append_case 27b_mtp1
        append_case 27b_dspark
        append_case 27b_dflash2
        append_case 27b_mtp2
        ;;
    35b_mtp)
        append_case 35b_mtp1
        append_case 35b_mtp2
        ;;
    35b_dspark)
        append_case 35b_dspark
        ;;
    35b_dflash2)
        append_case 35b_dflash2
        ;;
    35b_on)
        append_case 35b_mtp1
        append_case 35b_dspark
        append_case 35b_dflash2
        append_case 35b_mtp2
        ;;
    *) append_case "$case_name" ;;
    esac
done

need_27b=0
need_27b_mtp=0
need_27b_dspark=0
need_27b_dflash2=0
need_35b=0
need_35b_mtp=0
need_35b_dspark=0
need_35b_dflash2=0
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_off) need_27b=1 ;;
    27b_mtp1 | 27b_mtp2)
        need_27b=1
        need_27b_mtp=1
        ;;
    27b_dspark)
        need_27b=1
        need_27b_dspark=1
        ;;
    27b_dflash2)
        need_27b=1
        need_27b_dflash2=1
        ;;
    35b_off) need_35b=1 ;;
    35b_mtp1 | 35b_mtp2)
        need_35b=1
        need_35b_mtp=1
        ;;
    35b_dspark)
        need_35b=1
        need_35b_dspark=1
        ;;
    35b_dflash2)
        need_35b=1
        need_35b_dflash2=1
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
if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
    raise SystemExit(f"{path} must have a positive DSpark block_size")
PY
}

validate_affine_dflash2() {
    local option="$1"
    local dir="$2"
    DFLASH2_DIR="$dir" DFLASH2_OPTION="$option" python3 - <<'PY'
import json
import os
import struct
from pathlib import Path

option = os.environ["DFLASH2_OPTION"]
model_dir = Path(os.environ["DFLASH2_DIR"])
config_path = model_dir / "config.json"
try:
    config = json.loads(config_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"unable to read {config_path}: {exc}") from exc

quantization = config.get("quantization")
if not isinstance(quantization, dict) or quantization.get("mode") != "affine":
    raise SystemExit(f"{option} must name an affine-quantized DFlash2 checkpoint")
if config.get("architectures") != ["DFlash2DraftModel"]:
    raise SystemExit(f"{config_path} must describe one DFlash2DraftModel")

for tensor_path in sorted(model_dir.glob("*.safetensors")):
    try:
        with tensor_path.open("rb") as stream:
            header_bytes = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_bytes))
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise SystemExit(f"unable to inspect {tensor_path}: {exc}") from exc
    bf16_matrices = [name for name, metadata in header.items()
                     if name != "__metadata__"
                     and metadata.get("dtype") == "BF16"
                     and len(metadata.get("shape", [])) == 2
                     and not name.endswith(".scales")
                     and not name.endswith(".biases")]
    if bf16_matrices:
        raise SystemExit(
            f"{option} contains unquantized BF16 matrices in {tensor_path.name}; "
            "run qwen3x_spec_quantize dflash2 first"
        )
    invalid_affine = [name for name, metadata in header.items()
                      if name != "__metadata__"
                      and (name.endswith(".scales") or name.endswith(".biases"))
                      and metadata.get("dtype") != "BF16"]
    if invalid_affine:
        raise SystemExit(
            f"{option} contains non-BF16 affine parameters in {tensor_path.name}; "
            "regenerate it with qwen3x_spec_quantize dflash2"
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

ensure_optional_local_model() {
    local case_name="$1"
    local option="$2"
    local dir="$3"

    if ! model_present "$dir"; then
        echo "WARNING: skipping $case_name because its checkpoint is missing: $dir" >&2
        echo "Pass $option DIR to enable this case." >&2
        return 1
    fi
    echo "==> Found checkpoint: $dir"
}

if ((need_27b_mtp)); then
    ensure_optional_model 27b_mtp --mtp-27b "$MTP_27B_REPO" "$MTP_27B" || need_27b_mtp=0
fi
if ((need_27b_dspark)); then
    ensure_optional_model 27b_dspark --dspark-27b "$DSPARK_27B_REPO" "$DSPARK_27B" || need_27b_dspark=0
fi
if ((need_27b_dflash2)); then
    ensure_optional_local_model 27b_dflash2 --dflash2-27b "$DFLASH2_27B" || need_27b_dflash2=0
fi
if ((need_35b_mtp)); then
    ensure_optional_model 35b_mtp --mtp-35b "$MTP_35B_REPO" "$MTP_35B" || need_35b_mtp=0
fi
if ((need_35b_dspark)); then
    ensure_optional_model 35b_dspark --dspark-35b "$DSPARK_35B_REPO" "$DSPARK_35B" || need_35b_dspark=0
fi
if ((need_35b_dflash2)); then
    ensure_optional_local_model 35b_dflash2 --dflash2-35b "$DFLASH2_35B" || need_35b_dflash2=0
fi

runnable_cases=()
for case_name in "${selected_cases[@]}"; do
    case "$case_name" in
    27b_mtp1 | 27b_mtp2) ((need_27b_mtp)) || continue ;;
    27b_dspark) ((need_27b_dspark)) || continue ;;
    27b_dflash2) ((need_27b_dflash2)) || continue ;;
    35b_mtp1 | 35b_mtp2) ((need_35b_mtp)) || continue ;;
    35b_dspark) ((need_35b_dspark)) || continue ;;
    35b_dflash2) ((need_35b_dflash2)) || continue ;;
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
if ((need_27b_dflash2)); then
    require_dir "--dflash2-27b" "$DFLASH2_27B"
    validate_affine_dflash2 "--dflash2-27b" "$DFLASH2_27B"
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
if ((need_35b_dflash2)); then
    require_dir "--dflash2-35b" "$DFLASH2_35B"
    validate_affine_dflash2 "--dflash2-35b" "$DFLASH2_35B"
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

prompt_set_sha256() {
    local prompt_index
    for prompt_index in "${!PROMPTS[@]}"; do
        printf '%s\034%s\035' "${PROMPT_IDS[$prompt_index]}" "${PROMPTS[$prompt_index]}"
    done | shasum -a 256 | awk '{print $1}'
}

prompt_ids_csv() {
    local IFS=,
    printf '%s' "${PROMPT_IDS[*]}"
}

reference_config_mismatches() {
    local current_machine="$1"
    local current_os="$2"
    local current_arch="$3"
    local current_dirty="$4"
    local current_prompt_set_sha256="$5"
    local mismatches=""

    [[ "$current_machine" == "$REFERENCE_MACHINE" ]] || mismatches="machine"
    [[ "$current_os" == "$REFERENCE_OS_VERSION" ]] || mismatches="${mismatches:+$mismatches,}os"
    [[ "$current_arch" == "$REFERENCE_ARCH" ]] || mismatches="${mismatches:+$mismatches,}arch"
    [[ "$current_dirty" == "$REFERENCE_DIRTY" ]] || mismatches="${mismatches:+$mismatches,}dirty"
    [[ "$CACHE_BLOCK_TOKENS" == "$REFERENCE_CACHE_BLOCK_TOKENS" ]] ||
        mismatches="${mismatches:+$mismatches,}cache_block_tokens"
    [[ "$MAX_TOKENS" == "$REFERENCE_MAX_TOKENS" ]] || mismatches="${mismatches:+$mismatches,}max_tokens"
    [[ "$MAX_TOKENS_PER_REQUEST" == "$REFERENCE_MAX_TOKENS_PER_REQUEST" ]] ||
        mismatches="${mismatches:+$mismatches,}max_tokens_per_request"
    [[ "$CASE_COOLDOWN_SECS" == "$REFERENCE_CASE_COOLDOWN_SECS" ]] ||
        mismatches="${mismatches:+$mismatches,}case_cooldown_secs"
    [[ "$LOGGING" == "$REFERENCE_LOGGING" ]] || mismatches="${mismatches:+$mismatches,}logging"
    [[ "$SEED" == "$REFERENCE_SEED" ]] || mismatches="${mismatches:+$mismatches,}seed"
    [[ -z "$BLOCK_SPEC_TOKENS" ]] || mismatches="${mismatches:+$mismatches,}block_spec_tokens"
    [[ "$TEMPERATURE" == "$REFERENCE_TEMPERATURE" ]] || mismatches="${mismatches:+$mismatches,}temperature"
    [[ "$TOP_K" == "$REFERENCE_TOP_K" ]] || mismatches="${mismatches:+$mismatches,}top_k"
    [[ "$TOP_P" == "$REFERENCE_TOP_P" ]] || mismatches="${mismatches:+$mismatches,}top_p"
    [[ "$PROMPT_SET" == "$REFERENCE_PROMPT_SET" ]] || mismatches="${mismatches:+$mismatches,}prompt_set"
    [[ "$current_prompt_set_sha256" == "$REFERENCE_PROMPT_SET_SHA256" ]] ||
        mismatches="${mismatches:+$mismatches,}prompt_set_sha256"
    [[ -z "$TOKENIZER" ]] || mismatches="${mismatches:+$mismatches,}tokenizer"
    if ((need_27b)); then
        [[ "$(basename "$MODEL_27B")" == "$REFERENCE_MODEL_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}model_27b"
    fi
    if ((need_27b_mtp)); then
        [[ "$(basename "$MTP_27B")" == "$REFERENCE_MTP_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}mtp_27b"
    fi
    if ((need_27b_dspark)); then
        [[ "$(basename "$DSPARK_27B")" == "$REFERENCE_DSPARK_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}dspark_27b"
    fi
    if ((need_27b_dflash2)); then
        [[ "$(basename "$DFLASH2_27B")" == "$REFERENCE_DFLASH2_27B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}dflash2_27b"
    fi
    if ((need_35b)); then
        [[ "$(basename "$MODEL_35B")" == "$REFERENCE_MODEL_35B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}model_35b"
    fi
    if ((need_35b_mtp)); then
        [[ "$(basename "$MTP_35B")" == "$REFERENCE_MTP_35B_DIR_NAME" ]] ||
            mismatches="${mismatches:+$mismatches,}mtp_35b"
    fi
    echo "$mismatches"
}

reference_row() {
    # Controlled medians from REFERENCE_RUNS runs on REFERENCE_DATE at REFERENCE_COMMIT.
    # Fields: decode, TTFT, prompt throughput, inter-chunk p50, inter-chunk p95,
    # tokens/chunk, verified/proposed, conditional acceptance by index, input tokens,
    # sampled tokens, chunks, proposed tokens, verified tokens.
    case "$1:$2:$3:$4" in
    apple_m3_max_40_gpu_cores:27b_off:256:gsm8k_typing_average) echo "22.737|754.472|181.584|44.164|44.719|1.000|na|[]|137|256|256|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_off:384:gsm8k_typing_average) echo "22.752|808.526|169.444|44.162|44.743|1.000|na|[]|137|290|290|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_off:256:beijing_travel) echo "22.580|471.050|161.342|44.660|45.158|1.000|na|[]|76|256|256|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_off:384:beijing_travel) echo "21.253|447.709|169.753|47.334|49.428|1.000|na|[]|76|384|384|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:256:gsm8k_typing_average) echo "94.850|145.039|654.996|10.656|10.817|1.000|na|[]|95|256|256|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:1024:gsm8k_typing_average) echo "93.458|146.525|648.354|10.733|10.860|1.000|na|[]|95|1024|1024|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:256:beijing_travel) echo "96.381|71.553|475.174|10.447|10.829|1.000|na|[]|34|256|256|0|0" ;;
    apple_m3_max_40_gpu_cores:35b_off:1024:beijing_travel) echo "93.998|68.765|494.438|10.709|10.835|1.000|na|[]|34|1024|1024|0|0" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:256:gsm8k_typing_average) echo "39.001|768.498|178.270|48.710|49.179|1.882|0.888889|[0.888889]|137|256|136|135|120" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:384:gsm8k_typing_average) echo "39.506|765.964|178.860|48.484|48.624|1.899|0.905063|[0.905063]|137|302|159|158|143" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:256:beijing_travel) echo "34.671|448.468|169.466|48.388|48.777|1.662|0.666667|[0.666667]|76|256|154|153|102" ;;
    apple_m3_max_40_gpu_cores:27b_mtp1:384:beijing_travel) echo "33.822|447.526|169.823|48.650|49.092|1.634|0.636752|[0.636752]|76|384|235|234|149" ;;
    apple_m3_max_40_gpu_cores:27b_dspark:256:gsm8k_typing_average) echo "40.110|772.435|177.361|104.658|104.903|4.129|0.466042|[0.819672,0.840000,0.809524,0.852941,0.620690,0.777778,0.857143]|137|256|62|427|199" ;;
    apple_m3_max_40_gpu_cores:27b_dspark:384:gsm8k_typing_average) echo "44.528|772.793|177.279|104.776|104.970|4.597|0.525151|[0.845070,0.866667,0.826923,0.883721,0.684211,0.846154,0.909091]|137|331|72|497|261" ;;
    apple_m3_max_40_gpu_cores:27b_dspark:256:beijing_travel) echo "16.665|454.229|167.317|104.630|104.906|1.730|0.105928|[0.442177,0.338462,0.590909,0.615385,0.125000,0.000000,0.000000]|76|256|148|1029|109" ;;
    apple_m3_max_40_gpu_cores:27b_dspark:384:beijing_travel) echo "16.234|452.558|167.934|104.817|105.034|1.692|0.099874|[0.433628,0.336735,0.515152,0.529412,0.111111,0.000000,0.000000]|76|384|227|1582|158" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2:256:gsm8k_typing_average) echo "55.946|771.662|177.539|106.305|107.779|5.818|0.710963|[0.906977,0.897436,0.942857,0.939394,0.903226,0.928571,0.846154]|137|256|44|301|214" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2:384:gsm8k_typing_average) echo "56.804|772.765|177.286|105.858|107.652|5.894|0.717391|[0.913043,0.904762,0.947368,0.916667,0.909091,0.933333,0.857143]|137|277|47|322|231" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2:256:beijing_travel) echo "18.480|450.536|168.688|105.788|106.401|1.939|0.135224|[0.534351,0.428571,0.466667,0.500000,0.428571,0.000000,0.000000]|76|256|132|917|124" ;;
    apple_m3_max_40_gpu_cores:27b_dflash2:384:beijing_travel) echo "18.151|457.491|166.123|111.548|114.785|2.010|0.146617|[0.552632,0.466667,0.448980,0.545455,0.583333,0.000000,0.000000]|76|384|191|1330|195" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:256:gsm8k_typing_average) echo "30.224|787.979|173.862|88.716|92.539|2.639|0.833333|[0.885417,0.882353]|137|256|97|192|160" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:384:gsm8k_typing_average) echo "31.317|782.802|175.012|86.695|88.729|2.688|0.851852|[0.898148,0.896907]|137|293|109|216|184" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:256:beijing_travel) echo "21.833|459.214|165.500|86.647|89.182|1.869|0.437500|[0.580882,0.506329]|76|256|137|272|119" ;;
    apple_m3_max_40_gpu_cores:27b_mtp2:384:beijing_travel) echo "21.889|461.002|164.858|90.012|94.443|1.959|0.487179|[0.635897,0.532258]|76|384|196|390|190" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:256:gsm8k_typing_average) echo "151.476|148.565|639.450|12.982|13.273|1.954|0.961538|[0.961538]|95|256|131|130|125" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:1024:gsm8k_typing_average) echo "142.768|150.711|630.345|13.678|14.503|1.947|0.948571|[0.948571]|95|1024|526|525|498" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:256:beijing_travel) echo "144.933|73.713|461.245|12.898|13.138|1.855|0.861314|[0.861314]|34|256|138|137|118" ;;
    apple_m3_max_40_gpu_cores:35b_mtp1:1024:beijing_travel) echo "128.258|72.326|470.093|13.465|14.277|1.730|0.732657|[0.732657]|34|1024|592|591|433" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:256:gsm8k_typing_average) echo "147.380|157.113|604.661|18.870|19.289|2.753|0.885870|[0.945652,0.873563]|95|256|93|184|163" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:1024:gsm8k_typing_average) echo "142.607|155.651|610.339|19.410|19.911|2.753|0.880054|[0.940701,0.871060]|95|1024|372|742|653" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:256:beijing_travel) echo "135.331|75.093|452.772|18.744|19.082|2.510|0.762376|[0.861386,0.770115]|34|256|102|202|154" ;;
    apple_m3_max_40_gpu_cores:35b_mtp2:1024:beijing_travel) echo "118.957|73.090|465.183|19.105|19.589|2.265|0.635255|[0.760532,0.670554]|34|1024|452|902|573" ;;
    *) return 1 ;;
    esac
}

print_config_table() {
    CURRENT_COMMIT="$GIT_COMMIT" \
        CURRENT_DIRTY="$GIT_DIRTY" \
        CURRENT_MACHINE="$MACHINE" \
        CURRENT_OS="$OS_VERSION" \
        CURRENT_ARCH="$ARCH" \
        CURRENT_RUNS="$RUNS" \
        CURRENT_CASES="$CASES" \
        CURRENT_BLOCK_SPEC_TOKENS="${BLOCK_SPEC_TOKENS:-checkpoint}" \
        CURRENT_COOLDOWN="$CASE_COOLDOWN_SECS" \
        CURRENT_CAPACITY="$NUM_CACHE_PAGES pages; $CACHE_BLOCK_TOKENS tokens/block; $MAX_REQUESTS requests; $MAX_TOKENS tokens; $MAX_TOKENS_PER_REQUEST tokens/request" \
        CURRENT_SAMPLING="seed=$SEED; temperature=$TEMPERATURE; top_k=$TOP_K; top_p=$TOP_P; thinking=on" \
        CURRENT_PROMPTS="$PROMPT_SET: $PROMPT_IDS_CSV" \
        CURRENT_PROMPT_SHA="$PROMPT_SET_SHA256" \
        CURRENT_MODEL_27B="$MODEL_27B" \
        CURRENT_MTP_27B="$MTP_27B" \
        CURRENT_DSPARK_27B="$DSPARK_27B" \
        CURRENT_DFLASH2_27B="$DFLASH2_27B" \
        CURRENT_MODEL_35B="$MODEL_35B" \
        CURRENT_MTP_35B="$MTP_35B" \
        CURRENT_DSPARK_35B="$DSPARK_35B" \
        CURRENT_DFLASH2_35B="$DFLASH2_35B" \
        CURRENT_REFERENCE="enabled=$REFERENCE; $REFERENCE_DATE; $REFERENCE_COMMIT; $REFERENCE_MACHINE; runs=$REFERENCE_RUNS" \
        CURRENT_REFERENCE_MISMATCHES="${REFERENCE_CONFIG_MISMATCHES:-none}" \
        python3 - <<'PY'
import os
import textwrap

rows = [
    ("Source", f"{os.environ['CURRENT_COMMIT']} dirty={os.environ['CURRENT_DIRTY']}"),
    ("Host", f"{os.environ['CURRENT_MACHINE']} os={os.environ['CURRENT_OS']} arch={os.environ['CURRENT_ARCH']}"),
    ("Run", f"runs={os.environ['CURRENT_RUNS']} cooldown={os.environ['CURRENT_COOLDOWN']}s"),
    ("Cases", os.environ["CURRENT_CASES"]),
    ("Block Spec K", os.environ["CURRENT_BLOCK_SPEC_TOKENS"]),
    ("Capacity", os.environ["CURRENT_CAPACITY"]),
    ("Sampling", os.environ["CURRENT_SAMPLING"]),
    ("Prompts", os.environ["CURRENT_PROMPTS"]),
    ("Prompt SHA-256", os.environ["CURRENT_PROMPT_SHA"]),
    ("27B Main", os.environ["CURRENT_MODEL_27B"]),
    ("27B MTP", os.environ["CURRENT_MTP_27B"]),
    ("27B DSpark", os.environ["CURRENT_DSPARK_27B"]),
    ("27B DFlash2", os.environ["CURRENT_DFLASH2_27B"]),
    ("35B Main", os.environ["CURRENT_MODEL_35B"]),
    ("35B MTP", os.environ["CURRENT_MTP_35B"]),
    ("35B DSpark", os.environ["CURRENT_DSPARK_35B"]),
    ("35B DFlash2", os.environ["CURRENT_DFLASH2_35B"]),
    ("Reference", os.environ["CURRENT_REFERENCE"]),
    ("Ref mismatch", os.environ["CURRENT_REFERENCE_MISMATCHES"]),
]
key_width = max(len(key) for key, _ in rows)
value_width = 96
separator = "+-" + "-" * key_width + "-+-" + "-" * value_width + "-+"

print(separator)
print(f"| {'Configuration'.ljust(key_width)} | {'Value'.ljust(value_width)} |")
print(separator)
for key, value in rows:
    lines = textwrap.wrap(value, value_width, break_long_words=False, break_on_hyphens=False) or [""]
    for index, line in enumerate(lines):
        label = key if index == 0 else ""
        print(f"| {label.ljust(key_width)} | {line.ljust(value_width)} |")
print(separator)
PY
}

print_summary_table() {
    REPORT_FILE="$REPORT_FILE" python3 - <<'PY'
import os


def stable_value(encoded):
    values = encoded.split(",")
    return values[0] if len(set(values)) == 1 else "mixed"


rows = []
reference_status_counts = {}
with open(os.environ["REPORT_FILE"], encoding="utf-8") as report:
    for line in report:
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        sampled = stable_value(fields["samples"])
        proposed = stable_value(fields["proposed_spec"])
        verified = stable_value(fields["verified_spec"])
        acceptance = "-" if proposed == "0" else f"{verified}/{proposed}"
        prompt = {
            "gsm8k_typing_average": "gsm8k",
            "beijing_travel": "beijing",
        }.get(fields["prompt"], fields["prompt"])
        rows.append(
            [
                fields["label"],
                prompt,
                fields["max_new"],
                sampled,
                fields["median_decode_tok_s"],
                fields["median_tokens_per_chunk"],
                acceptance,
            ]
        )
        status = fields["reference_status"]
        reference_status_counts[status] = reference_status_counts.get(status, 0) + 1

headers = ["Case", "Prompt", "Max", "Out", "Tok/s", "Tok/ch", "Accept"]
numeric_columns = {2, 3, 4, 5, 6}
widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"


def format_row(values):
    cells = []
    for index, value in enumerate(values):
        cells.append(value.rjust(widths[index]) if index in numeric_columns else value.ljust(widths[index]))
    return "| " + " | ".join(cells) + " |"


print()
print(separator)
print(format_row(headers))
print(separator)
previous_case = None
for row in rows:
    if previous_case is not None and row[0] != previous_case:
        print(separator)
    print(format_row(row))
    previous_case = row[0]
print(separator)
print("Tok/s is median decode tokens/s. Accept is verified/proposed speculative tokens.")
statuses = ", ".join(
    f"{status}={count}" for status, count in sorted(reference_status_counts.items())
)
print(f"Reference: {statuses}. Use --show-runs for per-row deltas and details.")
PY
}

if [[ "$BUILD" -eq 1 ]]; then
    cargo build --release --bin qwen3_5_dense --bin qwen3_5_sparse --bin decode
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
    local prompt_id="$3"
    local prompt="$4"
    local run="$5"
    local server_log="$6"
    local tokenizer="$7"
    local out="/tmp/psi_dec_${label}_${tokens}_${prompt_id}_${run}.out"
    local server_log_offset
    server_log_offset="$(wc -c <"$server_log")"
    if ! target/release/decode \
        --server-url "http://127.0.0.1:${PORT}" \
        --max-sampled-tokens "$tokens" \
        --seed "$SEED" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --enable-thinking true \
        --chat-template auto \
        --show-stats \
        --hf-model-dir "$tokenizer" \
        --prompt-str "$prompt" >"$out" 2>&1; then
        echo "DECODE_FAILED label=$label max_new=$tokens prompt=$prompt_id run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
    local json
    json=$(grep "^{" "$out" | tail -n 1 || true)
    if [[ -z "$json" ]]; then
        echo "DECODE_STATS_MISSING label=$label max_new=$tokens prompt=$prompt_id run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
    if ! JSON_LINE="$json" SERVER_LOG="$server_log" SERVER_LOG_OFFSET="$server_log_offset" python3 - <<'PY'
import json
import os
import re

j = json.loads(os.environ["JSON_LINE"])
with open(os.environ["SERVER_LOG"], "rb") as f:
    f.seek(int(os.environ["SERVER_LOG_OFFSET"]))
    server_log = f.read().decode("utf-8", errors="replace")
server_log = re.sub(r"\x1b\[[0-9;]*m", "", server_log)
proposed = 0
verified = 0
spec_by_index = []
verified_by_index = []

def add_counts(total, values):
    total.extend([0] * (len(values) - len(total)))
    for index, value in enumerate(values):
        total[index] += value

def index_counts(line, field):
    match = re.search(rf"{field}=\[([^]]*)\]", line)
    if not match or not match.group(1).strip():
        return []
    return [int(value.strip()) for value in match.group(1).split(",")]

for line in server_log.splitlines():
    if 'phase="executor.batch.perf"' not in line:
        continue
    proposed_match = re.search(r"num_spec_tokens=(\d+)", line)
    verified_match = re.search(r"num_verified_tokens=(\d+)", line)
    if proposed_match and verified_match:
        proposed += int(proposed_match.group(1))
        verified += int(verified_match.group(1))
        add_counts(spec_by_index, index_counts(line, "num_spec_token_by_index"))
        add_counts(verified_by_index, index_counts(line, "num_verified_token_by_index"))
acceptance_rate = verified / proposed if proposed else 0.0
acceptance_rate_by_index = [
    verified_count / spec_count if spec_count else 0.0
    for spec_count, verified_count in zip(spec_by_index, verified_by_index)
]
tokens_per_chunk = j["sampled_tokens"] / j["chunk_count"]
encode_counts = lambda values: ":".join(str(value) for value in values) or "-"
encode_rates = lambda values: ":".join(f"{value:.6f}" for value in values) or "-"
print("{:.6f},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.6f},{:.6f},{},{},{}".format(
    j["decode_tokens_per_s"],
    j["chunk_count"],
    j["sampled_tokens"],
    j["input_tokens"],
    j["ttft_ms"],
    j["prompt_tokens_per_s"],
    j["inter_chunk_p50_ms"],
    j["inter_chunk_p95_ms"],
    proposed,
    verified,
    acceptance_rate,
    tokens_per_chunk,
    encode_counts(spec_by_index),
    encode_counts(verified_by_index),
    encode_rates(acceptance_rate_by_index),
))
PY
    then
        echo "DECODE_STATS_INVALID label=$label max_new=$tokens prompt=$prompt_id run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
}

run_server_case() {
    local label="$1"
    local token_list="$2"
    local tokenizer="$3"
    local prompt_index
    shift 3
    local log="/tmp/psi_dec_${label}.log"

    local server_command=("$@")
    local server_rust_log="${RUST_LOG:-info}"
    server_rust_log+=",inference-runtime-service::perf=debug"
    server_command+=(--logging "$LOGGING")
    RUST_LOG="$server_rust_log" "${server_command[@]}" >"$log" 2>&1 &
    ACTIVE_SERVER_PID=$!

    if ! wait_for_port; then
        echo "SERVER_START_FAILED $label" >&2
        tail -n 80 "$log" >&2 || true
        cleanup_active_server
        exit 1
    fi

    for prompt_index in "${!PROMPTS[@]}"; do
        local prompt_id="${PROMPT_IDS[$prompt_index]}"
        local prompt="${PROMPTS[$prompt_index]}"
        for tokens in $token_list; do
            echo "MEASURE case=$label prompt=$prompt_id max_new=$tokens runs=$RUNS"
            local vals=""
            local inputs=""
            local chunks=""
            local samples=""
            local ttfts=""
            local prompt_rates=""
            local inter_chunk_p50s=""
            local inter_chunk_p95s=""
            local proposed_specs=""
            local verified_specs=""
            local acceptance_rates=""
            local tokens_per_chunks=""
            local spec_by_index=""
            local verified_by_index=""
            for run in $(seq 1 "$RUNS"); do
                local parsed tokps chunk sampled input_tokens ttft prompt_rate
                local inter_chunk_p50 inter_chunk_p95 proposed_spec verified_spec
                local acceptance_rate tokens_per_chunk
                local run_spec_by_index run_verified_by_index acceptance_rate_by_index
                parsed=$(run_decode "$label" "$tokens" "$prompt_id" "$prompt" "$run" "$log" "$tokenizer")
                IFS=, read -r \
                    tokps chunk sampled input_tokens ttft prompt_rate \
                    inter_chunk_p50 inter_chunk_p95 proposed_spec verified_spec \
                    acceptance_rate tokens_per_chunk run_spec_by_index \
                    run_verified_by_index acceptance_rate_by_index <<<"$parsed"
                vals="$vals $tokps"
                inputs="$inputs $input_tokens"
                chunks="$chunks $chunk"
                samples="$samples $sampled"
                ttfts="$ttfts $ttft"
                prompt_rates="$prompt_rates $prompt_rate"
                inter_chunk_p50s="$inter_chunk_p50s $inter_chunk_p50"
                inter_chunk_p95s="$inter_chunk_p95s $inter_chunk_p95"
                proposed_specs="$proposed_specs $proposed_spec"
                verified_specs="$verified_specs $verified_spec"
                acceptance_rates="$acceptance_rates $acceptance_rate"
                tokens_per_chunks="$tokens_per_chunks $tokens_per_chunk"
                spec_by_index="$spec_by_index $run_spec_by_index"
                verified_by_index="$verified_by_index $run_verified_by_index"
                if ((SHOW_RUNS)); then
                    echo "RUN label=$label max_new=$tokens prompt=$prompt_id run=$run" \
                        "input_tokens=$input_tokens sampled=$sampled chunks=$chunk" \
                        "proposed_spec=$proposed_spec verified_spec=$verified_spec" \
                        "acceptance_rate=$acceptance_rate acceptance_rate_by_index=$acceptance_rate_by_index" \
                        "tokens_per_chunk=$tokens_per_chunk" \
                        "decode_tok_s=$tokps ttft_ms=$ttft prompt_tok_s=$prompt_rate" \
                        "inter_chunk_p50_ms=$inter_chunk_p50 inter_chunk_p95_ms=$inter_chunk_p95"
                fi
            done

            local reference_record=""
            local reference_decode=""
            local reference_ttft=""
            local reference_prompt_rate=""
            local reference_inter_chunk_p50=""
            local reference_inter_chunk_p95=""
            local reference_tokens_per_chunk=""
            local reference_acceptance_rate=""
            local reference_acceptance_by_index=""
            local reference_input_tokens=""
            local reference_sampled=""
            local reference_chunks=""
            local reference_proposed_spec=""
            local reference_verified_spec=""
            local reference_status="disabled"
            local reference_mismatch=""
            if ((REFERENCE)); then
                reference_record="$(reference_row "$MACHINE" "$label" "$tokens" "$prompt_id" || true)"
                if [[ -z "$reference_record" ]]; then
                    reference_status="no-reference-row"
                    reference_mismatch="row"
                else
                    IFS='|' read -r \
                        reference_decode reference_ttft reference_prompt_rate \
                        reference_inter_chunk_p50 reference_inter_chunk_p95 \
                        reference_tokens_per_chunk reference_acceptance_rate \
                        reference_acceptance_by_index reference_input_tokens \
                        reference_sampled reference_chunks reference_proposed_spec \
                        reference_verified_spec <<<"$reference_record"
                    if [[ -n "$REFERENCE_CONFIG_MISMATCHES" ]]; then
                        reference_status="config-mismatch"
                        reference_mismatch="$REFERENCE_CONFIG_MISMATCHES"
                    elif ((RUNS < REFERENCE_RUNS)); then
                        reference_status="insufficient-runs"
                        reference_mismatch="runs"
                    else
                        reference_status="comparable"
                    fi
                fi
            fi

            local summary
            summary="$(
                VALS="$vals" \
                    INPUTS="$inputs" \
                    CHUNKS="$chunks" \
                    SAMPLES="$samples" \
                    TTFTS="$ttfts" \
                    PROMPT_RATES="$prompt_rates" \
                    INTER_CHUNK_P50S="$inter_chunk_p50s" \
                    INTER_CHUNK_P95S="$inter_chunk_p95s" \
                    PROPOSED_SPECS="$proposed_specs" \
                    VERIFIED_SPECS="$verified_specs" \
                    ACCEPTANCE_RATES="$acceptance_rates" \
                    TOKENS_PER_CHUNKS="$tokens_per_chunks" \
                    SPEC_BY_INDEX="$spec_by_index" \
                    VERIFIED_BY_INDEX="$verified_by_index" \
                    LABEL="$label" \
                    TOKENS="$tokens" \
                    PROMPT_ID="$prompt_id" \
                    REFERENCE_DECODE="$reference_decode" \
                    REFERENCE_TTFT="$reference_ttft" \
                    REFERENCE_PROMPT_RATE="$reference_prompt_rate" \
                    REFERENCE_INTER_CHUNK_P50="$reference_inter_chunk_p50" \
                    REFERENCE_INTER_CHUNK_P95="$reference_inter_chunk_p95" \
                    REFERENCE_TOKENS_PER_CHUNK="$reference_tokens_per_chunk" \
                    REFERENCE_ACCEPTANCE_RATE="$reference_acceptance_rate" \
                    REFERENCE_ACCEPTANCE_BY_INDEX="$reference_acceptance_by_index" \
                    REFERENCE_INPUT_TOKENS="$reference_input_tokens" \
                    REFERENCE_SAMPLED="$reference_sampled" \
                    REFERENCE_CHUNKS="$reference_chunks" \
                    REFERENCE_PROPOSED_SPEC="$reference_proposed_spec" \
                    REFERENCE_VERIFIED_SPEC="$reference_verified_spec" \
                    REFERENCE_STATUS="$reference_status" \
                    REFERENCE_MISMATCH="$reference_mismatch" \
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
verified_specs = os.environ["VERIFIED_SPECS"].split()
acceptance_rates = [float(x) for x in os.environ["ACCEPTANCE_RATES"].split()]
tokens_per_chunks = [float(x) for x in os.environ["TOKENS_PER_CHUNKS"].split()]

def sum_index_counts(name):
    totals = []
    for encoded in os.environ[name].split():
        values = [] if encoded == "-" else [int(value) for value in encoded.split(":")]
        totals.extend([0] * (len(values) - len(totals)))
        for index, value in enumerate(values):
            totals[index] += value
    return totals

spec_by_index = sum_index_counts("SPEC_BY_INDEX")
verified_by_index = sum_index_counts("VERIFIED_BY_INDEX")
acceptance_rate_by_index = [
    verified / spec if spec else 0.0
    for spec, verified in zip(spec_by_index, verified_by_index)
]
acceptance_rate_by_index_text = "[{}]".format(
    ",".join(f"{value:.6f}" for value in acceptance_rate_by_index)
)
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
    "SUMMARY label={} max_new={} prompt={} median_decode_tok_s={:.3f} median_ttft_ms={:.3f} "
    "median_prompt_tok_s={:.3f} median_inter_chunk_p50_ms={:.3f} "
    "median_inter_chunk_p95_ms={:.3f} median_tokens_per_chunk={:.3f} "
    "median_acceptance_rate={} acceptance_rate_by_index={}"
).format(
    os.environ["LABEL"],
    os.environ["TOKENS"],
    os.environ["PROMPT_ID"],
    median_decode,
    median_ttft,
    median_prompt_rate,
    median_inter_chunk_p50,
    median_inter_chunk_p95,
    median_tokens_per_chunk,
    acceptance_rate_text,
    acceptance_rate_by_index_text,
)
reference_decode = os.environ.get("REFERENCE_DECODE", "")
reference_status = os.environ["REFERENCE_STATUS"]
reference_mismatch = os.environ.get("REFERENCE_MISMATCH", "")
if reference_status == "comparable" and (
    any(value != os.environ["REFERENCE_INPUT_TOKENS"] for value in inputs)
    or any(value != os.environ["REFERENCE_SAMPLED"] for value in samples)
    or any(value != os.environ["REFERENCE_CHUNKS"] for value in chunks)
    or any(value != os.environ["REFERENCE_PROPOSED_SPEC"] for value in proposed_specs)
    or any(value != os.environ["REFERENCE_VERIFIED_SPEC"] for value in verified_specs)
    or acceptance_rate_by_index_text != os.environ["REFERENCE_ACCEPTANCE_BY_INDEX"]
):
    reference_status = "trajectory-mismatch"
    reference_mismatch = "trajectory"
if reference_decode:
    reference_decode_value = float(reference_decode)
    reference_ttft = float(os.environ["REFERENCE_TTFT"])
    reference_inter_chunk_p95 = float(os.environ["REFERENCE_INTER_CHUNK_P95"])
    prefix += (
        " reference_decode_tok_s={:.3f} reference_ttft_ms={:.3f} "
        "reference_prompt_tok_s={} reference_inter_chunk_p50_ms={} "
        "reference_inter_chunk_p95_ms={:.3f} reference_tokens_per_chunk={} "
        "reference_acceptance_rate={} reference_acceptance_rate_by_index={}"
    ).format(
        reference_decode_value,
        reference_ttft,
        os.environ["REFERENCE_PROMPT_RATE"],
        os.environ["REFERENCE_INTER_CHUNK_P50"],
        reference_inter_chunk_p95,
        os.environ["REFERENCE_TOKENS_PER_CHUNK"],
        os.environ["REFERENCE_ACCEPTANCE_RATE"],
        os.environ["REFERENCE_ACCEPTANCE_BY_INDEX"],
    )
    if reference_status == "comparable":
        prefix += " decode_delta_pct={:+.2f} ttft_delta_pct={:+.2f} inter_chunk_p95_delta_pct={:+.2f}".format(
            100.0 * (median_decode - reference_decode_value) / reference_decode_value,
            100.0 * (median_ttft - reference_ttft) / reference_ttft,
            100.0 * (median_inter_chunk_p95 - reference_inter_chunk_p95) / reference_inter_chunk_p95,
        )
prefix += f" reference_status={reference_status}"
if reference_mismatch:
    prefix += f" reference_mismatch={reference_mismatch}"
print(
    "{} min_decode_tok_s={:.3f} max_decode_tok_s={:.3f} runs={} input_tokens={} samples={} chunks={} proposed_spec={} verified_spec={}".format(
        prefix,
        min(vals),
        max(vals),
        ",".join("{:.3f}".format(v) for v in vals),
        ",".join(inputs),
        ",".join(samples),
        ",".join(chunks),
        ",".join(proposed_specs),
        ",".join(verified_specs),
    )
)
PY
            )"
            printf '%s\n' "$summary" >>"$REPORT_FILE"
            if ((SHOW_RUNS)); then
                printf '%s\n' "$summary"
            fi
        done
    done

    cleanup_active_server
}

run_mtp_case() {
    local model_label="$1"
    local num_spec_tokens="$2"
    local token_list="$3"
    local server_binary="$4"
    local model_dir="$5"
    local mtp_model_dir="$6"
    local tokenizer="${TOKENIZER:-$model_dir}"
    local label="${model_label}_mtp${num_spec_tokens}"
    run_server_case "$label" "$token_list" "$tokenizer" "$server_binary" \
        --grpc-listen-addr "127.0.0.1:${PORT}" \
        --hf-model-dir "$model_dir" \
        --hf-spec-model-dir "$mtp_model_dir" \
        --spec-type mtp \
        --num-spec-tokens "$num_spec_tokens" \
        --num-cache-pages "$NUM_CACHE_PAGES" \
        --max-requests "$MAX_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
}

run_block_spec_case() {
    local model_label="$1"
    local spec_mode="$2"
    local token_list="$3"
    local server_binary="$4"
    local model_dir="$5"
    local spec_model_dir="$6"
    local tokenizer="${TOKENIZER:-$model_dir}"

    local label="${model_label}_${spec_mode}"
    set --
    if [[ -n "$BLOCK_SPEC_TOKENS" ]]; then
        label="${label}_k${BLOCK_SPEC_TOKENS}"
        set -- --num-spec-tokens "$BLOCK_SPEC_TOKENS"
    fi
    run_server_case "$label" "$token_list" "$tokenizer" "$server_binary" \
        --grpc-listen-addr "127.0.0.1:${PORT}" \
        --hf-model-dir "$model_dir" \
        --hf-spec-model-dir "$spec_model_dir" \
        --spec-type "$spec_mode" \
        "$@" \
        --num-cache-pages "$NUM_CACHE_PAGES" \
        --max-requests "$MAX_REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST"
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
    27b_mtp1 | 27b_mtp2)
        run_mtp_case 27b "${1#27b_mtp}" "256 384" target/release/qwen3_5_dense "$MODEL_27B" "$MTP_27B"
        ;;
    27b_dspark)
        run_block_spec_case 27b dspark "256 384" target/release/qwen3_5_dense "$MODEL_27B" "$DSPARK_27B"
        ;;
    27b_dflash2)
        run_block_spec_case 27b dflash2 "256 384" target/release/qwen3_5_dense "$MODEL_27B" "$DFLASH2_27B"
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
    35b_mtp1 | 35b_mtp2)
        run_mtp_case 35b "${1#35b_mtp}" "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$MTP_35B"
        ;;
    35b_dspark)
        run_block_spec_case 35b dspark "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$DSPARK_35B"
        ;;
    35b_dflash2)
        run_block_spec_case 35b dflash2 "256 1024" target/release/qwen3_5_sparse "$MODEL_35B" "$DFLASH2_35B"
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
PROMPT_SET_SHA256="$(prompt_set_sha256)"
PROMPT_IDS_CSV="$(prompt_ids_csv)"
REFERENCE_CONFIG_MISMATCHES="$(
    reference_config_mismatches "$MACHINE" "$OS_VERSION" "$ARCH" "$GIT_DIRTY" "$PROMPT_SET_SHA256"
)"
REPORT_FILE="$(mktemp "${TMPDIR:-/tmp}/psi_dec_qwen35_perf.XXXXXX")"
if ((SHOW_RUNS)); then
    echo "CONFIG commit=$GIT_COMMIT dirty=$GIT_DIRTY machine=$MACHINE os=$OS_VERSION arch=$ARCH runs=$RUNS build=$BUILD grpc_port=$PORT num_cache_pages=$NUM_CACHE_PAGES cache_block_tokens=$CACHE_BLOCK_TOKENS max_requests=$MAX_REQUESTS max_tokens=$MAX_TOKENS max_tokens_per_request=$MAX_TOKENS_PER_REQUEST mtp_num_spec_tokens=case-specific block_spec_num_spec_tokens=${BLOCK_SPEC_TOKENS:-checkpoint} cases=$CASES case_cooldown_secs=$CASE_COOLDOWN_SECS logging=$LOGGING seed=$SEED temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P enable_thinking=1 prompt_set=$PROMPT_SET prompt_count=${#PROMPTS[@]} prompt_ids=$PROMPT_IDS_CSV prompt_set_sha256=$PROMPT_SET_SHA256 tokenizer=${TOKENIZER:-auto-per-model} model_27b=$MODEL_27B mtp_27b=$MTP_27B dspark_27b=$DSPARK_27B dflash2_27b=$DFLASH2_27B model_35b=$MODEL_35B mtp_35b=$MTP_35B dspark_35b=$DSPARK_35B dflash2_35b=$DFLASH2_35B reference_enabled=$REFERENCE reference_machine=$REFERENCE_MACHINE reference_date=$REFERENCE_DATE reference_commit=$REFERENCE_COMMIT reference_dirty=$REFERENCE_DIRTY reference_os=$REFERENCE_OS_VERSION reference_arch=$REFERENCE_ARCH reference_runs=$REFERENCE_RUNS reference_cases=$REFERENCE_CASES reference_prompt_set=$REFERENCE_PROMPT_SET reference_prompt_set_sha256=$REFERENCE_PROMPT_SET_SHA256 reference_config_mismatches=${REFERENCE_CONFIG_MISMATCHES:-none}"
fi
print_config_table
for case_index in "${!selected_cases[@]}"; do
    case_name="${selected_cases[$case_index]}"
    if [[ "$case_index" -gt 0 && "$CASE_COOLDOWN_SECS" -gt 0 ]]; then
        echo "COOLDOWN before=$case_name seconds=$CASE_COOLDOWN_SECS"
        sleep "$CASE_COOLDOWN_SECS"
    fi
    run_named_case "$case_name"
done
print_summary_table
