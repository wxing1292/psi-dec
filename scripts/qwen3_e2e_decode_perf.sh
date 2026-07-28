#!/usr/bin/env bash
set -euo pipefail

RUNS=7
GRPC_PORT=50061
HTTP_PORT=8000
BUILD=1
PROMPT="你好，北京有什么好玩的景点？香山如何？早上去晚上去？单纯爬山么？还有什么可以在香山玩的？"
MODEL="${PSI_DEC_QWEN3_MODEL_DIR:-}"
TOKENIZER="${PSI_DEC_QWEN3_TOKENIZER_DIR:-}"
MAX_NEW_TOKENS="256,1024"
LOGGING=info
SEED=42
TEMPERATURE=0.7
TOP_K=20
TOP_P=0.8
NUM_CACHE_PAGES=393216
MAX_RUNNING_REQUESTS=8
MAX_REQUESTS=4
MAX_TOKENS=128
MAX_TOKENS_PER_REQUEST=64
CACHE_BLOCK_TOKENS=16
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
Usage: scripts/qwen3_e2e_decode_perf.sh [options]

Runs Qwen3 target-only end-to-end decode performance tests.
The script reports decode throughput, TTFT, prompt throughput, and inter-token latency.

Options:
  --runs N              Runs per output-token count. Default: 7
  --tokens LIST         Comma-separated output-token counts. Default: 256,1024
  --grpc-port N         gRPC server port. Default: 50061
  --http-port N         HTTP server port. Default: 8000
  --prompt TEXT         Prompt string.
  --seed N              Fixed request seed. Default: 42
  --temperature N       Sampling temperature. Default: 0.7
  --top-k N             Sampling top-k. Default: 20
  --top-p N             Sampling top-p. Default: 0.8
  --num-cache-pages N   Qwen3 GQA KV-cache pages. Default: 393216
  --max-requests N      Scheduler request capacity. Default: 4
  --max-tokens N        Scheduler flattened-token capacity. Default: 128
  --max-tokens-per-request N
                        Scheduler per-request token capacity. Default: 64
  --model DIR           Qwen3 model directory. Required unless
                        PSI_DEC_QWEN3_MODEL_DIR is set.
  --tokenizer DIR       Tokenizer and chat-template directory. Default: model directory.
                        PSI_DEC_QWEN3_TOKENIZER_DIR sets the initial value.
  --logging LEVEL       Server logging: info or debug. Default: info
  --no-build            Skip the release build.
  -h, --help            Show this help.

Examples:
  scripts/qwen3_e2e_decode_perf.sh --model models/Qwen3-14B-4bit
  scripts/qwen3_e2e_decode_perf.sh --no-build --tokens 256 --runs 3
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
    --tokens)
        [[ $# -ge 2 ]] || {
            echo "--tokens requires a value" >&2
            exit 2
        }
        MAX_NEW_TOKENS="$2"
        shift 2
        ;;
    --grpc-port)
        [[ $# -ge 2 ]] || {
            echo "--grpc-port requires a value" >&2
            exit 2
        }
        GRPC_PORT="$2"
        shift 2
        ;;
    --http-port)
        [[ $# -ge 2 ]] || {
            echo "--http-port requires a value" >&2
            exit 2
        }
        HTTP_PORT="$2"
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
    --model)
        [[ $# -ge 2 ]] || {
            echo "--model requires a value" >&2
            exit 2
        }
        MODEL="$2"
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
    --logging)
        [[ $# -ge 2 ]] || {
            echo "--logging requires a value" >&2
            exit 2
        }
        LOGGING="$2"
        shift 2
        ;;
    --no-build)
        BUILD=0
        shift
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

require_dir() {
    local option="$1"
    local dir="$2"
    if [[ -z "$dir" || ! -d "$dir" ]]; then
        echo "$option must name an existing directory" >&2
        exit 2
    fi
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required command is unavailable: $command_name" >&2
        exit 2
    fi
}

require_positive_integer "--runs" "$RUNS"
require_positive_integer "--grpc-port" "$GRPC_PORT"
require_positive_integer "--http-port" "$HTTP_PORT"
require_nonnegative_integer "--seed" "$SEED"
require_positive_integer "--top-k" "$TOP_K"
require_positive_integer "--num-cache-pages" "$NUM_CACHE_PAGES"
require_positive_integer "--max-requests" "$MAX_REQUESTS"
require_positive_integer "--max-tokens" "$MAX_TOKENS"
require_positive_integer "--max-tokens-per-request" "$MAX_TOKENS_PER_REQUEST"

if ((GRPC_PORT > 65535)); then
    echo "--grpc-port must be at most 65535" >&2
    exit 2
fi
if ((HTTP_PORT > 65535)); then
    echo "--http-port must be at most 65535" >&2
    exit 2
fi

case "$LOGGING" in
info | debug) ;;
*)
    echo "--logging must be info or debug" >&2
    exit 2
    ;;
esac

if [[ -z "$TOKENIZER" ]]; then
    TOKENIZER="$MODEL"
fi
require_dir "--model" "$MODEL"
require_dir "--tokenizer" "$TOKENIZER"

IFS=, read -r -a token_counts <<<"$MAX_NEW_TOKENS"
if [[ ${#token_counts[@]} -eq 0 ]]; then
    echo "--tokens must include at least one token count" >&2
    exit 2
fi
for tokens in "${token_counts[@]}"; do
    require_positive_integer "--tokens" "$tokens"
done

require_command nc
require_command pgrep
require_command python3
if [[ "$BUILD" -eq 1 ]]; then
    require_command cargo
fi

if ! TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" python3 - <<'PY'
import math
import os
import sys

try:
    temperature = float(os.environ["TEMPERATURE"])
    top_p = float(os.environ["TOP_P"])
except ValueError:
    sys.exit("--temperature and --top-p must be numbers")
if not math.isfinite(temperature) or temperature < 0:
    sys.exit("--temperature must be finite and non-negative")
if not math.isfinite(top_p) or not 0 <= top_p <= 1:
    sys.exit("--top-p must be finite and in [0, 1]")
PY
then
    exit 2
fi

CONFLICTING_PROCESS_PATTERN="target/release/qwen3|target/release/decode|inference-runtime-service|cargo bench|cargo run"
if pgrep -fl "$CONFLICTING_PROCESS_PATTERN" >/dev/null 2>&1; then
    echo "refusing to run while another Qwen, decode, or Cargo performance process is active:" >&2
    pgrep -fl "$CONFLICTING_PROCESS_PATTERN" >&2 || true
    exit 1
fi

if [[ "$BUILD" -eq 1 ]]; then
    cargo build --release -p inference-runtime-service --bin qwen3 --bin decode
fi

if [[ ! -x target/release/qwen3 || ! -x target/release/decode ]]; then
    echo "release binaries are unavailable; remove --no-build or build qwen3 and decode" >&2
    exit 1
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

wait_for_port() {
    for _ in $(seq 1 240); do
        if ! kill -0 "$ACTIVE_SERVER_PID" >/dev/null 2>&1; then
            return 1
        fi
        if nc -z 127.0.0.1 "$GRPC_PORT" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

run_decode() {
    local tokens="$1"
    local run="$2"
    local server_log="$3"
    local out="/tmp/psi_dec_qwen3_${tokens}_${run}.out"

    if ! target/release/decode \
        --server-url "http://127.0.0.1:${GRPC_PORT}" \
        --max-sampled-tokens "$tokens" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --seed "$SEED" \
        --chat-template auto \
        --show-stats \
        --hf-model-dir "$TOKENIZER" \
        --prompt-str "$PROMPT" >"$out" 2>&1; then
        echo "DECODE_FAILED max_new=$tokens run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi

    local json
    json="$(grep "^{" "$out" | tail -n 1 || true)"
    if [[ -z "$json" ]]; then
        echo "DECODE_STATS_MISSING max_new=$tokens run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi

    if ! JSON_LINE="$json" python3 - <<'PY'
import json
import os

stats = json.loads(os.environ["JSON_LINE"])
tokens_per_chunk = stats["sampled_tokens"] / stats["chunk_count"]
print("{:.6f},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f}".format(
    stats["decode_tokens_per_s"],
    stats["chunk_count"],
    stats["sampled_tokens"],
    stats["input_tokens"],
    stats["ttft_ms"],
    stats["prompt_tokens_per_s"],
    stats["inter_chunk_p50_ms"],
    stats["inter_chunk_p95_ms"],
    tokens_per_chunk,
))
PY
    then
        echo "DECODE_STATS_INVALID max_new=$tokens run=$run client_output=$out server_log=$server_log" >&2
        tail -n 80 "$out" >&2 || true
        tail -n 120 "$server_log" >&2 || true
        return 1
    fi
}

summarize_runs() {
    VALS="$1" \
        INPUTS="$2" \
        CHUNKS="$3" \
        SAMPLES="$4" \
        TTFTS="$5" \
        PROMPT_RATES="$6" \
        INTER_CHUNK_P50S="$7" \
        INTER_CHUNK_P95S="$8" \
        TOKENS_PER_CHUNKS="$9" \
        TOKENS="${10}" \
        python3 - <<'PY'
import os
import statistics

decode_rates = [float(value) for value in os.environ["VALS"].split()]
input_tokens = os.environ["INPUTS"].split()
chunks = os.environ["CHUNKS"].split()
samples = os.environ["SAMPLES"].split()
ttfts = [float(value) for value in os.environ["TTFTS"].split()]
prompt_rates = [float(value) for value in os.environ["PROMPT_RATES"].split()]
inter_chunk_p50s = [float(value) for value in os.environ["INTER_CHUNK_P50S"].split()]
inter_chunk_p95s = [float(value) for value in os.environ["INTER_CHUNK_P95S"].split()]
tokens_per_chunks = [float(value) for value in os.environ["TOKENS_PER_CHUNKS"].split()]

print(
    "SUMMARY max_new={} median_decode_tok_s={:.3f} median_ttft_ms={:.3f} "
    "median_prompt_tok_s={:.3f} median_inter_chunk_p50_ms={:.3f} "
    "median_inter_chunk_p95_ms={:.3f} median_tokens_per_chunk={:.3f} "
    "min_decode_tok_s={:.3f} max_decode_tok_s={:.3f} runs={} "
    "input_tokens={} samples={} chunks={}".format(
        os.environ["TOKENS"],
        statistics.median(decode_rates),
        statistics.median(ttfts),
        statistics.median(prompt_rates),
        statistics.median(inter_chunk_p50s),
        statistics.median(inter_chunk_p95s),
        statistics.median(tokens_per_chunks),
        min(decode_rates),
        max(decode_rates),
        ",".join("{:.3f}".format(value) for value in decode_rates),
        ",".join(input_tokens),
        ",".join(samples),
        ",".join(chunks),
    )
)
PY
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
PROMPT_SHA256="$(printf '%s' "$PROMPT" | shasum -a 256 | awk '{print $1}')"

echo "CONFIG commit=$GIT_COMMIT dirty=$GIT_DIRTY machine=$MACHINE os=$OS_VERSION arch=$ARCH model=$MODEL tokenizer=$TOKENIZER max_new=$MAX_NEW_TOKENS runs=$RUNS build=$BUILD grpc_port=$GRPC_PORT http_port=$HTTP_PORT num_cache_pages=$NUM_CACHE_PAGES cache_block_tokens=$CACHE_BLOCK_TOKENS max_running_requests=$MAX_RUNNING_REQUESTS max_requests=$MAX_REQUESTS max_tokens=$MAX_TOKENS max_tokens_per_request=$MAX_TOKENS_PER_REQUEST logging=$LOGGING seed=$SEED temperature=$TEMPERATURE top_k=$TOP_K top_p=$TOP_P enable_thinking=1 prompt_sha256=$PROMPT_SHA256 prompt_chars=${#PROMPT}"

SERVER_LOG="/tmp/psi_dec_qwen3.log"
target/release/qwen3 \
    --grpc-listen-addr "127.0.0.1:${GRPC_PORT}" \
    --http-listen-addr "127.0.0.1:${HTTP_PORT}" \
    --hf-model-dir "$MODEL" \
    --num-cache-pages "$NUM_CACHE_PAGES" \
    --max-requests "$MAX_REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --max-tokens-per-request "$MAX_TOKENS_PER_REQUEST" \
    --logging "$LOGGING" >"$SERVER_LOG" 2>&1 &
ACTIVE_SERVER_PID=$!

if ! wait_for_port; then
    echo "SERVER_START_FAILED server_log=$SERVER_LOG" >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    exit 1
fi

for tokens in "${token_counts[@]}"; do
    vals=""
    inputs=""
    chunks=""
    samples=""
    ttfts=""
    prompt_rates=""
    inter_chunk_p50s=""
    inter_chunk_p95s=""
    tokens_per_chunks=""

    for run in $(seq 1 "$RUNS"); do
        parsed="$(run_decode "$tokens" "$run" "$SERVER_LOG")"
        IFS=, read -r \
            decode_rate chunk_count sampled input_count ttft prompt_rate \
            inter_chunk_p50 inter_chunk_p95 tokens_per_chunk <<<"$parsed"
        vals="$vals $decode_rate"
        inputs="$inputs $input_count"
        chunks="$chunks $chunk_count"
        samples="$samples $sampled"
        ttfts="$ttfts $ttft"
        prompt_rates="$prompt_rates $prompt_rate"
        inter_chunk_p50s="$inter_chunk_p50s $inter_chunk_p50"
        inter_chunk_p95s="$inter_chunk_p95s $inter_chunk_p95"
        tokens_per_chunks="$tokens_per_chunks $tokens_per_chunk"

        echo "RUN max_new=$tokens run=$run input_tokens=$input_count sampled=$sampled chunks=$chunk_count tokens_per_chunk=$tokens_per_chunk decode_tok_s=$decode_rate ttft_ms=$ttft prompt_tok_s=$prompt_rate inter_chunk_p50_ms=$inter_chunk_p50 inter_chunk_p95_ms=$inter_chunk_p95"
    done

    summarize_runs \
        "$vals" \
        "$inputs" \
        "$chunks" \
        "$samples" \
        "$ttfts" \
        "$prompt_rates" \
        "$inter_chunk_p50s" \
        "$inter_chunk_p95s" \
        "$tokens_per_chunks" \
        "$tokens"
done
