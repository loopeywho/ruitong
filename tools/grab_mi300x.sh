#!/usr/bin/env bash
# Wait for MI300X capacity on RunPod, then capture a corpus and tear down.
#
# RunPod cannot reserve capacity it does not have — Savings Plans attach to an
# already-running pod, and true reserved capacity is an enterprise contract.
# This is the practical substitute: poll until stock appears, grab it, take the
# measurement, and give it back.
#
# TEARDOWN IS GUARANTEED. On 2026-07-28 a pod was left billing for 12.6 hours
# because a blocker arrived between "deploy" and "terminate" — $10.58 for zero
# compute, against $0.51 for the measurement that mattered. Every exit path
# here runs `cleanup`, including Ctrl-C, error, and timeout.
#
# Usage:
#   set -a; . .env.runpod; set +a
#   tools/grab_mi300x.sh [poll_seconds] [max_hours]

set -uo pipefail

POLL="${1:-300}"        # default: check every 5 minutes
MAX_HOURS="${2:-12}"    # give up after this long
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="AMD Instinct MI300X OAM"
POD_ID=""

# MI300X is AMD. The NVIDIA template used for the A40/Ada runs
# (vllm/vllm-openai:latest) is a CUDA image and will NOT start on ROCm — that
# is the same mistake that crash-looped two pods on 2026-07-28.
#
# ROCm needs a ROCm build of vLLM. Override with RUITONG_ROCM_IMAGE if this
# tag moves.
#
# 2026-08-01, two failed attempts before finding the actual cause:
#   1. dockerStartCmd was empty (env vars only — MODEL_NAME/VLLM_API_KEY).
#      Never bound port 8000 in 30 min (uptime stuck at 0, /health 404 the
#      whole time). Fixed to pass --model/--api-key explicitly, matching the
#      pattern already proven on the NVIDIA image the same day (A40 vs H100,
#      ready in ~120s each).
#   2. Same result even with the args fixed. Checked vLLM's own docs instead
#      of guessing again: `rocm/vllm` (AMD's image) is DEPRECATED — that was
#      the actual cause both times, not the invocation. Current image is
#      `vllm/vllm-openai-rocm:latest` (docs.vllm.ai/en/latest/deployment/docker),
#      the direct ROCm counterpart of the NVIDIA image that already works.
ROCM_IMAGE="${RUITONG_ROCM_IMAGE:-vllm/vllm-openai-rocm:latest}"
MODEL="${RUITONG_MODEL:-Qwen/Qwen3-8B}"

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (source .env.runpod)}"

cleanup() {
  if [ -n "$POD_ID" ]; then
    echo "[$(date +%H:%M:%S)] tearing down $POD_ID ..."
    curl -s -m 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
      "https://rest.runpod.io/v1/pods/$POD_ID" -o /dev/null \
      -w "  terminate: HTTP %{http_code}\n"
    # Verify, never assume. A failed DELETE that goes unnoticed is the
    # expensive failure mode.
    left=$(curl -s -m 30 -H "Authorization: Bearer $RUNPOD_API_KEY" \
      https://rest.runpod.io/v1/pods | grep -c "$POD_ID" || true)
    if [ "$left" != "0" ]; then
      echo "  !! POD $POD_ID MAY STILL BE RUNNING — CHECK console.runpod.io/pods"
    else
      echo "  confirmed gone."
    fi
    POD_ID=""
  fi
}
trap cleanup EXIT INT TERM

stock_of() {
  printf '{"query":"query { gpuTypes(input:{id:\\"%s\\"}) { displayName lowestPrice(input:{gpuCount:1}) { uninterruptablePrice stockStatus } } }"}' "$GPU_ID" \
   > /tmp/mi300x_q.json
  curl -s -m 30 -X POST https://api.runpod.io/graphql \
    -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d @/tmp/mi300x_q.json \
  | python3 -c 'import sys,json
try:
    g=json.load(sys.stdin)["data"]["gpuTypes"][0]
    lp=g.get("lowestPrice") or {}
    print(lp.get("stockStatus") or "NONE", lp.get("uninterruptablePrice") or "-")
except Exception:
    print("ERR -")'
}

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
echo "watching for MI300X — poll ${POLL}s, giving up in ${MAX_HOURS}h"

while :; do
  read -r status price <<<"$(stock_of)"
  echo "[$(date +%H:%M:%S)] MI300X stock=$status price=$price"
  [ "$status" != "NONE" ] && [ "$status" != "ERR" ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "gave up after ${MAX_HOURS}h — MI300X never appeared. Nothing was rented."
    exit 0
  fi
  sleep "$POLL"
done

echo "MI300X AVAILABLE — deploying"
KEY="sk-ruitong-$(openssl rand -hex 8)"
cat > /tmp/mi300x_pod.json <<JSON
{"name":"ruitong-cand-mi300x","imageName":"$ROCM_IMAGE","gpuTypeIds":["$GPU_ID"],
 "gpuCount":1,"containerDiskInGb":20,"volumeInGb":60,"volumeMountPath":"/workspace",
 "ports":["8000/http"],
 "dockerStartCmd":["--model","$MODEL","--api-key","$KEY"],
 "env":{"HF_HOME":"/workspace/.huggingface","VLLM_USE_TRITON_FLASH_ATTN":"0"}}
JSON
echo "  image=$ROCM_IMAGE  model=$MODEL"
POD_ID=$(curl -s -m 60 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" -d @/tmp/mi300x_pod.json \
  https://rest.runpod.io/v1/pods | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))')

if [ -z "$POD_ID" ]; then echo "deploy failed"; exit 1; fi
echo "pod $POD_ID deployed — waiting for vLLM"

URL="https://${POD_ID}-8000.proxy.runpod.net"
for i in $(seq 1 90); do   # hard cap ~30 min
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$URL/health" || true)
  [ "$code" = "200" ] && { echo "vLLM ready after ~$((i*20))s"; break; }
  if [ "$i" = "90" ]; then echo "TIMEOUT waiting for vLLM — tearing down"; exit 1; fi
  sleep 20
done

# ROCm has a documented history of logprobs coming back as -9999 sentinels
# (vllm#19305). Check that FIRST — a corpus of sentinels is worse than none,
# because it looks like data.
echo "smoke-testing logprobs before spending time on a full capture..."
curl -s -m 120 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say hello."}],"max_tokens":5,"temperature":0,"logprobs":true,"top_logprobs":5,"chat_template_kwargs":{"enable_thinking":false}}' \
  "$URL/v1/chat/completions" > /tmp/mi300x_smoke.json
python3 - <<'PY' || { echo "logprobs unusable on ROCm — see /tmp/mi300x_smoke.json"; exit 1; }
import json,sys
d=json.load(open('/tmp/mi300x_smoke.json'))
if 'error' in d: print("ERROR:",str(d)[:200]); sys.exit(1)
c=d['choices'][0].get('logprobs',{}).get('content')
if not c: print("no logprobs returned"); sys.exit(1)
bad=[t['logprob'] for e in c for t in e['top_logprobs'] if t['logprob']<=-9000]
print(f"positions={len(c)} sentinel_values={len(bad)}")
sys.exit(1 if bad else 0)
PY

echo "logprobs look real — capturing corpus"
cd "$REPO"
source .venv/bin/activate
RUITONG_API_KEY="$KEY" python tools/capture_corpus.py \
  --endpoint "$URL/v1" --model "$MODEL" --max-tokens 64 --top-k 20 \
  --label rocm-mi300x-qwen3-8b --out corpora/mi300x.json

echo
echo "=== CROSS-VENDOR COMPARISON: NVIDIA A40 vs AMD MI300X ==="
python tools/compare_corpora.py corpora/a40_v3.json corpora/mi300x.json \
  | tee reports/cross_vendor_a40_vs_mi300x.txt
echo
echo "done — teardown follows automatically"
