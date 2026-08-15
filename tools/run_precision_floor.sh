#!/usr/bin/env bash
# Same-GPU-type precision floor: NVIDIA A100 running fp16 vs the same A100
# running bf16. This is the "queued" table row "What is the cross-precision
# noise floor?" — same silicon, same model, only the serving dtype differs.
#
# TEARDOWN IS GUARANTEED. Every exit path runs `cleanup`, which terminates
# BOTH pods and verifies they are gone.
#
# Usage:  set -a; . .env.runpod; set +a; tools/run_precision_floor.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="vllm/vllm-openai:latest"
GPU="NVIDIA A100-SXM4-80GB"
MODEL="Qwen/Qwen3-8B"
FP16_POD=""; BF16_POD=""

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (source .env.runpod)}"

kill_pod() {
  local id="$1" label="$2"
  [ -z "$id" ] && return 0
  echo "  terminating $label ($id) ..."
  curl -s -m 30 -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$id" -o /dev/null -w "    HTTP %{http_code}\n"
  local left
  left=$(curl -s -m 30 -H "Authorization: Bearer $RUNPOD_API_KEY" \
    https://rest.runpod.io/v1/pods | grep -c "$id" || true)
  if [ "$left" != "0" ]; then
    echo "    !! $label MAY STILL BE RUNNING — check console.runpod.io/pods"
  else
    echo "    confirmed gone."
  fi
}

cleanup() {
  echo
  echo "[$(date +%H:%M:%S)] teardown"
  kill_pod "$FP16_POD" fp16; FP16_POD=""
  kill_pod "$BF16_POD" bf16; BF16_POD=""
}
trap cleanup EXIT INT TERM

deploy() {   # name, key, dtype -> pod id on stdout
  cat > "/tmp/rt_pod_$3.json" <<JSON
{"name":"$1","imageName":"$IMAGE","gpuTypeIds":["$GPU"],"gpuCount":1,
 "containerDiskInGb":20,"volumeInGb":25,"volumeMountPath":"/workspace",
 "ports":["8000/http"],
 "dockerStartCmd":["--model","$MODEL","--api-key","$2","--dtype","$3"],
 "env":{"HF_HOME":"/workspace/.huggingface"}}
JSON
  curl -s -m 60 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" -d @"/tmp/rt_pod_$3.json" \
    https://rest.runpod.io/v1/pods \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("id",""))'
}

FP16_KEY="sk-ruitong-$(openssl rand -hex 8)"
BF16_KEY="sk-ruitong-$(openssl rand -hex 8)"

echo "deploying fp16 and bf16 pods on $GPU ..."
FP16_POD=$(deploy ruitong-a100-fp16 "$FP16_KEY" float16)
BF16_POD=$(deploy ruitong-a100-bf16 "$BF16_KEY" bfloat16)
[ -z "$FP16_POD" ] || [ -z "$BF16_POD" ] && { echo "deploy failed (fp16='$FP16_POD' bf16='$BF16_POD')"; exit 1; }
echo "  fp16=$FP16_POD  bf16=$BF16_POD"

FP16_URL="https://${FP16_POD}-8000.proxy.runpod.net"
BF16_URL="https://${BF16_POD}-8000.proxy.runpod.net"

echo "waiting for both vLLM servers (hard cap 30 min) ..."
for i in $(seq 1 90); do
  a=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$FP16_URL/health" || true)
  b=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$BF16_URL/health" || true)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { echo "  both ready after ~$((i*20))s"; break; }
  [ "$i" = "90" ] && { echo "TIMEOUT — tearing down"; exit 1; }
  sleep 20
done

for pair in "fp16:$FP16_URL:$FP16_KEY" "bf16:$BF16_URL:$BF16_KEY"; do
  n=${pair%%:*}; r=${pair#*:}; u=${r%:*}; k=${r##*:}
  m=$(curl -s -m 25 -H "Authorization: Bearer $k" "$u/v1/models" \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))')
  echo "  $n serves: ${m:-<none>}"
  [ "$m" = "$MODEL" ] || { echo "model mismatch on $n — refusing to compare"; exit 1; }
done

cd "$REPO"
source .venv/bin/activate

echo "capturing fp16 corpus ..."
RUITONG_API_KEY="$FP16_KEY" python tools/capture_corpus.py \
  --endpoint "$FP16_URL/v1" --model "$MODEL" --max-tokens 64 --top-k 20 \
  --label cuda-a100-fp16-qwen3-8b-61 --out corpora/a100_fp16_61.json | tail -3 || exit 1

echo "capturing bf16 corpus ..."
RUITONG_API_KEY="$BF16_KEY" python tools/capture_corpus.py \
  --endpoint "$BF16_URL/v1" --model "$MODEL" --max-tokens 64 --top-k 20 \
  --label cuda-a100-bf16-qwen3-8b-61 --out corpora/a100_bf16_61.json | tail -3 || exit 1

echo
echo "=== PRECISION FLOOR: A100 fp16 vs A100 bf16 (61 prompts) ==="
python tools/compare_corpora.py corpora/a100_fp16_61.json corpora/a100_bf16_61.json \
  | tee reports/precision_floor_a100_fp16_vs_bf16.txt
echo
echo "RUN COMPLETE — teardown follows"
