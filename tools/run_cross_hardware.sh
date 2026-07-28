#!/usr/bin/env bash
# Re-run the A40 vs RTX 6000 Ada comparison over the widened 61-prompt corpus.
#
# D10's headline — 19% of prompts diverge across two NVIDIA GPUs — rests on
# 3 divergences in 16 prompts. That interval is far too wide for a number
# published on a public site. 61 prompts tightens it.
#
# TEARDOWN IS GUARANTEED. Every exit path runs `cleanup`, which terminates
# BOTH pods and verifies they are gone. A pod left idle across a blocker once
# cost $10.58 against $0.51 for the measurement that mattered.
#
# Usage:  set -a; . .env.runpod; set +a; tools/run_cross_hardware.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="pvcdqlwm9r"
MODEL="Qwen/Qwen3-8B"
REF_GPU="NVIDIA A40"
CAND_GPU="NVIDIA RTX 6000 Ada Generation"
REF_POD=""; CAND_POD=""

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
  kill_pod "$REF_POD" reference; REF_POD=""
  kill_pod "$CAND_POD" candidate; CAND_POD=""
}
trap cleanup EXIT INT TERM

deploy() {   # name, gpu, key -> pod id on stdout
  cat > /tmp/rt_pod.json <<JSON
{"name":"$1","templateId":"$TEMPLATE","gpuTypeIds":["$2"],"gpuCount":1,
 "containerDiskInGb":5,"volumeInGb":25,"volumeMountPath":"/workspace",
 "ports":["8000/http"],
 "env":{"HF_HOME":"/workspace/.huggingface","VLLM_API_KEY":"$3"}}
JSON
  curl -s -m 60 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" -d @/tmp/rt_pod.json \
    https://rest.runpod.io/v1/pods \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("id",""))'
}

REF_KEY="sk-ruitong-$(openssl rand -hex 8)"
CAND_KEY="sk-ruitong-$(openssl rand -hex 8)"

echo "deploying reference (A40) and candidate (RTX 6000 Ada) ..."
REF_POD=$(deploy ruitong-ref-a40-x  "$REF_GPU"  "$REF_KEY")
CAND_POD=$(deploy ruitong-cand-ada-x "$CAND_GPU" "$CAND_KEY")
[ -z "$REF_POD" ] || [ -z "$CAND_POD" ] && { echo "deploy failed (ref='$REF_POD' cand='$CAND_POD')"; exit 1; }
echo "  reference=$REF_POD  candidate=$CAND_POD"

REF_URL="https://${REF_POD}-8000.proxy.runpod.net"
CAND_URL="https://${CAND_POD}-8000.proxy.runpod.net"

echo "waiting for both vLLM servers (hard cap 30 min) ..."
for i in $(seq 1 90); do
  a=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$REF_URL/health"  || true)
  b=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$CAND_URL/health" || true)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { echo "  both ready after ~$((i*20))s"; break; }
  [ "$i" = "90" ] && { echo "TIMEOUT — tearing down"; exit 1; }
  sleep 20
done

# Both must serve the SAME model, or the comparison is meaningless.
for pair in "ref:$REF_URL:$REF_KEY" "cand:$CAND_URL:$CAND_KEY"; do
  n=${pair%%:*}; r=${pair#*:}; u=${r%:*}; k=${r##*:}
  m=$(curl -s -m 25 -H "Authorization: Bearer $k" "$u/v1/models" \
      | python3 -c 'import sys,json; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))')
  echo "  $n serves: ${m:-<none>}"
  [ "$m" = "$MODEL" ] || { echo "model mismatch on $n — refusing to compare"; exit 1; }
done

cd "$REPO"
source .venv/bin/activate

echo "capturing reference corpus (61 prompts, 3 fetches each) ..."
RUITONG_API_KEY="$REF_KEY" python tools/capture_corpus.py \
  --endpoint "$REF_URL/v1" --model "$MODEL" --max-tokens 64 --top-k 20 \
  --label cuda-a40-qwen3-8b-61 --out corpora/a40_61.json | tail -3 || exit 1

echo "capturing candidate corpus ..."
RUITONG_API_KEY="$CAND_KEY" python tools/capture_corpus.py \
  --endpoint "$CAND_URL/v1" --model "$MODEL" --max-tokens 64 --top-k 20 \
  --label cuda-rtx6000ada-qwen3-8b-61 --out corpora/rtx6000ada_61.json | tail -3 || exit 1

echo
echo "=== CROSS-HARDWARE COMPARISON (61 prompts) ==="
python tools/compare_corpora.py corpora/a40_61.json corpora/rtx6000ada_61.json \
  | tee reports/cross_hardware_61prompt.txt
echo
echo "RUN COMPLETE — teardown follows"
