# Ruitong Bridge — Deploying vLLM Backends

> **Documentation only — not executed as part of the test suite.**

## CUDA backend (standard vLLM)

```bash
nvidia-docker run -d --gpus all --rm \
  --name ruitong-cuda \
  --network host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096
```

Access at `http://localhost:8000` (default).

## Ascend backend (vllm-ascend on Atlas A2 / 910B)

```bash
docker run -d --rm \
  --name ruitong-ascend \
  --network host \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  quay.io/ascend/vllm-ascend:0.7.3-cann8.1.torch2.5.0 \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 4096
```

Access at `http://localhost:8001` (default).

### Compatibility matrix (pinned row)

| CANN  | torch | torch_npu | vLLM  | vllm-ascend   |
|-------|-------|-----------|-------|---------------|
| 8.1   | 2.5.0 | 2.5.0     | 0.7.3 | 0.7.3         |

**Target hardware:** Atlas A2 (910B) — single card, tensor-parallel = 1.

> **910A is NOT supported.** The vllm-ascend image only works with 910B-class
> hardware.

### Configuration

After starting both servers, set environment variables for the bridge:

```bash
export RUITONG_CUDA_BASE_URL="http://cuda-host:8000"
export RUITONG_ASCEND_BASE_URL="http://ascend-host:8001"
```

### Notes

- vllm-ascend requires CANN 8.1 and matching torch_npu versions. The pinned
  matrix row above has been verified to run together.
- The `quay.io/ascend/vllm-ascend` image bundles all dependencies.
- vLLM exposes the OpenAI-compatible API at `/v1/chat/completions`.
- The bridge (`Ruitong Bridge`) expects both backends to speak the same API.
