# Qwen3-8B Deployment and Testing Summary

## Date: 2025-09-16

## Deployment Information
- **Model**: Qwen/Qwen3-8B (from Hugging Face)
- **GPU**: NVIDIA GeForce RTX 5090 (Compute Capability 12.0)
- **Memory**: 32GB

## Deployment Issues and Solutions

### 1. Initial SGLang Issues
- **Problem**: CUDA kernel error with standard SGLang image
- **Error**: "no kernel image is available for execution on the device"
- **Solution**: Used sglang:blackwell-final-v2 image with torch_native backend

### 2. Working Configurations

#### vLLM (Best Performance)
```bash
docker run -d \
  --name qwen3-8b-vllm \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B
```

#### SGLang (Alternative)
```bash
docker run -d \
  --name qwen3-8b-sglang \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-8B \
  --attention-backend torch_native \
  --disable-cuda-graph \
  --disable-flashinfer
```

## Performance Results

### vLLM Performance
- **Single User**: 82.30 tok/s
- **10 Users**: 719.37 tok/s
- **50 Users**: 2,386.11 tok/s
- **100 Users**: 3,769.59 tok/s
- **Success Rate**: 100% at all loads
- **P95 Latency (100 users)**: 1.13s

### SGLang Performance
- **Single User**: 46.94 tok/s
- **10 Users**: 158.04 tok/s
- **50 Users**: 224.41 tok/s
- **100 Users**: 239.74 tok/s
- **Success Rate**: 100% at all loads
- **P95 Latency (100 users)**: 17.63s

## Key Findings
1. vLLM provides 15.7x better throughput at 100 concurrent users
2. Both frameworks achieved 100% success rate
3. RTX 5090 requires special configuration for SGLang (torch_native backend)
4. vLLM has significantly lower latency under high load

## Test Files Created
- load_test_results_20250916_185136.csv (vLLM 1-100 users)
- load_test_results_20250916_190230.csv (SGLang 1-100 users)
- token_speed_benchmark_20250916_184713.csv (vLLM benchmark)
- token_speed_benchmark_20250916_185852.csv (SGLang benchmark)

## Current Status
✅ Qwen3-8B successfully deployed with SGLang on port 8000