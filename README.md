# Qwen3-8B Deployment on RTX 5090

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Test Scripts](#test-scripts)
- [API Usage](#api-usage)

## 🎯 Overview

This repository contains the deployment configurations and performance benchmarks for Qwen3-8B model running on NVIDIA GeForce RTX 5090. The deployment was tested with both vLLM and SGLang inference frameworks.

### Key Achievements
- ✅ Successfully deployed Qwen3-8B on RTX 5090 (Compute Capability 12.0)
- ✅ Achieved 3,770 tok/s throughput with vLLM (100 concurrent users)
- ✅ 100% success rate across all load tests
- ✅ Resolved CUDA compatibility issues with SGLang

## 🖥️ System Requirements

### Hardware
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **System RAM**: 32GB+ recommended
- **Storage**: 50GB+ for model weights and Docker images

### Software
- **OS**: Linux (tested on kernel 6.15.10)
- **Docker**: 20.10+ with NVIDIA runtime
- **CUDA**: 12.0+ (compatible with RTX 5090)
- **Python**: 3.11+

## 🚀 Quick Start

### Using vLLM (Recommended - Best Performance)

```bash
docker run -d \
  --name qwen3-8b-vllm \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000
```

### Using SGLang (Alternative)

```bash
docker run -d \
  --name qwen3-8b-sglang \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e CUDA_LAUNCH_BLOCKING=1 \
  --shm-size 16g \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --disable-cuda-graph \
  --disable-flashinfer \
  --attention-backend torch_native
```

## 📊 Performance Benchmarks

### vLLM Performance Results

| Concurrent Users | Throughput (tok/s) | Success Rate | Avg Response Time | P95 Latency |
|-----------------|-------------------|--------------|-------------------|-------------|
| 1               | 82.30             | 100%         | 0.60s            | 0.60s       |
| 10              | 719.37            | 100%         | 0.69s            | 0.69s       |
| 25              | 1,232.91          | 100%         | 0.86s            | 0.86s       |
| 50              | 2,386.11          | 100%         | 0.89s            | 0.89s       |
| 100             | **3,769.59**      | 100%         | 1.13s            | 1.13s       |

### SGLang Performance Results

| Concurrent Users | Throughput (tok/s) | Success Rate | Avg Response Time | P95 Latency |
|-----------------|-------------------|--------------|-------------------|-------------|
| 1               | 46.94             | 100%         | 1.05s            | 1.05s       |
| 10              | 158.04            | 100%         | 2.68s            | 2.68s       |
| 25              | 197.54            | 100%         | 5.29s            | 5.29s       |
| 50              | 224.41            | 100%         | 9.34s            | 9.35s       |
| 100             | 239.74            | 100%         | 17.63s           | 17.63s      |

### Performance Comparison
- **vLLM** provides **15.7x better throughput** at 100 concurrent users
- **vLLM** has significantly lower latency under high load
- Both frameworks achieved **100% success rate** at all load levels

## 🔧 Deployment Options

### vLLM Configuration Options

```bash
--model              # Model path from HuggingFace
--host               # Server host (0.0.0.0 for all interfaces)
--port               # Server port (default: 8000)
--gpu-memory-utilization  # GPU memory usage (0.95 = 95%)
--max-model-len      # Maximum context length
--tensor-parallel-size    # Number of GPUs for tensor parallelism
--dtype              # Data type (auto, half, float16, bfloat16)
--quantization       # Quantization method (awq, gptq)
```

### SGLang Configuration Options

```bash
--model-path         # Model path from HuggingFace
--host               # Server host
--port               # Server port
--mem-fraction-static     # Static memory allocation (0.85 = 85%)
--schedule-policy    # Scheduling policy (lof, fcfs, random)
--attention-backend  # Backend (flashinfer, torch_native, triton)
--enable-torch-compile    # Enable torch compilation
--disable-cuda-graph      # Disable CUDA graphs (needed for RTX 5090)
--quantization       # Quantization (awq, gptq, int8)
```

## 🐛 Troubleshooting

### CUDA Kernel Error with SGLang

**Problem**: "no kernel image is available for execution on the device"

**Solution**: RTX 5090 requires specific settings due to its compute capability (12.0):
```bash
--attention-backend torch_native \
--disable-cuda-graph \
--disable-flashinfer \
--disable-radix-cache
```

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce `--mem-fraction-static` to 0.80 or lower
2. Lower `--max-model-len` to reduce context size
3. Use quantization (`--quantization awq`)

### Slow Performance

**Problem**: Lower than expected throughput

**Solutions**:
1. Ensure NVIDIA driver is up-to-date (575.64.05+)
2. Check GPU utilization with `nvidia-smi`
3. Use vLLM instead of SGLang for better performance
4. Verify no other processes are using GPU

## 📝 Test Scripts

### Simple Token Test
```python
# Test basic functionality
python3 simple_token_test.py 8000
```

### Comprehensive Benchmark
```python
# Run full benchmark suite
python3 token_speed_benchmark.py
```

### Load Testing (1-100 users)
```python
# Test concurrent performance
python3 concurrent_load_test.py
```

## 🔌 API Usage

### Completions Endpoint

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "Hello, how are you?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat Completions Endpoint

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 200
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 📁 Repository Structure

```
/home/qwen-8b/
├── README.md                                  # This file
├── QWEN3-8B_DEPLOYMENT_SUMMARY.md           # Deployment summary
├── load_test_results_20250916_185136.csv    # vLLM load test results
├── load_test_results_20250916_190230.csv    # SGLang load test results
├── token_speed_benchmark_20250916_184713.csv # vLLM benchmark results
└── token_speed_benchmark_20250916_185852.csv # SGLang benchmark results
```

## 📈 Key Findings

1. **vLLM is superior for production use** on RTX 5090
   - 15.7x better throughput at scale
   - Much lower latency
   - Simpler configuration

2. **SGLang requires special configuration** for RTX 5090
   - Must use torch_native backend
   - Disable CUDA graphs and FlashInfer
   - Still functional but lower performance

3. **Both frameworks are stable**
   - 100% success rate in all tests
   - No crashes or failures
   - Consistent performance

## 🔗 Resources

- [Qwen3-8B Model](https://huggingface.co/Qwen/Qwen3-8B)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [NVIDIA RTX 5090 Specs](https://www.nvidia.com/rtx-5090/)

## 📅 Deployment Date

**Date**: September 16, 2025
**Environment**: Linux 6.15.10, Docker, NVIDIA RTX 5090
**Tested By**: Cloud-Linuxer Team

## 📞 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/Cloud-Linuxer/qwen-8b)
- Check troubleshooting section above
- Review test results in CSV files

---

**Note**: This deployment was specifically optimized for RTX 5090 with its unique compute capability 12.0. Different GPUs may require different configurations.