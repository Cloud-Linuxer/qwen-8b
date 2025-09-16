#!/bin/bash

# Deploy Qwen3-8B with vLLM (Recommended - Best Performance)
# Achieves 3,770 tok/s with 100 concurrent users on RTX 5090

echo "🚀 Deploying Qwen3-8B with vLLM..."

# Stop existing container if running
docker stop qwen3-8b-vllm 2>/dev/null || true
docker rm qwen3-8b-vllm 2>/dev/null || true

# Run vLLM container
docker run -d \
  --name qwen3-8b-vllm \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --dtype auto \
  --trust-remote-code \
  --enable-prefix-caching

echo "⏳ Waiting for vLLM server to start..."
echo "   Model download and initialization may take 5-10 minutes..."

# Wait for server to be ready
for i in {1..60}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ vLLM server is ready!"
    echo "📡 API available at: http://localhost:8000"
    echo ""
    echo "📊 Test the deployment:"
    echo "   curl http://localhost:8000/v1/completions \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"model\": \"Qwen/Qwen3-8B\", \"prompt\": \"Hello, how are you?\", \"max_tokens\": 50}'"
    echo ""
    echo "📈 Expected Performance (RTX 5090):"
    echo "   - Single user: ~82 tok/s"
    echo "   - 10 users: ~719 tok/s"
    echo "   - 50 users: ~2,386 tok/s"
    echo "   - 100 users: ~3,770 tok/s"
    exit 0
  fi
  echo "   Checking server status... ($i/60)"
  sleep 10
done

echo "❌ Server failed to start. Check logs with: docker logs qwen3-8b-vllm"
exit 1