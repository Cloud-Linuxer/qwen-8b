#!/bin/bash

# Deploy Qwen3-8B with SGLang (Alternative - RTX 5090 Compatible)
# Special configuration required for RTX 5090 (Compute Capability 12.0)

echo "🚀 Deploying Qwen3-8B with SGLang..."
echo "⚠️  Using special configuration for RTX 5090 compatibility"

# Stop existing container if running
docker stop qwen3-8b-sglang 2>/dev/null || true
docker rm qwen3-8b-sglang 2>/dev/null || true

# Run SGLang container with RTX 5090 optimizations
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
  --disable-radix-cache \
  --attention-backend torch_native

echo "⏳ Waiting for SGLang server to start..."
echo "   Model download and initialization may take 5-10 minutes..."
echo "   Note: RTX 5090 requires special settings due to Compute Capability 12.0"

# Wait for server to be ready
for i in {1..60}; do
  if docker logs qwen3-8b-sglang 2>&1 | grep -q "Uvicorn running"; then
    echo "✅ SGLang server is ready!"
    echo "📡 API available at: http://localhost:8000"
    echo ""
    echo "📊 Test the deployment:"
    echo "   curl http://localhost:8000/v1/completions \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"model\": \"Qwen/Qwen3-8B\", \"prompt\": \"Hello, how are you?\", \"max_tokens\": 50}'"
    echo ""
    echo "📈 Expected Performance (RTX 5090):"
    echo "   - Single user: ~47 tok/s"
    echo "   - 10 users: ~158 tok/s"
    echo "   - 50 users: ~224 tok/s"
    echo "   - 100 users: ~240 tok/s"
    echo ""
    echo "💡 Note: vLLM provides significantly better performance for this hardware"
    exit 0
  fi
  echo "   Checking server status... ($i/60)"
  sleep 10
done

echo "❌ Server failed to start. Check logs with: docker logs qwen3-8b-sglang"
exit 1