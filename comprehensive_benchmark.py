#!/usr/bin/env python3
"""
Comprehensive Benchmark for Qwen3-8B: vLLM vs SGLang
Tests single user, multiple users, and monitors VRAM usage
"""

import requests
import time
import asyncio
import aiohttp
import subprocess
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple

def get_gpu_memory_usage() -> Dict:
    """Get current GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': int(values[0]),
                'memory_total_mb': int(values[1]),
                'memory_free_mb': int(values[2]),
                'gpu_utilization': int(values[3]),
                'memory_used_gb': round(int(values[0]) / 1024, 2),
                'memory_total_gb': round(int(values[1]) / 1024, 2),
                'memory_free_gb': round(int(values[2]) / 1024, 2)
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return {}

def test_single_user(port: int, server_name: str, max_tokens: int = 500) -> Dict:
    """Test single user performance with VRAM monitoring"""
    url = f"http://localhost:{port}/v1/completions"

    prompt = """Write a detailed analysis of artificial intelligence impact on society,
    covering economic, social, and ethical aspects. Be comprehensive and thorough."""

    payload = {
        "model": "Qwen/Qwen3-8B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }

    print(f"\n{'='*60}")
    print(f"Testing {server_name} - SINGLE USER")
    print(f"{'='*60}")

    # Get initial GPU memory
    initial_gpu = get_gpu_memory_usage()
    print(f"📊 Initial VRAM: {initial_gpu.get('memory_used_gb', 'N/A')} GB / {initial_gpu.get('memory_total_gb', 'N/A')} GB")

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        end_time = time.time()

        # Get GPU memory during inference
        inference_gpu = get_gpu_memory_usage()

        result = response.json()
        total_time = end_time - start_time

        # Extract metrics
        completion_tokens = result.get('usage', {}).get('completion_tokens', 0)
        prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

        print(f"✅ Request completed")
        print(f"   Time: {total_time:.2f}s")
        print(f"   Tokens: {completion_tokens}")
        print(f"   Speed: {tokens_per_second:.2f} tok/s")
        print(f"   VRAM during inference: {inference_gpu.get('memory_used_gb', 'N/A')} GB")
        print(f"   VRAM increase: {inference_gpu.get('memory_used_gb', 0) - initial_gpu.get('memory_used_gb', 0):.2f} GB")

        return {
            'server': server_name,
            'test_type': 'single_user',
            'total_time': total_time,
            'tokens': completion_tokens,
            'speed_tok_s': tokens_per_second,
            'vram_initial_gb': initial_gpu.get('memory_used_gb', 0),
            'vram_inference_gb': inference_gpu.get('memory_used_gb', 0),
            'vram_increase_gb': inference_gpu.get('memory_used_gb', 0) - initial_gpu.get('memory_used_gb', 0),
            'gpu_utilization': inference_gpu.get('gpu_utilization', 0)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

async def concurrent_request(session, url, payload, request_id):
    """Make a single async request"""
    try:
        start_time = time.time()
        async with session.post(url, json=payload) as response:
            result = await response.json()
            end_time = time.time()

            return {
                'request_id': request_id,
                'success': True,
                'time': end_time - start_time,
                'tokens': result.get('usage', {}).get('completion_tokens', 0)
            }
    except Exception as e:
        return {
            'request_id': request_id,
            'success': False,
            'error': str(e)
        }

async def test_multiple_users(port: int, server_name: str, num_users: int = 10, max_tokens: int = 200) -> Dict:
    """Test multiple concurrent users with VRAM monitoring"""
    url = f"http://localhost:{port}/v1/completions"

    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
        "Describe the water cycle process.",
        "What causes climate change?",
        "Explain blockchain technology.",
        "How do vaccines work?",
        "What is dark matter?",
        "Describe photosynthesis.",
        "How does the internet work?"
    ] * (num_users // 10 + 1)

    print(f"\n{'='*60}")
    print(f"Testing {server_name} - {num_users} CONCURRENT USERS")
    print(f"{'='*60}")

    # Get initial GPU memory
    initial_gpu = get_gpu_memory_usage()
    print(f"📊 Initial VRAM: {initial_gpu.get('memory_used_gb', 'N/A')} GB")

    # Create tasks
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_users):
            payload = {
                "model": "Qwen/Qwen3-8B",
                "prompt": prompts[i % len(prompts)],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            tasks.append(concurrent_request(session, url, payload, i))

        # Start all requests
        print(f"🚀 Sending {num_users} concurrent requests...")
        start_time = time.time()

        # Monitor GPU during requests
        max_vram = initial_gpu.get('memory_used_gb', 0)

        # Execute all requests
        results = await asyncio.gather(*tasks)

        # Get peak GPU memory
        peak_gpu = get_gpu_memory_usage()
        max_vram = max(max_vram, peak_gpu.get('memory_used_gb', 0))

        end_time = time.time()
        total_time = end_time - start_time

    # Analyze results
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    if successful:
        total_tokens = sum(r.get('tokens', 0) for r in successful)
        avg_response_time = sum(r.get('time', 0) for r in successful) / len(successful)
        throughput = total_tokens / total_time if total_time > 0 else 0

        print(f"✅ Completed {len(successful)}/{num_users} requests")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Throughput: {throughput:.2f} tok/s")
        print(f"   Avg response time: {avg_response_time:.2f}s")
        print(f"   Peak VRAM: {max_vram:.2f} GB")
        print(f"   VRAM increase: {max_vram - initial_gpu.get('memory_used_gb', 0):.2f} GB")

        if failed:
            print(f"   ⚠️ Failed requests: {len(failed)}")

        return {
            'server': server_name,
            'test_type': f'concurrent_{num_users}_users',
            'num_users': num_users,
            'successful_requests': len(successful),
            'failed_requests': len(failed),
            'total_time': total_time,
            'total_tokens': total_tokens,
            'throughput_tok_s': throughput,
            'avg_response_time': avg_response_time,
            'vram_initial_gb': initial_gpu.get('memory_used_gb', 0),
            'vram_peak_gb': max_vram,
            'vram_increase_gb': max_vram - initial_gpu.get('memory_used_gb', 0)
        }
    else:
        print(f"❌ All requests failed")
        return None

def run_comprehensive_benchmark(port: int, server_name: str) -> List[Dict]:
    """Run complete benchmark suite"""
    results = []

    print(f"\n{'#'*60}")
    print(f"# COMPREHENSIVE BENCHMARK: {server_name}")
    print(f"{'#'*60}")

    # Single user test
    print("\n1️⃣ Single User Test")
    single_result = test_single_user(port, server_name, max_tokens=500)
    if single_result:
        results.append(single_result)
    time.sleep(5)  # Cool down

    # Multiple users tests
    for num_users in [5, 10, 20, 50]:
        print(f"\n{num_users}️⃣ Testing {num_users} Concurrent Users")
        multi_result = asyncio.run(test_multiple_users(port, server_name, num_users, max_tokens=200))
        if multi_result:
            results.append(multi_result)
        time.sleep(5)  # Cool down between tests

    return results

def save_results(all_results: List[Dict], filename: str):
    """Save results to CSV file"""
    if not all_results:
        return

    # CSV output
    csv_path = f"/home/qwen-8b-repo/{filename}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['server', 'test_type', 'num_users', 'speed_tok_s', 'throughput_tok_s',
                     'vram_initial_gb', 'vram_peak_gb', 'vram_increase_gb',
                     'total_time', 'successful_requests', 'failed_requests']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            row = {k: result.get(k, '') for k in fieldnames}
            # Use speed for single user, throughput for multi
            if result['test_type'] == 'single_user':
                row['throughput_tok_s'] = ''
                row['num_users'] = 1
            else:
                row['speed_tok_s'] = ''
            writer.writerow(row)

    print(f"\n💾 Results saved to: {csv_path}")

def print_comparison(sglang_results: List[Dict], vllm_results: List[Dict]):
    """Print comparison between SGLang and vLLM"""
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON: SGLang vs vLLM")
    print(f"{'='*80}")

    # Create comparison table
    print("\n| Test | SGLang Speed | vLLM Speed | SGLang VRAM | vLLM VRAM | Winner |")
    print("|------|-------------|------------|-------------|-----------|---------|")

    test_types = ['single_user', 'concurrent_5_users', 'concurrent_10_users',
                  'concurrent_20_users', 'concurrent_50_users']

    for test_type in test_types:
        sg = next((r for r in sglang_results if r['test_type'] == test_type), None)
        vl = next((r for r in vllm_results if r['test_type'] == test_type), None)

        if sg and vl:
            if test_type == 'single_user':
                sg_speed = sg.get('speed_tok_s', 0)
                vl_speed = vl.get('speed_tok_s', 0)
            else:
                sg_speed = sg.get('throughput_tok_s', 0)
                vl_speed = vl.get('throughput_tok_s', 0)

            sg_vram = sg.get('vram_peak_gb', sg.get('vram_inference_gb', 0))
            vl_vram = vl.get('vram_peak_gb', vl.get('vram_inference_gb', 0))

            winner = "vLLM" if vl_speed > sg_speed else "SGLang"

            test_label = test_type.replace('_', ' ').title()
            print(f"| {test_label:<20} | {sg_speed:>11.2f} | {vl_speed:>10.2f} | {sg_vram:>11.2f} | {vl_vram:>9.2f} | {winner:<7} |")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test current server (SGLang is running on port 8000)
    print("🔍 Testing SGLang first...")
    sglang_results = run_comprehensive_benchmark(8000, "SGLang")
    save_results(sglang_results, f"sglang_benchmark_{timestamp}")

    print("\n" + "="*60)
    print("⏸️ Stopping SGLang and starting vLLM...")
    print("="*60)

    # Stop SGLang
    subprocess.run(["docker", "stop", "qwen3-8b-sglang"], capture_output=True)
    subprocess.run(["docker", "rm", "qwen3-8b-sglang"], capture_output=True)
    time.sleep(5)

    # Start vLLM
    subprocess.run([
        "docker", "run", "-d",
        "--name", "qwen3-8b-vllm",
        "--runtime", "nvidia",
        "--gpus", "all",
        "-p", "8000:8000",
        "-v", "~/.cache/huggingface:/root/.cache/huggingface",
        "vllm/vllm-openai:latest",
        "--model", "Qwen/Qwen3-8B",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.95",
        "--dtype", "auto",
        "--trust-remote-code"
    ])

    print("⏳ Waiting 90 seconds for vLLM to initialize...")
    time.sleep(90)

    # Test vLLM
    print("\n🔍 Testing vLLM...")
    vllm_results = run_comprehensive_benchmark(8000, "vLLM")
    save_results(vllm_results, f"vllm_benchmark_{timestamp}")

    # Combined results
    all_results = sglang_results + vllm_results
    save_results(all_results, f"combined_benchmark_{timestamp}")

    # Print comparison
    print_comparison(sglang_results, vllm_results)