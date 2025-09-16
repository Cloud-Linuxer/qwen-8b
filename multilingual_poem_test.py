#!/usr/bin/env python3
"""
Multilingual Poem Generation Test for Qwen3-8B
Tests both vLLM and SGLang with Chinese, Korean, and English poetry
"""

import requests
import time
import json
import sys
from datetime import datetime

def test_poem_generation(port, server_name, max_tokens=2048):
    """
    Test poem generation with multilingual prompt
    """
    url = f"http://localhost:{port}/v1/completions"

    # Multilingual poem prompt - requesting long poem in Chinese, Korean, and English
    prompt = """请创作一首超长的诗歌，至少1000字，交替使用中文、韩语和英语。主题是"四季之美与人生哲理"。

请按照以下格式创作：
1. 春天 - 中文诗句（描述春天的美景和新生）
2. 봄 - 한국어 시구 (봄의 아름다움과 희망)
3. Spring - English verses (beauty of spring and renewal)
4. 夏天 - 中文诗句（描述夏天的热情和成长）
5. 여름 - 한국어 시구 (여름의 열정과 성장)
6. Summer - English verses (passion and growth of summer)
7. 秋天 - 中文诗句（描述秋天的收获和感恩）
8. 가을 - 한국어 시구 (가을의 수확과 감사)
9. Autumn - English verses (harvest and gratitude)
10. 冬天 - 中文诗句（描述冬天的沉思和智慧）
11. 겨울 - 한국어 시구 (겨울의 사색과 지혜)
12. Winter - English verses (reflection and wisdom)

继续创作，让诗歌尽可能长而优美，包含深刻的人生哲理。每个季节都要有多个段落，充分展现语言之美：

春天 - 万物复苏的季节：
"""

    payload = {
        "model": "Qwen/Qwen3-8B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False
    }

    print(f"\n{'='*60}")
    print(f"Testing {server_name} (Port {port})")
    print(f"{'='*60}")
    print(f"Requesting {max_tokens} tokens of multilingual poetry...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Measure request time
        start_time = time.time()

        # Make request
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time

        # Parse response
        result = response.json()
        generated_text = result['choices'][0]['text']

        # Get token counts
        prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)
        completion_tokens = result.get('usage', {}).get('completion_tokens', 0)
        total_tokens = result.get('usage', {}).get('total_tokens', 0)

        # Calculate speed
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

        # Display results
        print(f"\n📊 Performance Metrics:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Prompt Tokens: {prompt_tokens}")
        print(f"   Completion Tokens: {completion_tokens}")
        print(f"   Total Tokens: {total_tokens}")
        print(f"   Speed: {tokens_per_second:.2f} tokens/second")

        # Show sample of generated poem
        print(f"\n📝 Generated Poem Sample (first 500 chars):")
        print("-" * 40)
        print(generated_text[:500])
        if len(generated_text) > 500:
            print("...")
            print(f"\n[Total generated text length: {len(generated_text)} characters]")
        print("-" * 40)

        # Check for multilingual content
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in generated_text)
        has_korean = any('\uac00' <= char <= '\ud7af' for char in generated_text)
        has_english = any('a' <= char.lower() <= 'z' for char in generated_text)

        print(f"\n🌍 Language Detection:")
        print(f"   Chinese: {'✅' if has_chinese else '❌'}")
        print(f"   Korean: {'✅' if has_korean else '❌'}")
        print(f"   English: {'✅' if has_english else '❌'}")

        return {
            'server': server_name,
            'port': port,
            'total_time': total_time,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'tokens_per_second': tokens_per_second,
            'text_length': len(generated_text),
            'has_chinese': has_chinese,
            'has_korean': has_korean,
            'has_english': has_english
        }

    except requests.exceptions.Timeout:
        print(f"❌ Request timeout after 300 seconds")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    """Main test function"""
    results = []

    # Check which servers are running
    print("🔍 Checking available servers...")

    servers_to_test = []

    # Check vLLM (port 8000)
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            servers_to_test.append(("vLLM", 8000))
            print("   ✅ vLLM server found on port 8000")
    except:
        print("   ❌ vLLM server not available on port 8000")

    # Check SGLang (port 8001)
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            servers_to_test.append(("SGLang", 8001))
            print("   ✅ SGLang server found on port 8001")
    except:
        print("   ❌ SGLang server not available on port 8001")

    if not servers_to_test:
        print("\n❌ No servers available for testing!")
        print("   Please start vLLM on port 8000 or SGLang on port 8001")
        return

    # Test each available server
    for server_name, port in servers_to_test:
        result = test_poem_generation(port, server_name, max_tokens=2048)
        if result:
            results.append(result)

    # Compare results if we have multiple servers
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📊 COMPARISON RESULTS")
        print(f"{'='*60}")

        for result in results:
            print(f"\n{result['server']}:")
            print(f"   Speed: {result['tokens_per_second']:.2f} tok/s")
            print(f"   Total Time: {result['total_time']:.2f}s")
            print(f"   Tokens Generated: {result['completion_tokens']}")
            print(f"   Multilingual: ", end="")
            if result['has_chinese'] and result['has_korean'] and result['has_english']:
                print("✅ All languages present")
            else:
                print("⚠️ Some languages missing")

        # Determine winner
        if len(results) == 2:
            vllm_result = next((r for r in results if r['server'] == 'vLLM'), None)
            sglang_result = next((r for r in results if r['server'] == 'SGLang'), None)

            if vllm_result and sglang_result:
                speed_ratio = vllm_result['tokens_per_second'] / sglang_result['tokens_per_second']
                print(f"\n🏆 Performance Summary:")
                print(f"   vLLM is {speed_ratio:.2f}x faster than SGLang")

    # Save results to CSV
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"/home/qwen-8b-repo/multilingual_poem_test_{timestamp}.csv"

        with open(csv_filename, 'w') as f:
            # Write header
            f.write("server,port,total_time,prompt_tokens,completion_tokens,total_tokens,")
            f.write("tokens_per_second,text_length,has_chinese,has_korean,has_english\n")

            # Write results
            for result in results:
                f.write(f"{result['server']},{result['port']},{result['total_time']:.2f},")
                f.write(f"{result['prompt_tokens']},{result['completion_tokens']},{result['total_tokens']},")
                f.write(f"{result['tokens_per_second']:.2f},{result['text_length']},")
                f.write(f"{result['has_chinese']},{result['has_korean']},{result['has_english']}\n")

        print(f"\n💾 Results saved to: {csv_filename}")

if __name__ == "__main__":
    main()