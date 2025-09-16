#!/usr/bin/env python3
"""
vLLM Multilingual Test - Force generation in Chinese, Korean, and English
"""

import requests
import time
import json
from datetime import datetime

def test_vllm_multilingual(max_tokens=3000):
    """Test vLLM with explicit multilingual prompts"""
    url = "http://localhost:8000/v1/completions"

    # More explicit multilingual prompt
    prompt = """Write a long poem about the four seasons and life philosophy. You MUST write in THREE languages:
1. Chinese (中文)
2. Korean (한국어)
3. English

Format EXACTLY like this:

【春天 Spring 봄】

中文诗句：
春风拂面暖人心，
万物复苏展新颜。
花开满园香四溢，
燕子归来筑新巢。

Korean verse (한국어 시):
봄바람이 불어와서
새로운 생명이 피어나고
꽃들이 만발하여
향기가 가득합니다

English verse:
Spring breeze gently touches my face,
Nature awakens from winter's embrace.
Flowers bloom in vibrant array,
New life begins this beautiful day.

【夏天 Summer 여름】

中文诗句：
"""

    payload = {
        "model": "Qwen/Qwen3-8B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.9,
        "top_p": 0.95,
        "stream": False
    }

    print("="*60)
    print("Testing vLLM with Explicit Multilingual Prompt")
    print("="*60)
    print(f"Requesting {max_tokens} tokens...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        end_time = time.time()
        total_time = end_time - start_time

        result = response.json()
        generated_text = result['choices'][0]['text']

        # Token counts
        prompt_tokens = result.get('usage', {}).get('prompt_tokens', 0)
        completion_tokens = result.get('usage', {}).get('completion_tokens', 0)
        total_tokens = result.get('usage', {}).get('total_tokens', 0)

        # Speed calculation
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

        print(f"\n📊 Performance Metrics:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   Prompt Tokens: {prompt_tokens}")
        print(f"   Completion Tokens: {completion_tokens}")
        print(f"   Total Tokens: {total_tokens}")
        print(f"   Speed: {tokens_per_second:.2f} tokens/second")

        # Display generated text sample
        print(f"\n📝 Generated Text Sample (first 1000 chars):")
        print("-" * 40)
        print(generated_text[:1000])
        if len(generated_text) > 1000:
            print("...")
            print(f"\n[Total generated: {len(generated_text)} characters]")
        print("-" * 40)

        # Language detection
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in generated_text)
        has_korean = any('\uac00' <= char <= '\ud7af' for char in generated_text)
        has_english = any('a' <= char.lower() <= 'z' for char in generated_text)

        print(f"\n🌍 Language Detection:")
        print(f"   Chinese: {'✅' if has_chinese else '❌'}")
        print(f"   Korean: {'✅' if has_korean else '❌'}")
        print(f"   English: {'✅' if has_english else '❌'}")

        # Save full result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/home/qwen-8b-repo/vllm_multilingual_{timestamp}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"vLLM Multilingual Test Results\n")
            f.write(f"Time: {timestamp}\n")
            f.write(f"Speed: {tokens_per_second:.2f} tok/s\n")
            f.write(f"Languages detected - CN: {has_chinese}, KR: {has_korean}, EN: {has_english}\n")
            f.write(f"\n{'='*50}\n\n")
            f.write(generated_text)

        print(f"\n💾 Full output saved to: {output_file}")

        return {
            'speed': tokens_per_second,
            'time': total_time,
            'tokens': completion_tokens,
            'has_chinese': has_chinese,
            'has_korean': has_korean,
            'has_english': has_english
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Test with explicit prompt
    result = test_vllm_multilingual(max_tokens=3000)

    if result:
        print(f"\n{'='*60}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Speed: {result['speed']:.2f} tokens/second")
        print(f"Time: {result['time']:.2f} seconds")
        print(f"Tokens: {result['tokens']}")
        print(f"Multilingual Success: ", end="")
        if result['has_chinese'] and result['has_korean'] and result['has_english']:
            print("✅ All three languages generated!")
        else:
            missing = []
            if not result['has_chinese']: missing.append("Chinese")
            if not result['has_korean']: missing.append("Korean")
            if not result['has_english']: missing.append("English")
            print(f"⚠️ Missing: {', '.join(missing)}")