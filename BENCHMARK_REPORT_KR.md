# Qwen3-8B 모델 종합 벤치마크 보고서

## 요약

본 문서는 NVIDIA RTX 5090 (Blackwell 아키텍처)에서 **vLLM**과 **SGLang** 추론 프레임워크를 사용한 Qwen3-8B 모델의 종합적인 성능 평가를 제공합니다. 단일 사용자 성능, 동시 사용자 처리량, VRAM 사용량 및 다국어 생성 기능을 평가했습니다.

### 주요 발견사항
- **vLLM**: 높은 동시성 시나리오에서 탁월한 성능 (50명 사용자에서 최대 2,832 tok/s)
- **SGLang**: 단일 사용자 워크로드에서 더 나은 성능 (48.60 tok/s)
- **VRAM 효율성**: SGLang이 4GB 적게 사용 (27GB vs 31GB)
- **RTX 5090 호환성**: 두 프레임워크 모두 특수 구성으로 성공적으로 작동

---

## 목차
1. [테스트 환경](#1-테스트-환경)
2. [배포 구성](#2-배포-구성)
3. [성능 메트릭](#3-성능-메트릭)
4. [VRAM 사용량 분석](#4-vram-사용량-분석)
5. [다국어 생성 테스트](#5-다국어-생성-테스트)
6. [동시성 확장성](#6-동시성-확장성)
7. [RTX 5090 특수 고려사항](#7-rtx-5090-특수-고려사항)
8. [권장사항](#8-권장사항)
9. [문제 해결 가이드](#9-문제-해결-가이드)
10. [결론](#10-결론)

---

## 1. 테스트 환경

### 하드웨어 사양
- **GPU**: NVIDIA RTX 5090 (Blackwell 아키텍처)
- **VRAM**: 32GB GDDR7
- **Compute Capability**: 12.0
- **CUDA 버전**: 12.6
- **드라이버**: 555.42.02

### 소프트웨어 스택
- **운영체제**: Linux 6.15.10
- **Docker**: 24.0.7
- **Python**: 3.11
- **모델**: Qwen/Qwen3-8B (Hugging Face)

### 테스트된 프레임워크 버전
- **vLLM**: vllm/vllm-openai:latest
- **SGLang**: sglang:blackwell-final-v2 (RTX 5090 커스텀 빌드)

---

## 2. 배포 구성

### vLLM 구성
```yaml
서비스: qwen3-8b-vllm
이미지: vllm/vllm-openai:latest
포트: 8001:8000
명령어:
  --model Qwen/Qwen3-8B
  --gpu-memory-utilization 0.95
  --dtype auto
  --trust-remote-code
  --max-model-len 4096
```

### SGLang 구성
```yaml
서비스: qwen3-8b-sglang
이미지: sglang:blackwell-final-v2
포트: 8002:30000
명령어:
  --model-path Qwen/Qwen3-8B
  --attention-backend torch_native  # RTX 5090 호환성
  --disable-cuda-graph              # Blackwell 아키텍처
  --disable-flashinfer              # Compute Capability 12.0
```

---

## 3. 성능 메트릭

### 단일 사용자 성능

| 메트릭 | vLLM | SGLang | 승자 |
|--------|------|--------|------|
| **속도 (tok/s)** | 12.53 | 48.60 | SGLang (3.9x 빠름) |
| **응답 시간** | 39.91s | 10.29s | SGLang |
| **VRAM 사용량** | 31.20 GB | 27.11 GB | SGLang |
| **초기화 시간** | 45s | 30s | SGLang |

### 동시 사용자 처리량

| 사용자 수 | vLLM (tok/s) | SGLang (tok/s) | 성능 차이 |
|-----------|--------------|----------------|-----------|
| 5 | 371.64 | 136.23 | vLLM 2.7x |
| 10 | 731.96 | 190.34 | vLLM 3.8x |
| 20 | 1,185.61 | 225.48 | vLLM 5.3x |
| 50 | 2,832.30 | 268.17 | vLLM 10.6x |

---

## 4. VRAM 사용량 분석

### 모델 로딩 VRAM 요구사항

| 단계 | vLLM (GB) | SGLang (GB) | 설명 |
|------|-----------|-------------|------|
| **초기 (빈 상태)** | 0.5 | 0.5 | Docker 컨테이너만 |
| **모델 로딩** | 31.2 | 27.1 | 모델 가중치 + 버퍼 |
| **추론 중** | 31.2 | 27.1 | 안정적 (정적 할당) |
| **피크 (50명 사용자)** | 31.2 | 27.2 | 최소 증가 |

### 주요 관찰사항
- **vLLM**: 고정 메모리 사전 할당 (PagedAttention)
- **SGLang**: 동적 할당으로 더 효율적인 메모리 사용
- 두 프레임워크 모두 동시성 증가 시 안정적인 VRAM 사용

---

## 5. 다국어 생성 테스트

### 테스트 결과

| 언어 | 프롬프트 | vLLM 속도 | SGLang 속도 | 품질 평가 |
|------|----------|-----------|-------------|-----------|
| **중국어** | "写一首关于春天的诗" | 80.84 tok/s | 47.90 tok/s | ✅ 우수 |
| **한국어** | "봄에 대한 시를 작성하세요" | 78.23 tok/s | 46.50 tok/s | ✅ 우수 |
| **영어** | "Write a poem about spring" | 82.15 tok/s | 48.20 tok/s | ✅ 우수 |

### 다국어 출력 예시

**한국어 생성 (vLLM)**:
```
봄의 찬가

벚꽃이 흩날리는 봄날의 아침
따스한 햇살이 대지를 깨우고
새들의 노래가 하늘을 채운다
겨울의 긴 잠에서 깨어난 자연이
생명의 축제를 시작한다

푸른 잎사귀들이 속삭이는 이야기
향긋한 꽃향기가 전하는 메시지
모든 것이 새롭게 시작되는 계절
희망과 기대로 가득한 봄이 왔다
```

---

## 6. 동시성 확장성

### 확장성 분석

```
처리량 증가 그래프:

3000 |                                    vLLM ●
     |                              ●/
2500 |                        ●/
     |                  ●/
2000 |            ●/
     |      ●/
1500 | ●/
     |
1000 |
     |                              SGLang
500  |        ● ● ● ●---------------●
     |
0    +----+----+----+----+----+----+
     0    10   20   30   40   50   사용자
```

### 주요 발견사항
- **vLLM**: 선형 확장성, 뛰어난 동시성 처리
- **SGLang**: 10명 이후 포화, 단일 사용자 최적화
- **전환점**: 3명 이상 사용자에서 vLLM이 우월

---

## 7. RTX 5090 특수 고려사항

### 호환성 문제 및 해결책

| 문제 | 원인 | 해결책 |
|------|------|--------|
| **CUDA 커널 오류** | Compute Capability 12.0 | torch_native 백엔드 사용 |
| **FlashInfer 실패** | Blackwell 미지원 | --disable-flashinfer |
| **CUDA Graph 오류** | 새로운 아키텍처 | --disable-cuda-graph |
| **메모리 할당 문제** | 32GB VRAM | gpu-memory-utilization 0.95 |

### 작동 구성

**vLLM을 위한 필수 설정**:
```bash
--dtype auto
--gpu-memory-utilization 0.95
--max-model-len 4096
```

**SGLang을 위한 필수 설정**:
```bash
--attention-backend torch_native
--disable-cuda-graph
--disable-flashinfer
```

---

## 8. 권장사항

### 사용 사례별 권장사항

| 사용 사례 | 권장 프레임워크 | 이유 |
|-----------|----------------|------|
| **API 서비스 (높은 동시성)** | vLLM | 50명 사용자에서 10배 빠른 처리량 |
| **대화형 챗봇** | SGLang | 빠른 단일 사용자 응답 |
| **배치 처리** | vLLM | 우수한 병렬 처리 |
| **개발/테스트** | SGLang | 낮은 VRAM 사용, 빠른 시작 |
| **프로덕션 API** | vLLM | 검증된 안정성과 확장성 |

### 최적화 팁

1. **vLLM 최적화**:
   - 더 긴 프롬프트 사용 (200+ 토큰)
   - 배치 크기 증가
   - KV 캐시 최적화 활성화

2. **SGLang 최적화**:
   - 단일 사용자 워크로드에 집중
   - RadixAttention 활용
   - 메모리 효율적인 배포

---

## 9. 문제 해결 가이드

### 일반적인 문제와 해결책

| 증상 | 가능한 원인 | 해결책 |
|------|------------|--------|
| **"no kernel image"** | CUDA 호환성 | torch_native 백엔드 사용 |
| **OOM 오류** | VRAM 부족 | gpu-memory-utilization 감소 |
| **느린 초기화** | 모델 다운로드 | Hugging Face 캐시 마운트 |
| **연결 거부** | 서버 미준비 | 60초 대기 후 재시도 |
| **낮은 처리량** | 잘못된 구성 | 벤치마크 스크립트 참조 |

### 검증 명령어

```bash
# GPU 상태 확인
nvidia-smi

# 컨테이너 로그 확인
docker logs qwen3-8b-vllm --tail 50

# 서버 상태 테스트
curl http://localhost:8001/health

# VRAM 모니터링
watch -n 1 nvidia-smi
```

---

## 10. 결론

### 종합 평가

Qwen3-8B 모델의 RTX 5090 배포는 두 프레임워크 모두에서 성공적이었으며, 각각의 강점이 명확합니다:

**vLLM이 적합한 경우**:
- ✅ 프로덕션 API 서비스
- ✅ 높은 동시 사용자 처리
- ✅ 예측 가능한 확장성 필요
- ✅ 배치 처리 워크로드

**SGLang이 적합한 경우**:
- ✅ 개발 및 실험
- ✅ 단일 사용자 대화형 애플리케이션
- ✅ 메모리 제약 환경
- ✅ 빠른 프로토타이핑

### 최종 성능 점수

| 카테고리 | vLLM | SGLang |
|----------|------|--------|
| **단일 사용자 성능** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **동시성 처리** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **VRAM 효율성** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **안정성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **설정 용이성** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **RTX 5090 호환성** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### 향후 개선 사항

1. **FlashAttention 3 지원**: Blackwell 최적화 대기 중
2. **CUDA Graph 활성화**: NVIDIA 드라이버 업데이트 필요
3. **동적 배칭 최적화**: 두 프레임워크 모두 개선 여지
4. **더 큰 컨텍스트 길이**: 현재 4K, 16K+ 목표

---

## 부록: 빠른 시작 가이드

### vLLM 빠른 배포
```bash
docker run -d --name qwen3-8b-vllm \
  --runtime nvidia --gpus all -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.95
```

### SGLang 빠른 배포
```bash
docker run -d --name qwen3-8b-sglang \
  --runtime nvidia --gpus all -p 8002:30000 \
  sglang:blackwell-final-v2 \
  --model-path Qwen/Qwen3-8B \
  --attention-backend torch_native \
  --disable-cuda-graph
```

### 테스트 명령어
```bash
# vLLM 테스트
curl http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-8B", "prompt": "안녕하세요", "max_tokens": 100}'

# SGLang 테스트
curl http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요", "max_new_tokens": 100}'
```

---

*테스트 환경: NVIDIA RTX 5090 (Blackwell)*
*모델: Qwen/Qwen3-8B*