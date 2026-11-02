# 배포본 검증 기록

검증일: 2026-11-02 · 현재 배포: **1.2.0**

**H200·Ironwood에서 직접 실행한 결과는 없습니다.** 사용자 GCP 자원을 생성하거나 변경하지 않았습니다. 아래의 현재 버전 검사와 v1.0의 실제 체크포인트 기록을 구분해 읽으세요.

## v1.2 — vLLM 방안과 API 도구

기존 JAX 모델 수식은 바꾸지 않았습니다. 표준 라이브러리 기반 `scripts/vllm_api_probe.py`와 vLLM 비교 문서를 추가했습니다.

| 검사 | 결과 | 해석 범위 |
|---|---|---|
| 전체 pytest | 24개 통과, skip 0개 | 기존 14개 + API 계약 검사 10개 |
| API HTTP 수집·반복 비교 | 통과 | 로컬 모형 HTTP 서버 사용, 실제 vLLM 아님 |
| 고정 token ID export | 통과 | 준비된 bundle의 정확한 prefix 추출 |
| 입력·응답 검증 | 통과 | prompt ID 누락/불일치, 문자열 후보, NaN, 여러 생성 토큰 거부 |
| 후보 비교 | 통과 | 다른 생성 토큰 점수를 직접 빼지 않음, 공통 후보 없는 경우 빈 값 |
| 환경·모델 기록 | 통과 | 테스트 가중치 파일 변경 시 SHA 변경 확인 |
| 문서·문법·의존성 | 통과 | 상대 링크, compileall, bash -n, pip check |

**실제 vLLM 서버, GPU·TPU, hidden state 추출과 Trace Replay는 실행하지 않았습니다.** worker adapter와 vLLM 커널 재실행은 설계 단계입니다. API 클라이언트를 레이어 내부 계측 완료 도구로 표현하지 않습니다. [v1.2 검사 기록](validation_evidence/v1_2_validation.json)을 참고하세요.

## v1.1 — Git 저장소 배포본 검사

Linux x86_64, Python 3.12.14의 새 가상환경에서 검사했습니다. JAX/JAXLIB는 0.12.0.1, NumPy는 2.5.4, 공식 reference는 PyTorch 2.6.0+cpu와 Transformers 4.51.3입니다.

| 검사 | 결과 | 확인한 범위 |
|---|---|---|
| `scripts/setup.sh cpu` | 성공 | 새 venv에서 editable CPU 설치, `pip check` |
| 설치 후 `doctor --backend cpu` | 성공 | CPU 장비 초기화, BF16 matmul |
| `pytest -q` | **14개 통과, skip 0개** | 작은 공식 Qwen2 reference 2개 포함 |
| 독립 CLI smoke | 성공 | 합성 2개 블록, 8토큰, BF16, 2회 반복 |
| smoke 반복 비교 | 24개 tensor에서 최대 절대 차이 0 | 이 CPU·합성 입력에 한함 |
| smoke 계측 영향 | 2개 검사에서 출력 차이 0 | 선택한 0번 블록의 두 반복 |
| smoke Q projection | FP64 대비 최대 절대 차이 약 3.46e-8 | 실제 입력을 BF16으로 만든 작은 합성 GEMM |
| 문서 상대 링크·버전 | 통과 | 링크 파일 존재와 패키지 버전 일관성 |
| Python·shell 문법 | 통과 | compileall, bash -n |
| wheel·소스 배포본 빌드 | 성공 | pyproject 기반 Python 패키징 |
| 독립 wheel 설치·실행 | 성공 | 새 venv에 wheel+CPU 의존성 설치, 소스 폴더 밖에서 version·doctor·smoke 확인 |

pytest는 11.06초에 완료됐습니다. 이는 이 환경의 검사 실행시간이며 모델의 추론 성능 측정값이 아닙니다.

이 버전은 실행 수식을 바꾸지 않고 패키징·CLI·입력 검증·문서를 정리했습니다. **v1.1에서 실제 0.5B 체크포인트 전체를 다시 실행하지는 않았습니다.** 아래 v1.0 결과를 v1.1 장비 검증으로 바꿔 표기하지 않습니다. 코드 hash가 바뀌므로 양 장비 실험은 같은 v1.1 버전으로 새로 실행합니다.

v1.1 검사 증거:

- [기계 판독 요약](validation_evidence/v1_1_cpu_validation.json)
- [CPU 개발·reference 환경 패키지 목록](CPU_DEV_ENV_V1_1.txt)

GitHub Actions workflow는 제공하지만 호스팅된 GitHub runner에서는 아직 실행하지 않았습니다. GKE YAML과 GPU·TPU 초기화도 사용자 장비에서 추가 검증해야 합니다.

## v1.0 — 실제 공개 체크포인트의 과거 CPU 검증

다음 기록과 기존 evidence JSON은 v1.0 배포본에서 생성했습니다. 합성 smoke와 달리 학습된 실제 Qwen2.5-0.5B 가중치를 사용한 CPU 검사입니다.

### 검증 환경

| 항목 | 값 |
|---|---|
| 실행 장치 | Linux x86_64 CPU |
| Python | 3.12.14 |
| JAX / JAXLIB | 0.12.0.1 / 0.12.0.1 |
| NumPy | 2.5.4 |
| 공식 구현 검증 | PyTorch 2.6.0+cpu / Transformers 4.51.3 |
| 실제 모델 | Qwen/Qwen2.5-0.5B-Instruct |
| 모델 revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| 실제 입력 | 제공한 3개 prompt의 앞부분, sample별 32토큰, padding 없음 |
| 전체 레이어 | 24개 |
| 상세 계측 | 0, 12, 23번 레이어 |

전체 모델은 공개된 실제 가중치를 사용했습니다. 작은 무작위 모델은 별도의 단위·통합 검사에만 사용했습니다.

### 완료한 검증

| 검사 | 결과 | 해석 범위 |
|---|---|---|
| 새 venv에서 `scripts/setup.sh cpu` | 성공 | CPU용 설치 경로 |
| 새 venv의 `doctor --backend cpu` | 성공 | JAX CPU 초기화 및 BF16 matmul |
| pytest | 7개 통과 | 지표, safetensors BF16 읽기, checksum, 구조 거부, 공식 Qwen2 수식 비교, 실행 파이프라인 |
| 실제 모델 FP32 전체 실행 | 3개 입력 완료 | 24개 블록과 logits 저장 |
| 실제 모델 BF16 전체 실행 | 3개 입력 × 2회 완료 | 반복 실행과 상세 계측 |
| 공식 Transformers와 FP32 비교 | 75개 tensor 검사 통과 | 24개 블록 출력+logits × 3개 입력 |
| 공식 구현과 logits Top-1 | 각 입력에서 100% 일치 | 고정된 32개 위치의 다음 토큰 점수, 일반 품질 보장 아님 |
| BF16 CPU 반복 비교 | 318개 저장 tensor에서 최대 절대 차이 0 | 이 CPU·입력·구성의 두 번 반복에 한함 |
| 12번 블록 동일 입력 재실행 | 3개 입력 × 2회 완료 | 블록 내부 tensor 저장 |
| 재실행 반복 비교 | 60개 tensor에서 최대 절대 차이 0 | 이 CPU 실험에 한함 |
| Q projection GEMM 분리 | 3개 입력 × 2회 완료 | bias 없는 행렬곱과 CPU FP64 기준 비교 |
| 보고서 생성 | CSV, Markdown, PNG 생성 확인 | CPU FP32↔BF16 예시 포함 |

공식 Transformers 대비 FP32 최대 절대 차이는 `8.544921875e-4`였습니다. 이식 검사에는 `atol=5e-4`, `rtol=5e-4`의 결합 기준을 사용했습니다. 최대 절대 차이가 atol보다 크더라도 상대 허용항을 포함한 기준을 만족할 수 있습니다. 이는 일반 BF16 GPU·TPU 합격 기준이 아닙니다.

Q projection의 FP32 출력과 CPU FP64 기준값 간 최대 절대 차이는 약 `1.20046e-5`, 가장 큰 정규화 L2 차이는 약 `3.35759e-7`이었습니다. **이 숫자는 CPU의 BF16 입력 행렬곱 검사 결과이며 GPU·TPU 차이 수치가 아닙니다.**

BF16 상세 계측 18회에서 출력만 반환하는 블록과 중간값을 함께 반환하는 블록의 출력 차이는 0이었습니다. 다른 장비나 컴파일러에서 같은 결과라고 가정하지 않고 매번 다시 검사합니다.

### 동봉 증거

- `docs/validation_evidence/hf_validation.json`: 공식 구현 검증 요약.
- `docs/validation_evidence/hf_reference_metrics.json`: 75개 tensor 상세 지표.
- `docs/validation_evidence/cpu_repeat_summary.json`: 동일 CPU 반복 비교 요약.
- `docs/validation_evidence/cpu_replay_summary.json`: 동일 입력 재실행 비교 요약.
- `docs/validation_evidence/cpu_gemm_reference_metrics.json`: CPU GEMM과 FP64 기준 비교.
- `docs/validation_evidence/cpu_instrumentation_checks.json`: 계측 영향 검사.
- `docs/CPU_INSTALL_LOCK.txt`: 새 CPU venv에서 설치된 패키지 목록.
- `examples/CPU_FP32_vs_BF16/`: 실제 CPU FP32·BF16 비교 보고서와 그래프. GPU·TPU 보고서로 사용하면 안 됩니다.

가중치, 원시 activation dump, venv는 ZIP에 포함하지 않았습니다. 같은 모델 revision과 명령으로 재생성할 수 있습니다.

### 사용자의 장비에서 추가로 확인할 것

1. H200의 CUDA 드라이버·JAX plugin 초기화.
2. Ironwood의 libtpu 초기화 및 실제 device 이름.
3. 사용자 GKE 노드풀에서 Pod 스케줄링과 권한·이미지 정책.
4. 두 장비에서 같은 bundle·정밀도·JAX/JAXLIB 버전으로 실행.
5. 128/512/2,048토큰과 실제 업무 입력으로 확장.
6. 운영 vLLM 커널, KV cache, 분산 통신, MoE가 필요하면 별도 계측 adapter를 구현해 검증.

TPU용 JAX dependency 해석에서는 `jax[tpu]==0.12.0.1`이 `libtpu==0.0.50.1`을 선택하는 것을 확인했습니다. 이 dependency 해석은 Ironwood에서의 실제 실행 검증을 대신하지 않습니다.
