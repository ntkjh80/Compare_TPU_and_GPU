# GPU·TPU Layer Parity

같은 모델과 입력을 GPU·TPU에서 실행하고, **어느 레이어에서 수치 차이가 나타나는지** 검사하는 Python 도구입니다.

첫 대상은 `Qwen/Qwen2.5-0.5B-Instruct`입니다. 공통 JAX 진단 구현으로 24개 Transformer 블록, 선택한 블록 내부, 최종 logits를 비교합니다.

> 현재 범위: 단일 JAX device, batch=1, prefill, BF16/FP32, KV cache 없음. 운영 vLLM 커널·분산 통신·MoE 비교는 별도 구현이 필요합니다. H200·Ironwood 실측 완료를 의미하지 않습니다. [검증 기록](docs/VALIDATION_KO.md)을 확인하세요.

## 바로 시작하기

Python 3.12가 있는 Linux 또는 macOS에서 저장소 루트로 이동합니다.

```bash
bash scripts/setup.sh cpu
source .venv/bin/activate
layer-parity --version
layer-parity doctor --backend cpu
layer-parity smoke --out runs/cpu-smoke
```

마지막 명령은 **작은 합성 가중치**로 전체 실행·레이어 재실행·행렬곱·보고서 생성을 확인합니다. 설치 후에는 모델 다운로드나 GPU가 필요하지 않습니다. 실제 Qwen 정확도나 GPU·TPU 차이를 측정한 결과는 아닙니다.

vLLM 환경에서는 [vLLM 비교 안내](docs/VLLM_KO.md)를 사용합니다. 고정 입력 API 클라이언트를 추가했으며, vLLM 레이어 내부 계측은 backend별 추가 구현이 필요합니다.

실제 모델로 실행하려면 [빠른 실행 안내](docs/QUICKSTART_KO.md)를 따르세요. GPU는 `bash scripts/setup.sh gpu`, TPU는 `bash scripts/setup.sh tpu`로 설치합니다. 서로 다른 호스트에서 각각 수행합니다.

## 제공하는 기능

| 명령 | 역할 |
|---|---|
| `doctor` | 요청한 실제 장비에서 행렬곱 실행. CPU 대체 실행 방지 |
| `download` | 공개 체크포인트의 commit SHA를 고정하여 다운로드 |
| `prepare` | 가중치·token ID·체크섬을 묶은 공통 입력 생성 |
| `run` | 모든 블록의 입력·출력, 선택한 블록의 내부 값, logits 저장 |
| `replay` | 양쪽에 동일한 블록 입력을 주어 해당 블록 자체의 차이 검사 |
| `matmul` | 실제 Q projection의 행렬곱과 CPU FP64 기준값 비교 |
| `compare` | CSV·Markdown·레이어 그래프·토큰 heatmap 생성 |
| `smoke` | 외부 모델 없이 CPU에서 실행 경로 확인 |

`layer-parity`와 `python -m parity`는 같은 명령입니다. 실행 결과는 새로운 출력 폴더에 저장합니다.

## 문서 읽는 순서

| 목적 | 문서 |
|---|---|
| 처음 설치하고 실행 | [빠른 실행 안내](docs/QUICKSTART_KO.md) |
| 계산 원리부터 이해 | [상세 기술 문서](docs/TECHNICAL_GUIDE_KO.md) |
| 전체 명령과 실험 절차 | [실행 매뉴얼](docs/RUNBOOK_KO.md) |
| vLLM API·레이어 계측 방안 | [vLLM 비교 안내](docs/VLLM_KO.md) |
| 기존 GKE에서 실행 | [GKE 구성](docs/GKE_KO.md) |
| GitHub·GitLab에 업로드 | [Git 등록 안내](docs/GIT_GUIDE_KO.md) |
| 코드 수정·검사·패키지 생성 | [개발 안내](docs/DEVELOPMENT_KO.md) |
| 실측 범위와 한계 확인 | [검증 기록](docs/VALIDATION_KO.md) |
| 전체 문서 탐색 | [문서 목차](docs/README.md) |

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/parity/` | 실제 Python 패키지 |
| `tests/` | 비교 지표, Qwen 수식, 실행 경로, CLI 검사 |
| `docs/` | 한국어 기술 문서와 검증 증거 |
| `examples/` | 공개 예제 입력 및 명확히 표시된 CPU 예시 보고서 |
| `scripts/` | 설치, 공식 구현 비교, 문서 링크 검사 |
| `deploy/` | 기존 GKE 노드풀용 Pod YAML |
| `.github/workflows/ci.yml` | CPU 테스트와 wheel 빌드 CI |
| `pyproject.toml` | 설치 의존성, 버전, CLI 정의 |

모델 가중치, 원시 tensor, 가상환경, 실행 결과 폴더는 `.gitignore`로 제외합니다. 소스와 문서만 Git에 올리는 구성을 기본으로 합니다.

## 결과를 어떻게 읽나요?

1. 같은 장비의 반복 실행 차이를 확인합니다.
2. GPU·TPU의 블록 출력 차이를 찾습니다.
3. 해당 블록을 동일한 입력으로 재실행합니다.
4. 필요하면 특정 행렬곱을 FP64 기준값과 비교합니다.
5. logits의 순위 변화와 실제 업무 평가 결과를 별도로 확인합니다.

단순히 최종 문장이 같다고 모든 계산이 같은 것은 아닙니다. 반대로 수치 차이가 있다는 이유만으로 모델 품질이 나쁘다고 판단할 수도 없습니다.

## 개발 검사

```bash
python -m pip install -e '.[cpu,dev]'
python -m pytest -q -m 'not reference'
python scripts/check_docs.py
python -m build
```

공식 Transformers 비교 검사는 [개발 안내](docs/DEVELOPMENT_KO.md)의 `reference` 설치 절차를 추가합니다. GitHub Actions는 이 검사까지 수행하도록 구성되어 있습니다. 실제 GitHub 실행 결과는 저장소에 올린 뒤 확인합니다.
