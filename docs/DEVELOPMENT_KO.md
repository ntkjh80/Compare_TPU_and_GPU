# 개발·검증·패키지 안내

## 1. 설치 설정의 기준

Python 패키지 설정의 기준 파일은 [`pyproject.toml`](../pyproject.toml)입니다. 실행 코드는 `src/parity/`에 있습니다. 저장소 루트에 우연히 있는 Python 파일이 import되는 문제를 줄이기 위해 `src` 배치를 사용합니다.

Python **3.12**를 사용합니다. GPU·TPU 설치는 Linux 호스트에서 수행합니다. CPU 경로는 Linux x86_64에서 검증했으며, 다른 OS·CPU 아키텍처의 wheel 제공 여부는 설치 때 확인해야 합니다.

| 설치 선택 | 포함 내용 |
|---|---|
| `.[cpu]` | 공통 의존성 + JAX CPU |
| `.[gpu]` | 공통 의존성 + JAX CUDA 12 |
| `.[cuda13]` | 공통 의존성 + JAX CUDA 13 |
| `.[tpu]` | 공통 의존성 + JAX/libtpu |
| `.[dev]` | pytest와 Python 빌드 도구 추가 |
| `.[reference]` | PyTorch·Transformers의 독립적인 Qwen 수식 검사 |

동일 venv에 GPU·TPU extra를 모두 설치하지 않습니다. 각 호스트마다 한 backend를 고릅니다. `pip install .`만 하면 JAX backend가 설치되지 않습니다. `requirements-common.txt`는 공통 의존성용 호환 진입점이며 전체 장비 환경 설치는 extra를 사용합니다.

## 2. CPU 개발 환경

새 가상환경을 만들고 editable 모드로 설치합니다.

```bash
python3.12 -m venv .venv-dev
source .venv-dev/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[cpu,dev]'
python -m pip check
```

`-e`는 수정한 `src/parity/`를 바로 import하게 합니다. 개발 중에는 편리하지만 저장소를 옮기거나 삭제하면 설치 경로도 깨집니다. 독립적으로 설치할 배포 파일은 아래 wheel 절차를 사용합니다.

기본 테스트는 모델을 다운로드하지 않습니다.

```bash
python -m pytest -q -m 'not reference'
layer-parity smoke --out runs/dev-smoke
python scripts/check_docs.py
python -m compileall -q src scripts tests
bash -n scripts/setup.sh scripts/run_side.sh
```

smoke는 작은 합성 모델로 전체 실행·블록 replay·Q projection GEMM·보고서 생성을 확인합니다. 학습된 Qwen 가중치의 재검증이 아닙니다.

## 3. 독립적인 공식 구현과 비교

PyTorch CPU wheel을 먼저 설치한 뒤 reference extra를 추가합니다. GPU CUDA 라이브러리까지 가져오는 불필요한 설치를 줄이는 순서입니다.

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[cpu,dev,reference]'
python -m pip check
python -m pytest -q
```

`reference` 표시가 붙은 검사는 작은 같은 가중치를 JAX와 공식 `Qwen2ForCausalLM`에 넣고 FP32 hidden state와 logits를 비교합니다. RoPE, grouped-query attention, RMSNorm, MLP, tied embedding 계산의 이식 오류를 찾습니다. 필요한 패키지가 없으면 이 검사들은 skip됩니다. skip을 통과로 해석하지 마세요.

실제 0.5B 체크포인트 비교는 별도 작업입니다. [실행 매뉴얼](RUNBOOK_KO.md)의 8절을 따릅니다. 실제 모델의 reference 검사는 다운로드, 메모리와 실행 시간이 더 필요하므로 일반 CI에서 매번 수행하지 않습니다.

## 4. 테스트가 확인하는 경계

| 검사 | 찾으려는 문제 |
|---|---|
| 지표 검사 | 작은 예제의 지표 오류, 0 벡터, NaN/Inf 처리 |
| safetensors 읽기 | BF16 비트를 FP32로 옮길 때 잘못 읽는 문제 |
| bundle 검증 | 파일 변조·누락·manifest에 없는 파일 |
| 구조 검사 | 미지원 모델 설정을 조용히 실행하는 문제 |
| 독립 reference | 공식 Qwen2 수식과의 이식 차이 |
| 실행 파이프라인 | 전체 실행·replay·GEMM·비교의 연결, 입력 hash 검증 |
| CLI 검사 | 저장소 밖 import, 잘못된 허용치, 기존 출력 덮어쓰기 |
| 문서 검사 | 끊어진 상대 파일 링크, 패키지 버전 불일치 |

이 검사는 H200·Ironwood의 실제 수치 일치성, 모든 Qwen 설정, 분산 실행, 모델 품질을 보장하지 않습니다. 장비에서 실행한 결과는 CPU 검사와 분리해 기록합니다.

## 5. wheel과 소스 배포본 만들기

저장소 루트에서 실행합니다.

```bash
python -m build
```

`dist/gpu_tpu_layer_parity-1.2.0-py3-none-any.whl`과 소스 배포용 `.tar.gz`가 만들어집니다. wheel에는 실행 패키지가, 소스 배포본에는 코드·문서·테스트가 들어갑니다. GitHub workflow 파일과 `scripts/vllm_api_probe.py` 같은 보조 스크립트는 Git 저장소 ZIP에서 사용합니다. wheel만 설치하면 이 스크립트들은 설치되지 않습니다.

별도 환경에 wheel과 CPU 의존성을 설치해 확인할 수 있습니다.

```bash
python3.12 -m venv /tmp/parity-wheel-1_2
/tmp/parity-wheel-1_2/bin/python -m pip install 'dist/gpu_tpu_layer_parity-1.2.0-py3-none-any.whl[cpu]'
cd /tmp
/tmp/parity-wheel-1_2/bin/layer-parity --version
/tmp/parity-wheel-1_2/bin/layer-parity doctor --backend cpu
/tmp/parity-wheel-1_2/bin/layer-parity smoke --out /tmp/parity-wheel-smoke-1_2
```

`/tmp`의 경로가 이미 있으면 새 이름을 사용하세요. CI의 wheel 확인은 이미 설치한 CPU 의존성을 재사용해 빠르게 수행합니다. 문서의 위 명령은 독립 환경에서 의존성까지 설치하는 방식입니다.

## 6. 버전을 바꾸는 방법

패키지 버전은 `pyproject.toml`의 `project.version`과 `src/parity/__init__.py`를 함께 바꿉니다. `CHANGELOG.md`에 변경 이유와 수치 해석 영향을 기록합니다.

JAX 버전을 바꿀 때는 cpu/gpu/cuda13/tpu extra의 JAX·JAXLIB 값을 함께 검토합니다. `scripts/setup.sh`의 `JAX_VERSION` 환경변수로 우회하지 않습니다. `pyproject.toml`이 버전의 단일 기준입니다.

1. 별도 브랜치와 새 venv에서 의존성을 변경합니다.
2. CPU 테스트와 공식 구현 비교를 통과시킵니다.
3. 해당 GPU·TPU에서 `doctor`를 실행합니다.
4. 같은 commit과 bundle로 양쪽 full run·replay를 다시 실행합니다.
5. 해석에 필요한 `pip freeze`, driver/libtpu 정보, `run.json`을 보관합니다.

직접 의존성은 고정했지만 모든 전이 의존성을 고정한 cross-platform lock은 아닙니다. `scripts/setup.sh`가 저장하는 `requirements-resolved-*.txt`와 `run.json`이 실제 환경 기록입니다. GPU와 TPU는 장비 plugin이 다르므로 CPU freeze를 그대로 장비에 설치하지 않습니다.

## 7. 다른 모델을 지원하려면

`--model-id`만 바꾸면 모든 모델이 실행되는 구조가 아닙니다. `io.py`의 `validate_config()`와 `model.py`가 Qwen2 Dense의 지정된 조건을 전제로 합니다.

새 구조를 추가할 때는 checkpoint 이름 매핑, embedding, 정규화, RoPE, attention mask, MLP, logits 계산을 공식 구현과 먼저 비교합니다. 준비·실행·replay가 같은 검증 함수를 사용하게 유지합니다. KV cache나 MoE, 분산 통신을 추가하면 현재 수치 실험과 별도 실행 모드로 구분하고 기록해야 합니다.

처음에는 작은 합성 모델의 수식 검사 → 실제 소형 checkpoint FP32 비교 → 장비별 반복 검사 → 양 장비 비교 순서로 확장하는 편이 원인 추적에 유리합니다.

## 8. 주요 명령 기본값

정확한 전체 인자는 `layer-parity 명령 --help`로 확인합니다.

| 항목 | 기본값 |
|---|---|
| 모델 | Qwen/Qwen2.5-0.5B-Instruct |
| 준비 입력 길이 | 최대 128토큰 |
| 준비 fill | truncate; 빠른 실행 예제는 repeat를 명시 |
| dtype / precision | bf16 / highest |
| 반복 횟수 | 3 |
| device index | 0 |
| 내부 capture 레이어 | 없음; 블록 입출력은 모두 저장 |
| 비교 반복 번호 | 양쪽 0 |
| 비교 허용치 | atol=1e-5, rtol=1e-3 |

출력 폴더가 비어 있지 않으면 중단합니다. `compare`는 NaN/Inf가 발견되면 결과를 기록한 뒤 종료 코드 2를 반환합니다. 허용치 초과율이 높다는 이유만으로 자동 실패하지 않으므로 업무 품질 gate가 필요하면 별도 기준을 정의하세요.

## 참고 자료

- [PyPA: pyproject.toml 작성](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [GitHub: Python 빌드·테스트 workflow](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [JAX: backend별 설치](https://docs.jax.dev/en/latest/installation.html)
