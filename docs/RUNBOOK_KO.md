# GPU·TPU 레이어 수치 비교 실행 매뉴얼

작성일: 2026-11-02 · 패키지 버전: 1.2.0

이 패키지는 동일한 Qwen2.5 공개 가중치와 입력을 GPU·TPU에 넣고 레이어별 수치 차이를 검사합니다. 설치, 데이터 준비, 실행, 동일 입력 재실행, 행렬곱 분리, CSV·그래프 생성 명령을 제공합니다.

**지원 범위:** `Qwen/Qwen2.5-0.5B-Instruct`, 공통 JAX 진단 구현, 단일 JAX device, batch=1, prefill, BF16 또는 FP32. GPU·TPU 실행 분기는 backend 선택만 다릅니다. 실제 모델 가중치는 실행 시 공식 저장소에서 내려받습니다. ZIP에는 가중치를 포함하지 않습니다.

**이 결과의 의미:** JAX GPU backend와 JAX TPU backend의 수치 비교입니다. 운영 중인 vLLM GPU/TPU 커널 자체의 일치성을 증명하지 않습니다. gpt-oss, Kimi, DeepSeek, GLM의 MoE 구조를 이 코드로 실행할 수 있다고 가정하지 않습니다. 해당 모델은 별도 adapter가 필요합니다.

## 1. 준비할 환경

| 구분 | 준비 항목 |
|---|---|
| GPU 실행 호스트 | NVIDIA H200을 사용할 수 있는 Linux VM 또는 GKE Pod |
| TPU 실행 호스트 | Ironwood(TPU7x)를 사용할 수 있는 단일 호스트 VM 또는 GKE Pod |
| Python | 양쪽 모두 Python 3.12 |
| JAX | 기본 고정값 0.12.0.1. GPU·TPU에서 같은 JAX/JAXLIB 버전을 사용 |
| GPU 드라이버 | CUDA 12 wheels 사용 시 Linux 드라이버 525 이상이 JAX 공식 최소 조건. 실제 설치 wheel의 추가 조건도 확인 |
| 파일 공간 | 호스트별 20 GiB 이상을 시작점으로 권장. 긴 입력·다수 반복 시 결과 저장량 추가 산정 |
| CPU RAM | 호스트별 16 GiB 이상을 시작점으로 권장. 환경·컴파일 방식에 따라 더 필요할 수 있음 |
| 네트워크 | 설치 시 PyPI·패키지 CDN, 모델 준비 시 Hugging Face와 다운로드 CDN 접근 |
| 결과 분석 PC | Linux/macOS에서 CPU 환경 사용 가능. Windows는 WSL2 또는 Linux 분석 호스트 사용 |

RAM·디스크 수치는 이 패키지를 위한 구성 제안이며 공식 모델 최소 사양이 아닙니다. 가중치를 FP32 `.npy`로 준비하기 때문에 저장 공간은 BF16 체크포인트보다 증가합니다.

TPU7x에는 구형 `gcloud compute tpus tpu-vm create` 경로를 적용하지 않습니다. 현재 Google 문서는 TPU7x에 Compute Engine 또는 GKE 경로를 안내합니다. 기존 H200·Ironwood 자원이 있다면 그대로 접근해서 아래 환경만 추가하면 됩니다.[S1][S2]

Ironwood 한 칩은 JAX에서 두 device로 보일 수 있습니다. 4칩 호스트에서 `jax.devices()`가 8개라고 해서 물리 칩 8개라는 뜻은 아닙니다. 이 패키지는 기본적으로 `--device-index 0` 한 개만 계산에 사용합니다. TPU 자원 할당·과금 단위가 device 한 개로 줄어드는 것은 아닙니다.[S2]

### 기존 VM/Pod에 접속할 수 있다면

GPU 호스트에서 확인합니다.

```bash
nvidia-smi
python3.12 --version
df -h .
```

TPU 호스트에서는 Python 버전과 디스크를 확인합니다. 장비 접근 여부는 설치 후 `doctor`로 확인합니다.

Ubuntu 24.04 VM에서 Python이 없다면, 관리자 권한이 있는 경우에만 다음 OS 패키지를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv unzip
```

Ubuntu 22.04처럼 기본 저장소에 Python 3.12가 없는 환경은 조직에서 제공한 Python 3.12 또는 아래 Python 컨테이너를 사용하세요. 기존 vLLM 환경의 Python·CUDA 패키지를 바꾸지 않습니다.

### GKE에서 별도 실험 Pod가 필요한 경우

[GKE 구성](GKE_KO.md)의 순서를 사용합니다. 기존 노드풀을 지정하는 Pod YAML 두 개가 들어 있습니다. Docker-in-Docker는 필요하지 않습니다. 클러스터·노드풀을 새로 만드는 작업은 포함하지 않습니다.

## 2. 압축 해제와 설치

두 호스트 모두 ZIP을 풀고 `gpu-tpu-layer-parity` 폴더로 이동합니다. 아래부터 모든 명령의 현재 디렉터리는 이 폴더입니다.

GPU 호스트:

```bash
bash scripts/setup.sh gpu
source .venv/bin/activate
python -m parity doctor --backend gpu
```

TPU 호스트:

```bash
bash scripts/setup.sh tpu
source .venv/bin/activate
python -m parity doctor --backend tpu
```

분석 전용 CPU PC:

```bash
bash scripts/setup.sh cpu
source .venv/bin/activate
python -m parity doctor --backend cpu
```

`doctor`는 실제 장비에서 BF16 행렬곱을 실행하고 backend, device 이름, 패키지 버전을 출력합니다. GPU 요청 시 GPU 초기화가 실패하면 CPU로 대신 실행하지 않습니다. `kind`가 H200 또는 사용하려는 TPU인지 직접 확인하세요.

GPU 설치는 `jax[cuda12]`, TPU 설치는 `jax[tpu]`를 사용합니다. JAX 공식 설치 경로입니다.[S3] CUDA 13 환경을 별도로 검증하려면 `bash scripts/setup.sh cuda13`을 사용합니다. CUDA 13은 드라이버 최소 조건이 달라집니다.

GCP가 해당 Ironwood 이미지에 다른 JAX/libtpu 조합을 지정했다면 그 조합을 우선 확인하세요. `pyproject.toml`의 JAX·JAXLIB 버전을 양쪽 extra에서 함께 바꾸고 검증한 동일 commit을 두 장비에 배포합니다. 1.1부터는 `JAX_VERSION` 환경변수로 버전을 덮어쓰지 않습니다. 자세한 절차는 [개발 안내](DEVELOPMENT_KO.md)의 6절을 참고하세요.

다른 이름의 가상환경을 만들려면 다음과 같이 실행합니다.

```bash
PARITY_VENV=.venv-parity bash scripts/setup.sh tpu
source .venv-parity/bin/activate
```

설치 결과는 `requirements-resolved-gpu.txt` 또는 `requirements-resolved-tpu.txt`에 저장됩니다. GPU·TPU의 CUDA/libtpu 패키지는 다르지만 JAX·JAXLIB·NumPy 버전은 같아야 합니다. `compare`가 이 버전 차이를 검사합니다.

## 3. 모델과 입력은 한 곳에서만 준비

GPU 호스트 또는 CPU 준비 PC 한 곳에서 실행합니다. 기본 모델은 공식 설정상 24개 레이어, hidden size 896, attention head 14개, KV head 2개입니다.[S4]

```bash
python -m parity download --out checkpoint-qwen

python -m parity prepare \
  --model-dir checkpoint-qwen \
  --prompts examples/prompts.jsonl \
  --seq-len 128 \
  --fill repeat \
  --out bundle-128
```

`download`는 처음 조회한 모델 revision을 commit SHA로 확정한 뒤 해당 버전만 받습니다. `prepare`는 다음을 생성합니다.

| 파일 | 내용 |
|---|---|
| `bundle-128/config.json` | 모델 구조 |
| `bundle-128/weights/*.npy` | 체크포인트 값을 정확히 FP32로 확장한 가중치 |
| `bundle-128/inputs/*.npy` | 양쪽에서 그대로 읽을 token ID |
| `bundle-128/prompts.json` | 원문과 실제 사용 토큰 수 |
| `bundle-128/manifest.json` | 원본 revision, 파일 SHA-256, 입력 생성 방법 |

`--fill repeat`는 짧은 토큰열을 반복해서 정확히 128개로 맞춥니다. 고정 shape의 수치 검사 용도입니다. 자연스러운 문장 평가에는 `--fill truncate`를 사용합니다. truncate는 실제 길이를 유지하고 최대 길이만 제한합니다. 현재 구현은 padding 없이 sample마다 batch=1로 실행합니다. 채팅 템플릿이나 생성 품질 평가를 자동 적용하지 않습니다.

**동일한 `bundle-128` 폴더를 GPU와 TPU에 복사하세요. 각 장비에서 별도로 토큰화하거나 가중치를 다시 변환하지 않습니다.** SSH 가능한 환경의 예시는 다음과 같습니다.

```bash
tar -czf bundle-128.tar.gz bundle-128
scp bundle-128.tar.gz YOUR_TPU_HOST:/path/to/gpu-tpu-layer-parity/
# TPU 호스트에서 압축 해제
tar -xzf bundle-128.tar.gz
```

`YOUR_TPU_HOST`와 원격 경로는 실제 접근 주소로 바꿉니다. 사내 환경은 승인된 파일 전달 경로를 사용해도 됩니다. 실행 때 SHA-256이 일치하지 않으면 중단됩니다.

이미 모델을 로컬에 보관하고 있다면 `download`를 생략하고 해당 폴더를 `--model-dir`로 지정할 수 있습니다. `config.json`, `tokenizer.json`, 전체 `.safetensors` 파일이 필요합니다. 양자화 전용 checkpoint는 지원하지 않습니다.

## 4. GPU와 TPU에서 전체 모델 실행

GPU 호스트:

```bash
python -m parity run \
  --backend gpu --bundle bundle-128 --out results/gpu-bf16 \
  --dtype bf16 --precision highest --repeats 3 \
  --capture-layers 0 12 23
```

TPU 호스트:

```bash
python -m parity run \
  --backend tpu --bundle bundle-128 --out results/tpu-bf16 \
  --dtype bf16 --precision highest --repeats 3 \
  --capture-layers 0 12 23
```

레이어 번호는 **0부터 시작**합니다. 위 명령은 모든 24개 블록의 입력·출력과 최종 logits를 저장합니다. 지정한 0·12·23번 블록만 내부 연산을 추가 저장합니다.

`highest`를 지정해도 BF16 피연산자가 FP32 원본으로 복구되지는 않습니다. 이 패키지는 행렬곱의 FP32 누산을 요청하고 지정한 활성 dtype으로 명시적으로 변환합니다. 정규화 통계·softmax는 FP32로 계산합니다. backend별 내부 누산 순서까지 강제하는 것은 아닙니다.[S5]

대표 결과 파일:

```text
results/gpu-bf16/run.json
results/gpu-bf16/instrumentation_checks.json
results/gpu-bf16/repeat_00/korean/layer_000.input.npy
results/gpu-bf16/repeat_00/korean/layer_000.output.npy
results/gpu-bf16/repeat_00/korean/layer_012.detail_q_proj.npy
results/gpu-bf16/repeat_00/korean/logits.npy
```

기존 결과 경로가 비어 있지 않으면 중단합니다. 재실행은 새 이름을 사용하세요. `run.json`의 `complete: true`는 모든 sample과 반복 실행을 마쳤다는 뜻입니다.

## 5. 결과를 한 곳으로 모아 비교

두 결과 디렉터리를 분석 PC의 `results/gpu-bf16`, `results/tpu-bf16`로 복사합니다. 분석 환경은 CPU 설치로 충분합니다.

```bash
python -m parity compare \
  --left results/gpu-bf16 \
  --right results/tpu-bf16 \
  --out reports/gpu-vs-tpu-bf16
```

| 산출물 | 확인할 내용 |
|---|---|
| `report.md` | 차이가 큰 tensor, 장비 정보, 설정 차이 |
| `metrics.csv` | 최대/평균 절대 차이, RMSE, P99, 정규화 L2, cosine, 허용치 초과율 |
| `token_metrics.csv` | 각 레이어·토큰 위치의 정규화 L2 |
| `layer_difference.png` | 블록 출력 차이 |
| `token_heatmap.png` | 첫 sample의 레이어×토큰 위치 차이 |
| `summary.json` | 기계 판독용 요약과 계측 영향 검사 |

`metrics.csv`의 logits 행에는 Top-1 일치율도 있습니다. 결과가 100%라고 모든 logits가 동일하다는 뜻은 아닙니다.

장비 자체의 반복 변동도 확인합니다. 아래 GPU 명령을 TPU 결과에도 같은 방식으로 적용합니다.

```bash
python -m parity compare \
  --left results/gpu-bf16 --right results/gpu-bf16 \
  --left-repeat 0 --right-repeat 1 \
  --out reports/gpu-repeat-0-vs-1
```

기본 허용치 `atol=1e-5`, `rtol=1e-3`은 차이 분포를 보는 진단 설정입니다. BF16 모델의 합격 기준이 아닙니다. 비유한 값이 발견되면 보고서를 저장한 뒤 종료 코드 2를 반환합니다. 그 외 결과는 품질 합격 여부를 자동 판정하지 않습니다.

## 6. 차이가 큰 레이어를 같은 입력으로 재실행

예를 들어 12번 블록을 검사합니다. **GPU에서 저장한 `results/gpu-bf16`을 TPU에도 그대로 복사**합니다. 양쪽 모두 이 동일한 디렉터리를 `--source-run`으로 지정합니다. GPU는 GPU 입력, TPU는 TPU 입력을 따로 사용하면 이 실험의 목적을 충족하지 못합니다.

GPU:

```bash
python -m parity replay \
  --backend gpu --bundle bundle-128 \
  --source-run results/gpu-bf16 --layer 12 \
  --dtype bf16 --precision highest --repeats 3 \
  --out results/gpu-replay-12
```

TPU:

```bash
python -m parity replay \
  --backend tpu --bundle bundle-128 \
  --source-run results/gpu-bf16 --layer 12 \
  --dtype bf16 --precision highest --repeats 3 \
  --out results/tpu-replay-12
```

분석 PC:

```bash
python -m parity compare \
  --left results/gpu-replay-12 --right results/tpu-replay-12 \
  --out reports/replay-12
```

재실행은 해당 블록의 모든 내부 지점을 저장합니다. 비교 도구는 재실행에 사용된 공통 입력의 SHA-256도 검사합니다. 해당 레이어의 가중치는 공통 bundle에서 다시 가져옵니다. 현재 실험은 cache가 없는 prefill이므로 position은 0부터 입력 길이−1까지 동일합니다.

블록의 `plain_output.npy`는 출력만 반환하는 계산 결과입니다. `output.npy`는 모든 중간값을 함께 반환하는 계측 계산의 결과입니다. 이 둘이 다르면 컴파일·연산 결합에 계측 영향이 있습니다. `instrumentation_checks.json`에 기록됩니다. 계측이 수치 결과를 바꿀 수 있다는 점은 JAX 공식 문서에서도 설명합니다.[S6]

## 7. 실제 행렬곱을 따로 비교

12번 블록의 Q projection을 검사하는 예시입니다. 아래 두 명령은 같은 GPU full run의 `detail_input_norm.npy`와 공통 checkpoint의 Q 가중치를 사용합니다.

GPU:

```bash
python -m parity matmul \
  --backend gpu --bundle bundle-128 \
  --source-run results/gpu-bf16 --layer 12 \
  --dtype bf16 --precision highest --repeats 3 \
  --out results/gpu-gemm-12
```

TPU:

```bash
python -m parity matmul \
  --backend tpu --bundle bundle-128 \
  --source-run results/gpu-bf16 --layer 12 \
  --dtype bf16 --precision highest --repeats 3 \
  --out results/tpu-gemm-12
```

분석 PC:

```bash
python -m parity compare \
  --left results/gpu-gemm-12 --right results/tpu-gemm-12 \
  --out reports/gemm-12
```

`reference_metrics.json`은 각 장비 결과와 CPU FP64 기준 계산의 차이입니다. 기준값은 실제 장비에 입력된 반올림 완료 값을 CPU로 가져와 FP64로 변환한 뒤 계산합니다. 다음 세 결과를 함께 봅니다.

1. GPU ↔ TPU: 장비 실행 결과 간 차이.
2. GPU ↔ CPU FP64: GPU 결과와 고정밀 기준값의 차이.
3. TPU ↔ CPU FP64: TPU 결과와 고정밀 기준값의 차이.

`accumulator_fp32.npy`는 FP32 출력 요청 결과이고 `rounded_activation.npy`는 활성 dtype으로 변환한 값입니다. 칩 내부의 모든 부분 누산을 직접 읽는 기능은 아닙니다. 이 분리 실험은 bias를 제외한 `A @ Wᵀ`만 계산합니다. `layer.detail_q_proj`와 완전히 동일한 연산 경계라고 가정하지 마세요.

## 8. FP32 비교와 공식 모델 구현 검증

FP32 실험은 새 결과 디렉터리에 저장합니다. GPU·TPU 각각 실행합니다.

```bash
python -m parity run \
  --backend gpu --bundle bundle-128 --out results/gpu-fp32 \
  --dtype fp32 --precision highest --repeats 1 --capture-layers 0
```

TPU에서는 `--backend tpu`, `--out results/tpu-fp32`로 바꿉니다. BF16 체크포인트를 FP32로 확장해 계산하므로 잃어버린 학습 당시 FP32 가중치를 복원하는 실험이 아닙니다.

필요하면 별도 CPU 검증 환경에서 공식 Transformers 구현과 JAX FP32 결과를 확인할 수 있습니다. 이 도구의 실행 환경에 torch를 설치할 필요는 없으며 아래 검증을 할 때만 필요합니다.

```bash
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers==4.51.3' 'safetensors==0.5.3' 'pytest==8.3.5'

python scripts/validate_hf.py \
  --model-dir checkpoint-qwen --bundle bundle-128 \
  --jax-run results/gpu-fp32 --out reports/hf-reference
```

공식 모델과 비교할 때도 동일한 체크포인트를 사용해야 합니다. 검증 스크립트의 허용치는 진단 코드의 이식 검사용이며 GPU·TPU BF16 합격 기준이 아닙니다. 실제로 이 배포본을 어디까지 검사했는지는 [검증 기록](VALIDATION_KO.md)에 기록합니다.

## 9. 저장량과 확장 순서

기본 모델에서 FP32 logits의 저장량은 sample 한 개·반복 한 번당 `토큰 수 × 151,936 × 4 bytes`입니다. 128토큰이면 약 77.8 MB입니다. 2,048토큰이면 약 1.24 GB입니다. 여기에 블록 입출력과 선택한 내부 텐서가 추가됩니다.

Attention score/probability는 `head 수 × 토큰 수²` 크기입니다. 긴 입력에서 모든 레이어를 상세 저장하지 마세요. 128토큰으로 설치와 결과 경로를 확인한 뒤 512, 2,048 순서로 늘립니다. `--capture-layers`를 생략하면 모든 블록 입출력만 저장합니다.

이 패키지는 처리량·지연시간 측정용이 아닙니다. CPU 복사·디스크 저장·컴파일 시간이 실행에 포함됩니다. 성능 실험은 계측을 제거한 실제 운영 경로로 따로 진행하세요. 또한 단일 device 결과에서 분산 all-reduce, KV cache, MoE routing, 운영 fused kernel의 수치 영향을 추론하지 않습니다.

## 10. 오류가 날 때

| 증상 | 확인할 항목 |
|---|---|
| GPU가 보이지 않음 | `nvidia-smi`, Pod의 GPU 할당, 드라이버, JAX CUDA extra |
| TPU 초기화 실패 | TPU 할당, 단일 호스트 조건, JAX/libtpu 호환성, 기존 프로세스의 장비 점유 |
| TPU device가 8개 보임 | Ironwood의 chiplet 노출 구조일 수 있음. 물리 칩 수와 구분 |
| `checksum 불일치` | bundle 복사 완료 여부. 양쪽에서 재생성하지 말고 원본을 다시 복사 |
| `설정 불일치` | dtype, precision, JAX/JAXLIB/NumPy 버전 확인 |
| `replay_inputs 불일치` | 양쪽 모두 동일 source-run의 입력을 사용했는지 확인 |
| 내부 tensor가 없음 | full run의 `--capture-layers`에 해당 레이어를 넣었는지 확인. 또는 replay 수행 |
| 입력 길이 증가 후 OOM | 상세 계측 레이어와 입력 길이를 줄이고 저장량 확인 |
| 출력 폴더가 비어 있지 않음 | 결과를 덮어쓰지 않도록 새 `--out` 지정 |
| Pod가 Pending | 실제 노드풀 이름, 장비 여유, taint/toleration, CPU/RAM/스토리지 요청, 조직 정책 확인 |
| GPU 라이브러리 충돌 | JAX 공식 문서의 `LD_LIBRARY_PATH`와 CUDA wheel 로딩 경로 확인. 기존 환경을 무작정 수정하지 말고 별도 venv/Pod에서 재검사 |

## 11. 코드 구성

| 파일 | 역할 |
|---|---|
| `src/parity/model.py` | Qwen2 진단 수식: RMSNorm, GQA, RoPE, Attention, SwiGLU |
| `src/parity/prepare.py` | revision 고정 다운로드, 가중치·입력 묶음 생성 |
| `src/parity/run.py` | 장비 검사, 전체 실행, 동일 입력 재실행, GEMM |
| `src/parity/metrics.py` | FP64 비교 지표 |
| `src/parity/compare.py` | 조건 검사, CSV·MD·PNG 보고서 |
| `src/parity/io.py` | 체크섬, safetensors 읽기, 메타데이터 |
| `scripts/setup.sh` | 별도 venv 설치 |
| `scripts/validate_hf.py` | 공식 Transformers와 FP32 비교 |
| `tests/test_core.py` | 수식·파일·재실행·보고서·오류 검출 검사 |
| `deploy/*.yaml` | 기존 GKE 노드풀용 실험 Pod 템플릿 |

## 참고한 1차 자료

- [S1] [Google Cloud TPU VM 생성 경로와 TPU7x 제한](https://docs.cloud.google.com/tpu/docs/create-tpu-vm)
- [S2] [Google Cloud TPU7x 구조 및 device 노출](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [S3] [JAX 공식 설치](https://docs.jax.dev/en/latest/installation.html)
- [S4] [Qwen2.5-0.5B-Instruct 공식 설정](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
- [S5] [JAX 행렬곱 정밀도](https://docs.jax.dev/en/latest/201/precision.html)
- [S6] [JAX 계측과 수치 변화](https://docs.jax.dev/en/latest/201/debugging.html)
- [S7] [Transformers Qwen2 구현, v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py)
- [S8] [GKE Ironwood](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood)
- [S9] [GKE TPU 작업 배포](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)

문서의 VM/Pod 설치 명령은 사용자가 접근 가능한 자원에서 실행하는 절차입니다. 이 패키지를 만드는 과정에서 사용자의 GCP 자원을 생성하거나 변경하지 않았습니다.
