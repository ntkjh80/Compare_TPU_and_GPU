# 처음 실행하는 순서

목표는 먼저 **설치와 파일 생성 경로가 동작하는지** 확인하고, 이어서 **실제 Qwen 모델을 GPU·TPU에서 비교**하는 것입니다.

## 1. 작업 폴더 열기

ZIP을 풀었다면 `gpu-tpu-layer-parity` 폴더로 이동합니다. Git에서 받았다면 clone한 저장소로 이동합니다. 아래 명령은 모두 저장소 루트 기준입니다.

```bash
cd gpu-tpu-layer-parity
python3.12 --version
```

## 2. CPU에서 설치 확인

```bash
bash scripts/setup.sh cpu
source .venv/bin/activate
layer-parity --version
layer-parity doctor --backend cpu
layer-parity smoke --out runs/first-smoke
```

예상되는 확인 항목은 다음과 같습니다.

- 버전 출력: `1.2.0`
- doctor의 `requested_backend`: `cpu`
- smoke 완료 후 `runs/first-smoke/smoke.json`의 `completed`: `true`
- 보고서: `runs/first-smoke/report/report.md`

smoke의 가중치는 난수로 생성한 작은 값입니다. 학습된 모델이 아니며 좋은 문장을 생성하는지 평가하지 않습니다. 설치, 중간값 저장, replay, 행렬곱, 비교 보고서가 연결되는지 검사합니다.

같은 출력 폴더는 덮어쓰지 않습니다. 다시 실행할 때는 `runs/second-smoke`처럼 새 이름을 사용하세요.

## 3. 실제 GPU와 TPU에 설치

CPU 확인 환경과 별도로 GPU·TPU 각각의 호스트에 같은 저장소 버전을 배포합니다.

GPU 호스트:

```bash
bash scripts/setup.sh gpu
source .venv/bin/activate
layer-parity doctor --backend gpu
```

TPU 호스트:

```bash
bash scripts/setup.sh tpu
source .venv/bin/activate
layer-parity doctor --backend tpu
```

장비 이름이 실제 H200 또는 대상 TPU인지 확인합니다. GPU/TPU 초기화가 실패하면 실행을 중단합니다. CPU로 바꿔 실행한 결과를 GPU/TPU 결과로 저장하지 않습니다.

GKE 접근부터 필요한 경우 [GKE 구성](GKE_KO.md)을 사용합니다. 이미 Python을 실행할 수 있는 GPU·TPU VM/Pod가 있다면 그대로 진행합니다.

## 4. 실제 모델을 한 번만 준비

GPU 호스트 또는 준비용 CPU PC에서 실행합니다.

```bash
layer-parity download --out data/checkpoint-qwen
layer-parity prepare \
  --model-dir data/checkpoint-qwen \
  --prompts examples/prompts.jsonl \
  --seq-len 128 --fill repeat \
  --out data/bundle-128
```

`data/bundle-128` 폴더 전체를 다른 장비의 같은 경로로 복사합니다. 각 장비에서 독립적으로 다시 만들지 않습니다. 공유 bundle에는 token ID와 가중치가 이미 들어 있으므로 장비 실행 단계에서 토큰화하지 않습니다.

`repeat`는 고정 길이 연산 검사입니다. 자연스러운 문장 길이를 유지하려면 `--fill truncate`를 사용하세요. 입력 문장을 직접 바꾸려면 예시처럼 한 줄에 JSON 하나를 작성합니다.

```json
{"id":"sample01","text":"이 문장을 입력으로 사용합니다."}
```

## 5. 실제 모델의 첫 비교

GPU 호스트:

```bash
layer-parity run --backend gpu --bundle data/bundle-128 \
  --out runs/gpu-bf16 --dtype bf16 --precision highest \
  --repeats 3 --capture-layers 0 12 23
```

TPU 호스트:

```bash
layer-parity run --backend tpu --bundle data/bundle-128 \
  --out runs/tpu-bf16 --dtype bf16 --precision highest \
  --repeats 3 --capture-layers 0 12 23
```

두 결과 디렉터리를 분석 PC에 모은 뒤 실행합니다.

```bash
layer-parity compare --left runs/gpu-bf16 --right runs/tpu-bf16 \
  --out reports/gpu-vs-tpu
```

먼저 `reports/gpu-vs-tpu/report.md`를 읽고, 상세 숫자는 `metrics.csv`에서 확인합니다. 차이가 있는 것 자체를 오류로 판정하지 않습니다.

## 6. 12번 블록에서 차이가 크다면

GPU 결과 `runs/gpu-bf16`을 TPU에도 복사합니다. 두 장비 모두 같은 GPU 실행의 입력을 기준으로 사용합니다.

```bash
# GPU 호스트
layer-parity replay --backend gpu --bundle data/bundle-128 \
  --source-run runs/gpu-bf16 --layer 12 --out runs/gpu-replay-12
```

```bash
# TPU 호스트
layer-parity replay --backend tpu --bundle data/bundle-128 \
  --source-run runs/gpu-bf16 --layer 12 --out runs/tpu-replay-12
```

```bash
# 결과를 모은 분석 PC
layer-parity compare --left runs/gpu-replay-12 --right runs/tpu-replay-12 \
  --out reports/replay-12
```

여기서도 차이가 크면 내부 projection·attention·MLP 값을 조사합니다. 행렬곱만 분리하는 명령과 FP64 기준 계산은 [실행 매뉴얼](RUNBOOK_KO.md)의 7절에 있습니다.

## 7. 첫 실행에서 확인할 결과

| 확인 항목 | 의미 |
|---|---|
| `run.json`의 `complete: true` | 모든 입력과 반복을 끝냈음 |
| 올바른 backend와 device 이름 | 원하는 장비에서 실행했음 |
| `nonfinite_tensors: 0` | 비교한 값에 NaN/Inf가 없었음 |
| 자체 반복 비교 | 같은 장비의 실행 변동 파악 |
| `instrumentation_checks.json` | 중간값 계측이 출력에 미친 영향 |
| logits Top-1 일치율 | 고정 입력 위치에서 다음 토큰 선택이 같은 비율 |

이 항목들은 실험이 제대로 수행됐는지 확인하는 기준입니다. 모델 품질 합격 기준은 실제 업무 데이터와 평가 지표로 별도로 정합니다.
