#!/usr/bin/env bash
# GPU와 TPU에서 동일한 인자를 사용해 각각 실행합니다.
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:?gpu 또는 tpu 또는 cpu 지정}"
bundle="${2:?bundle 경로 지정}"
output="${3:?새 결과 경로 지정}"
python -m parity doctor --backend "$backend"
python -m parity run --backend "$backend" --bundle "$bundle" --out "$output" \
  --dtype bf16 --precision highest --repeats 3 --capture-layers 0
