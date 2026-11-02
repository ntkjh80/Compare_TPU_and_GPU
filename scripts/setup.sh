#!/usr/bin/env bash
# 기존 vLLM/연구 환경을 변경하지 않도록 별도 venv에 설치합니다.
set -euo pipefail
cd "$(dirname "$0")/.."
target="${1:-cpu}"
case "$target" in
  cpu) extra="cpu" ;;
  gpu|cuda12) extra="gpu" ;;
  cuda13) extra="cuda13" ;;
  tpu) extra="tpu" ;;
  *) echo '사용법: bash scripts/setup.sh cpu|gpu|cuda12|cuda13|tpu'; exit 1 ;;
esac
python_bin="${PYTHON_BIN:-python3.12}"
env_dir="${PARITY_VENV:-.venv}"
if [ -n "${JAX_VERSION:-}" ]; then
  echo '버전은 pyproject.toml에서 관리합니다. JAX_VERSION 대신 pyproject.toml의 양쪽 의존성을 함께 수정하세요.'
  exit 1
fi
if [ -e "$env_dir" ]; then
  echo "venv 경로가 이미 있습니다: $env_dir. 기존 환경은 활성화해서 사용하거나 PARITY_VENV로 새 경로를 지정하세요."
  exit 1
fi
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
"$env_dir/bin/python" -m pip install -e ".[${extra}]"
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "requirements-resolved-${target}.txt"
printf '설치 완료. 다음 명령: source %s/bin/activate\n' "$env_dir"
