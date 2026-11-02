# Git 저장소로 등록하기

이 배포본은 소스 코드, 문서, 테스트, CI 설정을 포함합니다. 원격 저장소를 생성하거나 계정에 push한 상태는 아닙니다. ZIP에는 `.git` 이력을 넣지 않았으므로 원하는 GitHub·GitLab 저장소의 첫 커밋으로 사용할 수 있습니다.

## 1. 새 저장소의 첫 커밋

ZIP을 풀고 `gpu-tpu-layer-parity` 폴더 안에서 실행합니다. Git이 설치되어 있어야 합니다.

```bash
git init -b main
git add .
git status --short
git diff --cached --check
git diff --cached --stat
git commit -m "Add GPU TPU layer parity toolkit"
```

`git status`는 이번에 올릴 파일을 보여 줍니다. `src/`, `tests/`, `docs/`, `scripts/`, `.github/`와 설정 파일이 포함됩니다. 가상환경과 다운로드한 모델이 보인다면 커밋 전에 경로와 `.gitignore`를 확인하세요.

Git이 이름·이메일을 요구하면 실제 작성자 정보를 이 저장소에 설정한 뒤 commit 명령을 다시 실행합니다.

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"
```

위 두 값은 예시 자리표시자입니다. 그대로 커밋하지 마세요.

## 2. 원격 저장소에 올리기

GitHub 또는 GitLab에서 **빈 저장소**를 만듭니다. 여기에는 이미 README가 있으므로 웹에서 README·라이선스·gitignore를 자동 생성하지 않으면 첫 push가 간단합니다. 공개 범위는 프로젝트의 공유 목적에 맞게 선택합니다.

아래 URL 전체를 생성한 저장소의 실제 SSH 또는 HTTPS URL로 바꿉니다.

```bash
git remote add origin https://github.com/YOUR_ORG/gpu-tpu-layer-parity.git
git remote -v
git push -u origin main
```

인증은 GitHub·GitLab의 기존 SSH 키 또는 credential manager를 사용합니다. 토큰을 소스 파일이나 원격 URL에 넣지 않습니다.

이미 origin이 있으면 `git remote -v`로 목적지를 먼저 확인합니다. 원격에 커밋이 있는 경우 강제 push로 덮어쓰지 말고 다음 절처럼 기존 저장소에서 작업하세요.

## 3. 기존 저장소에 추가하는 경우

기존 저장소를 clone한 폴더에서 새 브랜치를 만든 뒤 파일을 통합합니다.

```bash
git switch -c add-gpu-tpu-parity
```

이 패키지를 기존 Python 프로젝트의 루트에 그대로 덮어쓰면 `pyproject.toml` 등이 충돌할 수 있습니다. 독립 하위 폴더 `tools/gpu-tpu-layer-parity/`에 넣거나 기존 패키지 설정에 맞게 통합합니다. 하위 폴더에 넣었다면 그 폴더에서 설치·검사 명령을 실행합니다. GitHub Actions는 저장소 루트의 `.github/workflows/`에서만 읽으므로 workflow를 옮기고 `working-directory`와 경로를 조정해야 합니다.

검사를 통과한 뒤 기존 팀의 commit·push·PR 절차를 따릅니다. 이미 Git으로 관리 중인 폴더에서는 `git init`을 다시 할 필요가 없습니다.

## 4. Git에 넣는 것과 제외하는 것

| 넣는 파일 | 이유 |
|---|---|
| 코드·테스트·`pyproject.toml` | 같은 구현과 의존성 조건 재현 |
| 문서·공개 prompt 예시 | 실행 방법과 입력 형식 공유 |
| 작고 선별된 검증 요약 | 어떤 조건에서 무엇을 확인했는지 기록 |
| CI와 Pod 템플릿 | 반복 검증과 배포 절차 공유 |

| 제외 경로·형식 | 이유 |
|---|---|
| `.venv*/`, `venv/` | 장비마다 설치하는 환경 |
| `data/`, `checkpoint*/`, `bundle*/` | 모델 및 공통 입력의 대용량 파일 |
| `runs/`, `results*/`, `reports/` | 반복 생성되는 원시 결과 |
| `*.npy`, `*.npz`, `*.safetensors` | 대용량 tensor와 모델 가중치 |
| `.env`, 키 파일, credentials | 실행 환경의 개인·계정 정보 |
| `build/`, `dist/` | 빌드할 때 재생성되는 산출물 |

`.gitignore`는 **아직 추적하지 않는 파일**에 적용됩니다. 이미 추적 중인 파일을 자동으로 제거하지 않습니다. 파일을 삭제하지 않고 추적만 해제해야 한다면 대상을 정확히 확인한 뒤 `git rm --cached 실제파일`을 사용합니다.

제외 규칙은 파일을 실제로 만들지 않고도 확인할 수 있습니다.

```bash
git check-ignore --no-index data/model.safetensors runs/example/logits.npy .venv/bin/python
```

실제 업무 prompt나 결과를 공유하려면 내용을 검토한 작은 요약만 별도로 선별합니다. `.gitignore`는 내용의 민감성을 판정하는 도구가 아닙니다.

## 5. 실험과 Git 버전 연결

GPU와 TPU에 **같은 commit**을 배포합니다. 실험 전에 아래 출력과 설치 목록을 기록합니다.

```bash
git rev-parse HEAD
git status --short
python -m pip freeze
layer-parity --version
```

`run.json`은 Python 소스의 SHA-256도 기록합니다. 같은 Git commit에서 로컬 코드를 바꿨다면 source hash가 달라질 수 있습니다. 비교 도구는 기본적으로 서로 다른 코드의 결과를 거부합니다. 패키지 버전 문자열만 같다고 같은 구현인 것은 아닙니다.

실험 보고에는 commit, 모델 revision, bundle manifest, 장비, dtype, precision, 입력 길이와 반복 횟수를 함께 남깁니다. 원시 tensor를 Git에 넣지 않아도 어떤 실험인지 추적할 수 있습니다.

## 6. CI 확인

GitHub에 push하면 `.github/workflows/ci.yml`이 CPU 검사를 실행하도록 구성되어 있습니다. 웹의 Actions에서 실행 결과를 확인합니다. workflow를 제공한 것과 실제 원격 CI가 성공한 것은 다릅니다.

GitLab에서는 이 GitHub workflow가 실행되지 않습니다. [개발 안내](DEVELOPMENT_KO.md)의 같은 설치·검사 명령을 GitLab CI의 Python 3.12 job에 옮겨 사용하세요.

## 라이선스와 모델 파일

배포본에는 소유자와 공개 정책을 임의로 정한 LICENSE를 추가하지 않았습니다. 공개 배포 시 프로젝트에 적용할 라이선스는 저장소 소유자가 선택합니다. 모델 가중치는 ZIP에 포함하지 않으며, 다운로드한 Qwen 모델의 이용 조건은 모델 저장소의 라이선스를 확인합니다.
