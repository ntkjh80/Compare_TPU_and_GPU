# 변경 이력

## 1.2.0 — 2026-11-02

- vLLM 고정 token ID API 수집·비교 스크립트 추가. 레이어 tensor 수집과 구분.
- 서버 환경·checkpoint 파일 hash 기록, 공통 입력 export, 동일 서버 반복 비교.
- 응답 token ID와 logprob 검사, 서로 다른 선택 토큰의 점수 오비교 방지.
- vLLM JAX·TorchTPU 경로, 공식 hidden state 추출, worker 계측, KV decode 설계 문서 추가.
- API 경로는 로컬 모형 HTTP 서버로 검사. 실제 vLLM·H200·Ironwood 실행은 미수행.

v1.2에서도 GPU·TPU 양쪽에 동일한 소스 버전을 배포하세요. 기존 버전의 실제 모델 검사 기록은 과거 증거로 보존합니다.

## 1.1.0 — 2026-11-02

- Git 저장소용 `src/` 구조, pyproject 패키징, `layer-parity` 명령 추가.
- CPU/GPU/CUDA13/TPU 설치 extra와 개발·공식 reference extra 구분.
- 외부 모델 다운로드 없는 CPU smoke 명령 추가.
- GitHub Actions CPU 검사와 wheel 빌드 workflow 추가.
- 계산 원리, 실제 tensor 크기, 코드 흐름, 결과 해석, Git 등록 문서 추가.
- 비유한/음수 허용치와 음수 반복 번호의 CLI 입력 거부.
- replay/matmul에서도 지원 모델 설정 검사 적용.
- Linear의 반올림 순서에 대한 주석을 실제 구현에 맞게 수정. 계산 수식 자체는 변경하지 않음.
- 가중치·tensor·인증 파일·실행 결과를 제외하는 Git 규칙 보강.

v1.0 실행 결과와 v1.1 실행 결과는 코드 SHA-256이 다르므로 기본 비교에서 거부됩니다. 양쪽 장비에 같은 v1.1 소스를 배포하고 다시 실행하세요.

## 1.0.0 — 2026-11-02

- Qwen2.5 Dense 진단 구현, 블록 캡처, 동일 입력 replay, Q projection GEMM.
- 실제 공개 가중치의 CPU FP32/BF16 실행 및 Transformers 비교 기록.
- GKE Pod 예시, 한국어 실행 매뉴얼, CSV/Markdown/PNG 보고서.
