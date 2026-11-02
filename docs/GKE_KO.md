# 기존 GKE에서 H200·Ironwood 실험 Pod 구성

이 절차는 **이미 준비된 GKE Standard 클러스터와 GPU/TPU 노드풀**을 사용합니다. 클러스터 또는 노드풀 생성은 수행하지 않습니다. 운영 서빙 Pod와 별도의 실험 Pod를 사용합니다. Docker-in-Docker는 필요하지 않습니다.

## 1. 관리 PC에서 접근 확인

```bash
gcloud container clusters get-credentials YOUR_CLUSTER \
  --location YOUR_CLUSTER_LOCATION --project YOUR_PROJECT

kubectl get nodes -L cloud.google.com/gke-nodepool,node.kubernetes.io/instance-type
kubectl auth can-i create pods
kubectl auth can-i create pods/exec
```

프로젝트, 클러스터, 위치는 실제 값으로 바꿉니다. 서로 다른 GPU/TPU 클러스터를 사용하면 각 명령에 `--context`를 지정하거나 연결 대상을 바꾼 후 실행합니다. namespace가 지정된 환경은 `-n YOUR_NAMESPACE`를 모든 kubectl 명령에 일관되게 추가하세요.

H200은 해당 GPU가 설치된 노드풀을 지정합니다. TPU는 **단일 호스트 Ironwood 노드풀**을 지정합니다. 현재 Google 문서에서 TPU7x의 단일 호스트 구성은 4칩이며 `tpu7x-standard-4t` 머신을 사용합니다. 이 패키지는 multi-host JAX 초기화나 분산 통신을 사용하지 않습니다.

## 2. YAML의 노드풀 이름 수정

- `deploy/gpu-pod.yaml`의 `YOUR_GPU_NODE_POOL`
- `deploy/tpu-pod.yaml`의 `YOUR_TPU_NODE_POOL`

기존 노드풀을 정확히 선택하므로 노드 자동 프로비저닝용 TPU topology/placement policy 설정을 이 템플릿에 새로 추가하지 않습니다. 조직의 namespace, image registry, taint 정책이 있으면 해당 항목을 조정합니다. 이미지 digest 고정이 필요한 조직은 승인된 Python 3.12 이미지로 교체하세요.

GPU Pod는 GPU 1개를 요청합니다. TPU Pod는 호스트의 4칩을 요청하지만 Python 진단 코드는 첫 번째 JAX device만 사용합니다. 장비 전체 활용률이나 처리량을 비교하는 구성이 아닙니다.

## 3. Pod 생성 및 파일 복사

```bash
kubectl apply -f deploy/gpu-pod.yaml
kubectl apply -f deploy/tpu-pod.yaml
kubectl wait --for=condition=Ready pod/parity-gpu --timeout=300s
kubectl wait --for=condition=Ready pod/parity-tpu --timeout=300s
```

Ready가 되지 않으면 `kubectl describe pod parity-tpu` 또는 GPU Pod의 Events를 확인합니다. 조직 정책상 root 컨테이너가 금지돼 있다면 승인된 사용자·쓰기 가능한 volume 경로에 맞춰 YAML을 조정해야 합니다. 이 템플릿에는 privileged 모드를 요구하지 않습니다.

관리 PC에서 압축을 푼 폴더의 **상위 디렉터리**로 이동한 후 복사합니다.

```bash
kubectl cp gpu-tpu-layer-parity parity-gpu:/work/gpu-tpu-layer-parity
kubectl cp gpu-tpu-layer-parity parity-tpu:/work/gpu-tpu-layer-parity
```

이미 venv나 결과를 생성한 폴더 전체를 복사하면 불필요하게 커질 수 있습니다. 배포 ZIP을 새로 푼 소스 폴더를 복사하세요. 기본 Python Debian 이미지에는 kubectl cp에 필요한 tar가 있습니다. 다른 이미지에서는 tar 유무를 확인합니다.

## 4. 양쪽 Pod에 venv 설치

```bash
kubectl exec parity-gpu -- bash -lc \
  'cd /work/gpu-tpu-layer-parity && bash scripts/setup.sh gpu'

kubectl exec parity-tpu -- bash -lc \
  'cd /work/gpu-tpu-layer-parity && bash scripts/setup.sh tpu'

kubectl exec parity-gpu -- bash -lc \
  'cd /work/gpu-tpu-layer-parity && .venv/bin/python -m parity doctor --backend gpu'

kubectl exec parity-tpu -- bash -lc \
  'cd /work/gpu-tpu-layer-parity && .venv/bin/python -m parity doctor --backend tpu'
```

이 이미지에는 NVIDIA 드라이버를 설치하지 않습니다. GPU 드라이버·장비 주입은 GKE GPU 노드가 제공하고, CUDA 사용자 공간 라이브러리는 JAX의 pip extra로 설치합니다. 호스트 드라이버가 준비되지 않았다면 관리자 측에서 먼저 구성해야 합니다.

## 5. 공통 bundle 복사

[실행 매뉴얼](RUNBOOK_KO.md) 3절에 따라 준비 PC에서 `bundle-128`을 한 번 생성합니다. 그 폴더가 현재 위치에 있다고 가정합니다.

```bash
kubectl cp bundle-128 parity-gpu:/work/gpu-tpu-layer-parity/bundle-128
kubectl cp bundle-128 parity-tpu:/work/gpu-tpu-layer-parity/bundle-128
```

대용량 복사는 조직의 GCS/PVC/Filestore 경로를 사용해도 됩니다. 파일 내용과 manifest를 변경하지 않습니다. 두 장비에서 각각 새 bundle을 생성하지 않습니다.

## 6. 실행과 결과 회수

GPU:

```bash
kubectl exec parity-gpu -- bash -lc \
  'cd /work/gpu-tpu-layer-parity && .venv/bin/python -m parity run --backend gpu --bundle bundle-128 --out results/gpu-bf16 --dtype bf16 --precision highest --repeats 3 --capture-layers 0 12 23'
```

TPU:

```bash
kubectl exec parity-tpu -- bash -lc \
  'cd /work/gpu-tpu-layer-parity && .venv/bin/python -m parity run --backend tpu --bundle bundle-128 --out results/tpu-bf16 --dtype bf16 --precision highest --repeats 3 --capture-layers 0 12 23'
```

관리 PC의 결과 저장 위치에서 실행합니다.

```bash
mkdir -p results
kubectl cp parity-gpu:/work/gpu-tpu-layer-parity/results/gpu-bf16 results/gpu-bf16
kubectl cp parity-tpu:/work/gpu-tpu-layer-parity/results/tpu-bf16 results/tpu-bf16
```

`results/*/run.json`의 `complete`와 파일 복사 완료를 확인한 후 [실행 매뉴얼](RUNBOOK_KO.md) 5절의 `compare`를 실행합니다. 동일 입력 재실행에는 GPU full run을 TPU Pod에 추가 복사해야 합니다.

```bash
kubectl cp results/gpu-bf16 parity-tpu:/work/gpu-tpu-layer-parity/results/gpu-bf16
```

이후 `kubectl exec -it parity-tpu -- bash`로 접속해서 `/work/gpu-tpu-layer-parity`에서 venv를 활성화하고 [실행 매뉴얼](RUNBOOK_KO.md) 6·7절을 수행할 수 있습니다.

## 7. 종료

이 템플릿의 `/work`는 **emptyDir**입니다. Pod 삭제·노드 축출 시 데이터가 사라질 수 있습니다. 결과를 회수한 다음 실험 Pod를 종료합니다.

```bash
kubectl delete pod parity-gpu parity-tpu
```

Pod 삭제는 기존 노드풀을 삭제하지 않습니다. 자원 할당 및 비용 정리는 조직의 노드풀·예약 정책에 따라 별도로 처리합니다.

## 근거

- [Google: TPU7x 구조와 단일 호스트 구성](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Google: Ironwood in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood)
- [Google: TPU workload 배포](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [JAX: GPU/TPU 설치](https://docs.jax.dev/en/latest/installation.html)

YAML은 제공된 사용자 클러스터에서 실제 배포 검증하지 않았습니다. 장비·노드풀 접근 권한과 조직별 정책을 확인하고 실행해야 합니다.
