"""공통 입력 묶음과 결과 파일. pickle/원격 모델 코드는 실행하지 않습니다."""
import hashlib
import json
import platform
import struct
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def new_dir(path):
    """이전 실험 결과가 섞이지 않도록 비어 있지 않은 경로는 거부합니다."""
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise ValueError(f"출력 경로가 비어 있지 않습니다: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def safetensor_arrays(path):
    """공개 safetensors 형식의 F32/F16/BF16만 읽습니다. BF16 값은 FP32로 정확히 표현됩니다."""
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("손상된 safetensors 헤더")
        n = struct.unpack("<Q", raw)[0]
        if n > 100_000_000 or n + 8 > path.stat().st_size:
            raise ValueError("비정상 safetensors 헤더 길이")
        header = json.loads(f.read(n))
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"지원하지 않는 가중치 dtype: {key} {meta['dtype']}")
        lo, hi = meta["data_offsets"]
        shape = tuple(meta["shape"])
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if lo < 0 or hi - lo != size or 8 + n + hi > path.stat().st_size:
            raise ValueError(f"잘못된 tensor 범위: {key}")
        a = np.memmap(path, dtype=dtype, mode="r", offset=8 + n + lo, shape=shape)
        if meta["dtype"] == "BF16":
            result = (a.astype(np.uint32) << 16).view(np.float32)
        else:
            result = np.asarray(a, dtype=np.float32).copy()
        yield key, result


def validate_config(c):
    """지원하지 않는 구조를 조용히 잘못 실행하지 않습니다."""
    if c.get("model_type") != "qwen2":
        raise ValueError("이 버전은 Qwen2/Qwen2.5 Dense의 qwen2 구조만 지원합니다.")
    if c.get("use_sliding_window") or c.get("rope_scaling") or c.get("quantization_config"):
        raise ValueError("sliding window / scaled RoPE / 양자화 checkpoint는 지원하지 않습니다.")
    if c.get("hidden_act", "silu") != "silu":
        raise ValueError("silu 이외 활성화 함수는 지원하지 않습니다.")
    if c["hidden_size"] % c["num_attention_heads"] or c["num_attention_heads"] % c["num_key_value_heads"]:
        raise ValueError("지원하지 않는 attention head 구성")


def make_manifest(root, metadata):
    root = Path(root)
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and not any(s.startswith(".") for s in p.relative_to(root).parts)}
    write_json(root / "manifest.json", {"schema": 1, "metadata": metadata, "files": files})


def verify_bundle(root):
    root = Path(root).resolve()
    m = read_json(root / "manifest.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p.name != "manifest.json" and not any(s.startswith(".") for s in p.relative_to(root).parts)}
    if actual != set(m["files"]):
        raise ValueError("입력 묶음에 manifest에 없는 파일이 있거나 파일이 누락되었습니다.")
    for rel, expected in m["files"].items():
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha256(p) != expected:
            raise ValueError(f"입력 묶음 checksum 불일치: {rel}")
    return sha256(root / "manifest.json")


def load_weights(root):
    return {p.stem: np.load(p, allow_pickle=False, mmap_mode="r") for p in sorted((Path(root) / "weights").glob("[!.]*.npy"))}


def environment():
    packages = {}
    for name in ["jax", "jaxlib", "libtpu", "numpy", "ml_dtypes", "scipy", "jax-cuda12-plugin", "jax-cuda13-plugin"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    # 실행된 모듈과 설치 메타데이터가 다른 환경은 비교 전에 바로 드러나게 합니다.
    import sys
    for name in ["jax", "jaxlib", "numpy"]:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "__version__"):
            if packages.get(name) != module.__version__:
                raise RuntimeError(f"{name} 설치 메타데이터와 import 버전이 다릅니다. 새 venv에서 설치하세요.")
    return {"python": platform.python_version(), "system": platform.platform(), "packages": packages}
