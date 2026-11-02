"""FP64로 계산하는 대칭 비교 지표. 비유한 값을 정상으로 처리하지 않습니다."""
import numpy as np


def metrics(a, b, atol=1e-5, rtol=1e-3):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"비교 shape 불일치 또는 빈 텐서: {a.shape}, {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    bad = int(a.size - finite.sum())
    if bad:
        return {"elements": int(a.size), "nonfinite": bad, "max_abs": None,
                "mae": None, "rmse": None, "p99_abs": None, "rel_l2": None,
                "cosine": None, "outside_tolerance_fraction": None, "max_index": None}
    diff = np.abs(a - b)
    na, nb = np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel())
    denom = max(na, nb, 1e-30)
    cosine = float(np.dot(a.ravel(), b.ravel()) / (na * nb)) if na and nb else (1.0 if na == nb else 0.0)
    return {"elements": int(a.size), "nonfinite": 0, "max_abs": float(diff.max()),
            "mae": float(diff.mean()), "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "p99_abs": float(np.quantile(diff, .99)), "rel_l2": float(np.linalg.norm(diff.ravel()) / denom),
            "cosine": cosine, "outside_tolerance_fraction": float(np.mean(diff > atol + rtol * np.maximum(np.abs(a), np.abs(b)))),
            "max_index": list(map(int, np.unravel_index(int(diff.argmax()), diff.shape)))}
