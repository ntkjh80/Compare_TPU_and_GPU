#!/usr/bin/env python3
"""공식 Transformers 구현과 JAX FP32 결과를 비교합니다. 별도 CPU 검증 환경용입니다."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from parity.io import read_json, write_json, verify_bundle, new_dir, sha256
from parity.metrics import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--jax-run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    run = Path(a.jax_run)
    m = read_json(run / "run.json")
    if m["dtype"] != "fp32" or m["mode"] != "full" or m["bundle_sha256"] != verify_bundle(a.bundle):
        raise ValueError("같은 bundle의 FP32 full run이 필요합니다.")
    identity = read_json(Path(a.bundle) / "manifest.json")
    for name, digest in identity["metadata"].get("source_files", {}).items():
        if sha256(Path(a.model_dir) / name) != digest:
            raise ValueError("공식 reference 모델의 가중치가 bundle 원본과 다릅니다.")
    out = new_dir(a.out)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, local_files_only=True,
                torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False).eval()
    rows = []
    for file in sorted((Path(a.bundle) / "inputs").glob("[!.]*.npy")):
        capture, handles = {}, []
        for i, layer in enumerate(model.model.layers):
            def hook(module, inputs, output, index=i):
                y = output[0] if isinstance(output, tuple) else output
                capture[f"layer_{index:03d}.output.npy"] = y.detach().numpy().copy()
            handles.append(layer.register_forward_hook(hook))
        with torch.inference_mode():
            result = model(torch.from_numpy(np.load(file, allow_pickle=False).astype(np.int64)), use_cache=False)
        capture["logits.npy"] = result.logits.numpy()
        for handle in handles:
            handle.remove()
        for name, y in capture.items():
            x = np.load(run / "repeat_00" / file.stem / name, allow_pickle=False)
            row = {"sample": file.stem, "tensor": name, **metrics(x, y, atol=5e-4, rtol=5e-4)}
            if name == "logits.npy":
                row["top1_agreement"] = float(np.mean(x.argmax(-1) == y.argmax(-1)))
            rows.append(row)
    write_json(out / "hf_reference_metrics.json", rows)
    failed = any(r["nonfinite"] or r["outside_tolerance_fraction"] > 0 for r in rows)
    write_json(out / "validation.json", {"passed": not failed, "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "atol": 5e-4, "rtol": 5e-4,
                "note": "FP32 진단 구현의 이식 확인 기준. BF16 GPU/TPU 합격 기준은 아닙니다.",
                "max_abs": max(r["max_abs"] or 0 for r in rows), "tensors": len(rows)})
    print("PASS" if not failed else "FAIL", out)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
