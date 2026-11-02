"""실험 조건 검증 후 CSV, Markdown, PNG 보고서를 생성합니다."""
import csv
from pathlib import Path
import numpy as np
from .io import read_json, write_json, new_dir
from .metrics import metrics


def compare(args):
    left, right = Path(args.left), Path(args.right)
    lm, rm = read_json(left / "run.json"), read_json(right / "run.json")
    hard = ["bundle_sha256", "mode", "layer", "replay_inputs", "operand_sha256", "source_sha256"]
    for key in hard:
        if lm.get(key) != rm.get(key):
            raise ValueError(f"비교 불가: {key} 불일치")
    if not lm.get("complete") or not rm.get("complete"):
        raise ValueError("완료되지 않은 실행입니다.")
    changes = [key for key in ["dtype", "precision", "operator_policy"] if lm.get(key) != rm.get(key)]
    for package in ["jax", "jaxlib", "numpy"]:
        if lm["environment"]["packages"].get(package) != rm["environment"]["packages"].get(package):
            changes.append("version:" + package)
    if changes and not args.allow_setting_differences:
        raise ValueError(f"설정 불일치: {changes}. 의도한 설정 비교일 때만 --allow-setting-differences를 사용하세요.")
    lr = left / f"repeat_{args.left_repeat:02d}"
    rr = right / f"repeat_{args.right_repeat:02d}"
    lf = {str(p.relative_to(lr)) for p in lr.rglob("*.npy")}
    rf = {str(p.relative_to(rr)) for p in rr.rglob("*.npy")}
    if not lf or lf != rf:
        raise ValueError(f"비교 tensor 목록 불일치/빈 결과. left-only={sorted(lf-rf)[:5]}, right-only={sorted(rf-lf)[:5]}")
    out = new_dir(args.out)
    rows, token_rows = [], []
    for rel in sorted(lf):
        a, b = np.load(lr / rel, allow_pickle=False), np.load(rr / rel, allow_pickle=False)
        row = {"tensor": rel, "shape": str(a.shape), **metrics(a, b, args.atol, args.rtol)}
        if rel.endswith("/logits.npy") and row["nonfinite"] == 0:
            row["top1_agreement"] = float(np.mean(a.argmax(-1) == b.argmax(-1)))
            # 1, 2위 점수 차이. 작은 값에서는 순위가 바뀌기 쉽습니다.
            top = np.partition(a, -2, axis=-1)[..., -2:]
            row["left_min_top1_margin"] = float((top.max(-1) - top.min(-1)).min())
        rows.append(row)
        if ".output.npy" in rel and "detail_" not in rel and a.ndim == 3:
            for token in range(a.shape[1]):
                token_rows.append({"tensor": rel, "token": token,
                                   "rel_l2": metrics(a[0, token], b[0, token])["rel_l2"]})
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "token_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["tensor", "token", "rel_l2"])
        writer.writeheader()
        writer.writerows(token_rows)
    finite = [r for r in rows if r["rel_l2"] is not None]
    worst = sorted(finite, key=lambda x: x["rel_l2"], reverse=True)[:20]
    flags = {"setting_differences": changes, "nonfinite_tensors": sum(r["nonfinite"] > 0 for r in rows),
             "atol": args.atol, "rtol": args.rtol, "tensor_count": len(rows),
             "left_backend": lm["backend"], "right_backend": rm["backend"],
             "note": "허용치 초과율은 탐색 지표이며 모델 합격/불합격 판정이 아닙니다."}
    instrument = []
    for label, root in [("left", left), ("right", right)]:
        p = root / "instrumentation_checks.json"
        if p.exists():
            check = read_json(p)
            instrument.append({"side": label, "checks": len(check),
                               "changed": sum(r["max_abs"] != 0 for r in check),
                               "nonfinite": sum(r["nonfinite"] > 0 for r in check)})
    flags["instrumentation"] = instrument
    write_json(out / "summary.json", flags)
    lines = ["# GPU / TPU 레이어 수치 비교", "", f"- 좌측: `{lm['backend']}` / `{lm['device_kind']}`",
             f"- 우측: `{rm['backend']}` / `{rm['device_kind']}`", f"- 모드: `{lm['mode']}`",
             f"- 비교 tensor 수: {len(rows)}", f"- 비유한 값이 있는 tensor 수: {flags['nonfinite_tensors']}",
             f"- 의도적으로 허용한 설정 차이: {changes}", f"- 계측 영향 검사: {instrument}",
             "", "> 이 보고서는 수치 차이를 측정합니다. GPU/TPU 어느 쪽도 자동으로 정답으로 취급하지 않습니다.",
             "> 같은 장비끼리 비교한 결과는 GPU 대 TPU 실측 결과가 아닙니다.",
             "> plain 출력과 detail 출력의 계측 영향이 0이 아니면 내부 단계 값의 해석에 주의해야 합니다.",
             "", "## 정규화 L2 차이가 큰 tensor", "", "| Tensor | 정규화 L2 | 최대 절대 차이 |", "|---|---:|---:|"]
    lines += [f"| {r['tensor']} | {r['rel_l2']:.6g} | {r['max_abs']:.6g} |" for r in worst]
    lines += ["", "## 지표 해석", "", "- 정규화 L2 = ‖A−B‖₂ / max(‖A‖₂, ‖B‖₂, 1e−30).",
              "- 허용치 초과: |A−B| > atol + rtol × max(|A|, |B|). 기본값은 진단용이며 산업 표준 합격 기준이 아닙니다.",
              "- 원소별 지표 계산은 CPU FP64로 수행합니다. NaN/Inf는 정상 차이로 평균 내지 않습니다.",
              "- 실제 logits는 metrics.csv의 top1_agreement와 최소 1·2위 margin을 함께 확인합니다.",
              "- FP64 matmul 기준 대비 값은 각 실행 폴더의 reference_metrics.json에 있습니다."]
    if token_rows:
        label = f"{lm['backend'].upper()} {lm['dtype'].upper()} vs {rm['backend'].upper()} {rm['dtype'].upper()}"
        plot(rows, token_rows, out, label)
        lines += ["", "## 레이어별 차이", "", "![레이어별 차이](layer_difference.png)",
                  "", "## 토큰별 차이", "", "![토큰별 차이](token_heatmap.png)",
                  "", "Heatmap은 첫 번째 sample입니다. 나머지 sample의 정확한 값은 token_metrics.csv에 있습니다."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"보고서 생성: {out / 'report.md'}")
    if flags["nonfinite_tensors"]:
        raise SystemExit(2)


def plot(rows, tokens, out, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    samples = sorted({r["tensor"].split("/")[0] for r in tokens})
    for sample in samples:
        subset = [r for r in rows if r["tensor"].startswith(sample + "/") and r["tensor"].endswith(".output.npy")]
        vals = [(int(r["tensor"].split("layer_")[1].split(".")[0]), r["rel_l2"]) for r in subset]
        vals = [(i, v) for i, v in vals if v is not None]
        if vals:
            ax.plot([x[0] for x in vals], [x[1] for x in vals], marker=".", label=sample)
    ax.set(xlabel="Layer (zero-based)", ylabel="Normalized L2 difference", title=f"Block output difference | {label}")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "layer_difference.png", dpi=160)
    plt.close(fig)
    selected = [r for r in tokens if r["tensor"].startswith(samples[0] + "/")]
    names = sorted({r["tensor"] for r in selected})
    width = max(r["token"] for r in selected) + 1
    matrix = np.full((len(names), width), np.nan)
    for r in selected:
        matrix[names.index(r["tensor"]), r["token"]] = np.nan if r["rel_l2"] is None else r["rel_l2"]
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Normalized L2 difference")
    ax.set(xlabel="Token position", ylabel="Layer row", title=f"{label} | sample: {samples[0]}")
    fig.tight_layout()
    fig.savefig(out / "token_heatmap.png", dpi=160)
    plt.close(fig)
