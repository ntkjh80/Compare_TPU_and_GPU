"""python -m parity <명령>. --help로 모든 인자를 확인할 수 있습니다."""
import argparse
from . import __version__


def main():
    p = argparse.ArgumentParser(description="GPU/TPU 레이어 수치 일치성 검사")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke", help="모델 다운로드 없이 합성 입력으로 CPU 실행 경로 검사")
    s.add_argument("--out", required=True)
    d = sub.add_parser("download", help="공식 checkpoint의 commit을 고정해서 다운로드")
    d.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    d.add_argument("--revision", default="main")
    d.add_argument("--out", required=True)
    q = sub.add_parser("prepare", help="양쪽 장비에서 공유할 입력/가중치 묶음 생성")
    q.add_argument("--model-dir", required=True)
    q.add_argument("--prompts", default="examples/prompts.jsonl")
    q.add_argument("--seq-len", type=int, default=128)
    q.add_argument("--fill", choices=["truncate", "repeat"], default="truncate")
    q.add_argument("--out", required=True)
    for name in ["doctor", "run", "replay", "matmul"]:
        q = sub.add_parser(name)
        q.add_argument("--backend", choices=["cpu", "gpu", "tpu"], required=True)
        q.add_argument("--device-index", type=int, default=0)
        q.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
        q.add_argument("--precision", choices=["default", "high", "highest"], default="highest")
        if name != "doctor":
            q.add_argument("--bundle", required=True)
            q.add_argument("--out", required=True)
            q.add_argument("--repeats", type=int, default=3)
        if name in ["replay", "matmul"]:
            q.add_argument("--source-run", required=True)
            q.add_argument("--layer", type=int, required=True)
        if name == "run":
            q.add_argument("--capture-layers", type=int, nargs="*", default=[])
    q = sub.add_parser("compare")
    q.add_argument("--left", required=True)
    q.add_argument("--right", required=True)
    q.add_argument("--left-repeat", type=int, default=0)
    q.add_argument("--right-repeat", type=int, default=0)
    q.add_argument("--atol", type=float, default=1e-5)
    q.add_argument("--rtol", type=float, default=1e-3)
    q.add_argument("--allow-setting-differences", action="store_true")
    q.add_argument("--out", required=True)
    a = p.parse_args()
    if a.command == "smoke":
        from .smoke import smoke
        smoke(a)
        return
    if hasattr(a, "repeats") and a.repeats < 1:
        p.error("repeats는 1 이상이어야 합니다.")
    if a.command in ["download", "prepare"]:
        from . import prepare
        getattr(prepare, a.command)(a)
    elif a.command == "compare":
        from .compare import compare
        import math
        if not math.isfinite(a.atol) or not math.isfinite(a.rtol) or a.atol < 0 or a.rtol < 0:
            p.error("허용치는 유한한 0 이상의 수여야 합니다.")
        if a.left_repeat < 0 or a.right_repeat < 0:
            p.error("repeat 번호는 0 이상이어야 합니다.")
        compare(a)
    else:
        from . import run
        getattr(run, a.command)(a)


if __name__ == "__main__":
    main()
