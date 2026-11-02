#!/usr/bin/env python3
"""저장소 Markdown의 상대 파일 링크와 버전 일관성을 검사합니다. 외부 URL은 접속하지 않습니다."""
from pathlib import Path
import re
import sys
import tomllib


def main():
    root = Path(__file__).resolve().parents[1]
    problems = []
    for path in sorted(root.rglob("*.md")):
        if any(s.startswith(".") or s in {"build", "dist", "runs", "reports"} for s in path.relative_to(root).parts):
            continue
        content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            if re.match(r"^[a-zA-Z]+:", target) or target.startswith("#"):
                continue
            target = target.split("#", 1)[0].strip("<>")
            if target and not (path.parent / target).exists():
                problems.append(f"{path.relative_to(root)} → {target}")
    config = tomllib.loads((root / "pyproject.toml").read_text())
    code = (root / "src/parity/__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', code)
    if match is None or match[1] != config["project"]["version"]:
        problems.append("pyproject.toml과 parity.__version__ 불일치")
    if problems:
        print("문서/버전 검사 실패:\n" + "\n".join(problems))
        return 1
    print("문서 상대 링크와 패키지 버전 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
