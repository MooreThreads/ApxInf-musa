#!/usr/bin/env python3
"""Enforce the isolation contract for the backend-native MUSA GR00T port."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2] / "crates/apxinf-model/src/musa_gr00t"
FORBIDDEN = {
    "CUDA GR00T module": r"\bcrate::gr00t\b|\bcrate::\{[^;]*\bgr00t\b",
    "CUDA backend": r"\bapxinf_cuda(?:_new)?\b",
    "obsolete MUSA feature": r"musa-backend",
    "CUDA feature gate": r'feature\s*=\s*"cuda"',
}


def source(path: Path) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)


def main() -> int:
    violations = []
    paths = sorted(ROOT.rglob("*.rs"))
    if not paths:
        violations.append("musa_gr00t: expected Rust module sources are missing")
    for path in paths:
        code = source(path)
        for description, pattern in FORBIDDEN.items():
            if re.search(pattern, code):
                violations.append(
                    f"{path.relative_to(ROOT)}: forbidden dependency on {description}"
                )
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("MUSA GR00T module dependency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
