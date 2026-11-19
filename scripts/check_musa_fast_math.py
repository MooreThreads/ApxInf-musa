#!/usr/bin/env python3
"""Check that shared MUSA kernels gate fast transcendentals by generation.

`__musa_exp_f`, `__musa_tanh_f` and `__sincosf` are the MP31 (S5000/S4000)
fast paths. They are not a faster spelling of the same function on MP22
(M1000): commit 5b7d172 substituted them unguarded in `kernels/custom/` and
PI0.5 BF16 produced NaN, its INT8 graph fell to cosine 0.1329 and GR00T BF16
to 0.9611. Reverting the four substitutions restored the Orin gate.

Shared kernels must therefore route them through the `apxinf_exp_f`,
`apxinf_tanh_f` and `apxinf_sincosf` helpers in `kernels/custom/math.cuh`,
which take the intrinsic only under `__MUSA_ARCH__ >= 310` and the libm call
otherwise (the same rule the vendored `mutlass::fast_tanh` uses).

Only `kernels/custom/` is scanned. `kernels/mp31_fp8/` and
`kernels/mp31_attention/` are compiled exclusively under their own
generation/vendor gates and already guard every intrinsic locally.

Exit status: 0 clean, 1 violations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREE = ROOT / "crates/apxinf-musa/kernels/custom"
MATH = TREE / "math.cuh"

# Longest first so `__musa_exp_f` is not reported as a prefix of another token.
TOKENS = (
    "__musa_sigmoid_f",
    "__musa_exp2_f",
    "__musa_exp_f",
    "__musa_tanh_f",
    "__musa_log2_f",
    "__musa_fdividef",
    "__musa_rsqrt",
    "__sincosf",
)
GUARD_HELPERS = ("apxinf_exp_f", "apxinf_tanh_f", "apxinf_sincosf")


def strip_comments(text: str) -> list[str]:
    """Return the source lines with // and /* */ comments removed."""
    lines = []
    in_block = False
    for raw in text.splitlines():
        line = ""
        index = 0
        while index < len(raw):
            if in_block:
                end = raw.find("*/", index)
                if end == -1:
                    index = len(raw)
                else:
                    in_block = False
                    index = end + 2
                continue
            if raw.startswith("//", index):
                break
            if raw.startswith("/*", index):
                in_block = True
                index += 2
                continue
            line += raw[index]
            index += 1
        lines.append(line)
    return lines


def guarded(conditions: list[str]) -> bool:
    return any(
        "__MUSA_ARCH__" in condition and "310" in condition
        for condition in conditions
    )


def main() -> int:
    problems: list[str] = []

    math_text = MATH.read_text()
    for helper in GUARD_HELPERS:
        if helper not in math_text:
            problems.append(
                f"{MATH.relative_to(ROOT)}: missing {helper}; the shared fast-math "
                "helpers are the only place the intrinsics may live"
            )
    if "__MUSA_ARCH__" not in math_text or "310" not in math_text:
        problems.append(
            f"{MATH.relative_to(ROOT)}: the helpers lost their "
            "__MUSA_ARCH__ >= 310 guard"
        )

    scanned = 0
    for path in sorted(TREE.glob("*.cuh")):
        scanned += 1
        conditions: list[str] = []
        for number, line in enumerate(strip_comments(path.read_text()), 1):
            stripped = line.strip()
            if stripped.startswith(("#if", "#ifdef", "#ifndef")):
                conditions.append(stripped)
            elif stripped.startswith("#elif"):
                if conditions:
                    conditions[-1] = stripped
            elif stripped.startswith("#else"):
                # `#else` is the negation of the guard: an intrinsic there is
                # *not* generation-gated, so drop the condition text.
                if conditions:
                    conditions[-1] = "#else"
            elif stripped.startswith("#endif"):
                if conditions:
                    conditions.pop()
            for token in TOKENS:
                if token not in line:
                    continue
                if guarded(conditions):
                    continue
                problems.append(
                    f"{path.relative_to(ROOT)}:{number}: {token} outside a "
                    "__MUSA_ARCH__ >= 310 guard; route it through apxinf_* in "
                    "kernels/custom/math.cuh (see doc/musa-backend.md#fast-transcendentals)"
                )

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"musa fast math: {scanned} headers checked, all intrinsics generation-gated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
