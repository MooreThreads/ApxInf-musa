"""Offline TileLang operator package and its shared builder result."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BuildResult:
    kernels: dict[str, Any]
    scratch_bytes: int
