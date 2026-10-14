from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriftConfig:
    psi_threshold: float = 0.2
    ks_alpha: float = 0.05
    features: list[str] = field(default_factory=list)
    reference_path: str | None = None
    current_path: str | None = None
    buckets: int = 10
