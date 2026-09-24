from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuthResult:
    overall_score: float
    verdict: str
    confidence: float
    components: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    heatmap: Optional[object] = None
    aligned_query: Optional[object] = None
    reference: Optional[object] = None
    metadata: dict = field(default_factory=dict)


def verdict_for(score):
    if score >= 85:
        return "GENUINE"
    if score >= 70:
        return "LIKELY GENUINE"
    if score >= 55:
        return "UNCERTAIN"
    if score >= 35:
        return "LIKELY COUNTERFEIT"
    return "COUNTERFEIT"