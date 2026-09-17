from typing import Any


class QualityScoreCalculator:
    """Calculate an overall dataset quality score."""

    DEFAULT_WEIGHTS = {
        "critical": 20,
        "high": 10,
        "medium": 5,
        "low": 2,
    }

    def __init__(
        self,
        weights: dict[str, int] | None = None,
    ) -> None:
        """Initialize the quality score calculator."""
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def calculate(
        self,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate a quality score from detected issues."""
        score = 100

        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        deductions = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for issue in issues:
            severity = str(issue.get("severity", "medium")).lower()

            if severity not in self.weights:
                severity = "medium"

            severity_counts[severity] += 1
            deductions[severity] += self.weights[severity]

        total_deduction = sum(deductions.values())

        score = max(0, score - total_deduction)

        return {
            "score": score,
            "max_score": 100,
            "severity_counts": severity_counts,
            "deductions": deductions,
            "total_deduction": total_deduction,
            "issue_count": len(issues),
            "quality_level": self._quality_level(score),
        }

    @staticmethod
    def _quality_level(score: int) -> str:
        """Convert a numerical score into a quality level."""
        if score >= 90:
            return "excellent"

        if score >= 75:
            return "good"

        if score >= 50:
            return "fair"

        if score >= 25:
            return "poor"

        return "critical"
