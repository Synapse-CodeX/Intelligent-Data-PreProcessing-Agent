from typing import Any

from app.reporting.issue_prioritizer import IssuePrioritizer
from app.reporting.quality_score import QualityScoreCalculator


class QualityService:
    """Orchestrate dataset quality scoring."""

    def __init__(self) -> None:
        self.score_calculator = QualityScoreCalculator()
        self.issue_prioritizer = IssuePrioritizer()

    def analyze(
        self,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a quality score and prioritized issues."""
        prioritized_issues = self.issue_prioritizer.prioritize(issues)

        score_result = self.score_calculator.calculate(prioritized_issues)

        return {
            "quality": score_result,
            "issues": prioritized_issues,
        }
