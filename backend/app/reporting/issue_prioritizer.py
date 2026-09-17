from typing import Any


class IssuePrioritizer:
    """Prioritize detected data-quality issues."""

    SEVERITY_PRIORITY = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }

    def prioritize(
        self,
        issues: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort issues by severity and impact."""
        prioritized = []

        for index, issue in enumerate(issues):
            severity = str(issue.get("severity", "medium")).lower()

            if severity not in self.SEVERITY_PRIORITY:
                severity = "medium"

            enriched_issue = {
                **issue,
                "severity": severity,
                "priority_rank": index,
            }

            prioritized.append(enriched_issue)

        prioritized.sort(
            key=lambda issue: (
                self.SEVERITY_PRIORITY[issue["severity"]],
                -self._impact_value(issue),
            )
        )

        for rank, issue in enumerate(
            prioritized,
            start=1,
        ):
            issue["priority_rank"] = rank

        return prioritized

    @staticmethod
    def _impact_value(
        issue: dict[str, Any],
    ) -> float:
        """Extract an optional impact value."""
        impact = issue.get("impact", 0)

        try:
            return float(impact)
        except (TypeError, ValueError):
            return 0.0
