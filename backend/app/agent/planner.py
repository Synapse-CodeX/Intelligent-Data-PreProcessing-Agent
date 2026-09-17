
from app.agent.state import AgentAction, AgentState
from app.schemas.issues import DataQualityIssue


class AgentPlanner:
    """Generate preprocessing actions from detected data-quality issues."""

    def plan(self, state: AgentState) -> list[AgentAction]:
        """Generate a list of proposed actions for the current agent state."""
        actions: list[AgentAction] = []

        for issue in state.issues:
            action = self._plan_for_issue(issue)

            if action is not None:
                actions.append(action)

        return actions

    def _plan_for_issue(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction | None:
        """Map a data-quality issue to a preprocessing action."""

        issue_type = issue.issue_type
        column = issue.column

        if issue_type == "missing_values":
            return self._plan_missing_values(issue)

        if issue_type == "outliers":
            return self._plan_outliers(issue)

        if issue_type == "categorical_inconsistency":
            return self._plan_categorical_consistency(issue)

        if issue_type == "high_cardinality":
            return self._plan_high_cardinality(issue)

        if issue_type == "constant_feature":
            return self._plan_constant_feature(issue)

        if issue_type == "skewed_distribution":
            return self._plan_skewness(issue)

        if issue_type == "datetime":
            return self._plan_datetime(issue)

        if issue_type == "data_leakage":
            return self._plan_data_leakage(issue)

        return AgentAction(
            action_type="manual_review",
            column=column,
            reason=(
                f"No automatic preprocessing strategy is defined "
                f"for issue type '{issue_type}'."
            ),
        )

    def _plan_missing_values(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan an imputation strategy for missing values."""
        strategy = issue.metadata.get("recommended_strategy", "median")

        return AgentAction(
            action_type="missing_value_imputation",
            column=issue.column,
            parameters={
                "strategy": strategy,
            },
            reason=(
                f"Missing values detected in '{issue.column}'. "
                f"Use {strategy} imputation."
            ),
        )

    def _plan_outliers(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan an outlier treatment strategy."""
        strategy = issue.metadata.get(
            "recommended_strategy",
            "iqr_capping",
        )

        return AgentAction(
            action_type="outlier_treatment",
            column=issue.column,
            parameters={
                "strategy": strategy,
            },
            reason=(f"Outliers detected in '{issue.column}'. Apply {strategy}."),
        )

    def _plan_categorical_consistency(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan categorical-value normalization."""
        return AgentAction(
            action_type="categorical_normalization",
            column=issue.column,
            parameters={
                "strip_whitespace": True,
                "normalize_case": True,
            },
            reason=(
                f"Inconsistent categorical values detected in "
                f"'{issue.column}'. Normalize categorical labels."
            ),
        )

    def _plan_high_cardinality(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan a strategy for high-cardinality categorical columns."""
        strategy = issue.metadata.get(
            "recommended_strategy",
            "frequency_encoding",
        )

        return AgentAction(
            action_type="high_cardinality_encoding",
            column=issue.column,
            parameters={
                "strategy": strategy,
            },
            reason=(f"High cardinality detected in '{issue.column}'. Use {strategy}."),
        )

    def _plan_constant_feature(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan removal of a constant feature."""
        return AgentAction(
            action_type="drop_column",
            column=issue.column,
            parameters={
                "reason": "constant_feature",
            },
            reason=(
                f"Column '{issue.column}' contains a constant value "
                "and provides no predictive variance."
            ),
        )

    def _plan_skewness(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan a transformation for a highly skewed feature."""
        strategy = issue.metadata.get(
            "recommended_strategy",
            "log1p",
        )

        return AgentAction(
            action_type="distribution_transformation",
            column=issue.column,
            parameters={
                "strategy": strategy,
            },
            reason=(
                f"Highly skewed distribution detected in "
                f"'{issue.column}'. Apply {strategy} transformation."
            ),
        )

    def _plan_datetime(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan datetime feature extraction."""
        return AgentAction(
            action_type="datetime_feature_extraction",
            column=issue.column,
            parameters={
                "features": [
                    "year",
                    "month",
                    "day",
                    "day_of_week",
                ],
            },
            reason=(
                f"Datetime column '{issue.column}' detected. "
                "Extract useful temporal features."
            ),
        )

    def _plan_data_leakage(
        self,
        issue: DataQualityIssue,
    ) -> AgentAction:
        """Plan manual review for potential data leakage."""
        return AgentAction(
            action_type="manual_review",
            column=issue.column,
            parameters={
                "risk": "data_leakage",
                "requires_approval": True,
            },
            reason=(
                f"Potential data leakage detected in '{issue.column}'. "
                "This requires human review before modification."
            ),
        )
