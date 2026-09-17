from app.agent.state import (
    AgentAction,
    AgentState,
    DatasetContext,
    QualityContext,
)
from app.schemas.issues import DataQualityIssue


def test_dataset_context_defaults():
    context = DatasetContext()

    assert context.rows == 0
    assert context.columns == 0
    assert context.column_names == []


def test_dataset_context():
    context = DatasetContext(
        name="customers.csv",
        rows=100,
        columns=5,
        column_names=[
            "age",
            "income",
            "city",
            "gender",
            "target",
        ],
    )

    assert context.name == "customers.csv"
    assert context.rows == 100
    assert context.columns == 5
    assert len(context.column_names) == 5


def test_quality_context_defaults():
    quality = QualityContext()

    assert quality.score == 100
    assert quality.max_score == 100
    assert quality.quality_level == "excellent"
    assert quality.issue_count == 0


def test_agent_action():
    action = AgentAction(
        action_type="median_imputation",
        column="income",
        reason="Income is numerical and skewed.",
    )

    assert action.action_type == "median_imputation"
    assert action.column == "income"
    assert action.approved is False


def test_agent_state_defaults():
    state = AgentState()

    assert state.dataset.rows == 0
    assert state.issues == []
    assert state.actions == []
    assert state.validation_result == {}
    assert state.execution_result == {}
    assert state.messages == []
    assert state.status == "initialized"


def test_agent_state_with_issues():
    issue = DataQualityIssue(
        issue_type="missing_values",
        severity="high",
        description="Income contains missing values.",
        column="income",
        affected_count=2,
    )

    state = AgentState(
        dataset=DatasetContext(
            name="customers.csv",
            rows=100,
            columns=5,
            column_names=[
                "age",
                "income",
                "city",
                "gender",
                "target",
            ],
        ),
        issues=[issue],
        quality=QualityContext(
            score=85,
            quality_level="good",
            issue_count=1,
            severity_counts={
                "critical": 0,
                "high": 1,
                "medium": 0,
                "low": 0,
            },
        ),
    )

    assert len(state.issues) == 1
    assert state.issues[0].issue_type == "missing_values"
    assert state.issues[0].column == "income"
    assert state.quality.score == 85


def test_agent_state_with_action():
    action = AgentAction(
        action_type="median_imputation",
        column="income",
        parameters={"strategy": "median"},
        reason="The column is numerical and skewed.",
    )

    state = AgentState(
        actions=[action],
        status="planning",
    )

    assert len(state.actions) == 1
    assert state.actions[0].action_type == "median_imputation"
    assert state.actions[0].parameters["strategy"] == "median"
    assert state.status == "planning"


def test_agent_state_serialization():
    state = AgentState(
        dataset=DatasetContext(
            name="customers.csv",
            rows=50,
            columns=3,
            column_names=["age", "income", "city"],
        ),
        quality=QualityContext(
            score=90,
            quality_level="excellent",
            issue_count=0,
        ),
        status="ready",
    )

    data = state.model_dump()

    assert data["dataset"]["name"] == "customers.csv"
    assert data["dataset"]["rows"] == 50
    assert data["quality"]["score"] == 90
    assert data["status"] == "ready"