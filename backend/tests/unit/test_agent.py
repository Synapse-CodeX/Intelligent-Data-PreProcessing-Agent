import pandas as pd
from app.agent.planner import AgentPlanner
from app.agent.state import (
    AgentAction,
    AgentState,
    DatasetContext,
    QualityContext,
)
from app.agent.tool_registry import ToolRegistry
from app.agent.tools import (
    DatasetInspectionTool,
    DetectionTool,
    QualityAnalysisTool,
)
from app.schemas.issues import DataQualityIssue

# ---------------------------------------------------------------------------
# Agent State Tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Agent Tool Tests
# ---------------------------------------------------------------------------


def test_dataset_inspection_tool():
    df = pd.DataFrame(
        {
            "age": [25, 30, None],
            "income": [40000, 50000, 60000],
            "city": ["Kolkata", "Mumbai", "Delhi"],
        }
    )

    tool = DatasetInspectionTool()

    result = tool.run(df=df)

    assert result["success"] is True
    assert result["rows"] == 3
    assert result["columns"] == 3
    assert result["column_names"] == [
        "age",
        "income",
        "city",
    ]
    assert result["missing_values"]["age"] == 1


def test_dataset_inspection_empty_dataset():
    df = pd.DataFrame()

    tool = DatasetInspectionTool()

    result = tool.run(df=df)

    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_detection_tool():
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 35],
            "income": [40000, 50000, 60000, 70000],
        }
    )

    tool = DetectionTool()

    result = tool.run(df=df)

    assert result["success"] is True
    assert "issue_count" in result
    assert "issues" in result
    assert isinstance(result["issues"], list)


def test_detection_tool_empty_dataset():
    df = pd.DataFrame()

    tool = DetectionTool()

    result = tool.run(df=df)

    assert result["success"] is False
    assert result["issues"] == []


def test_quality_analysis_tool():
    issues = [
        {
            "type": "missing_values",
            "severity": "high",
        },
        {
            "type": "outliers",
            "severity": "medium",
        },
    ]

    tool = QualityAnalysisTool()

    result = tool.run(issues=issues)

    assert result["success"] is True
    assert result["quality"]["score"] == 85
    assert result["quality"]["issue_count"] == 2
    assert len(result["issues"]) == 2


def test_quality_analysis_tool_no_issues():
    tool = QualityAnalysisTool()

    result = tool.run(issues=[])

    assert result["success"] is True
    assert result["quality"]["score"] == 100
    assert result["quality"]["issue_count"] == 0


# ---------------------------------------------------------------------------
# Tool Registry Tests
# ---------------------------------------------------------------------------


def test_tool_registry_register_and_get():
    registry = ToolRegistry()
    tool = DatasetInspectionTool()

    registry.register(tool)

    assert registry.has("inspect_dataset")
    assert registry.get("inspect_dataset") is tool


def test_tool_registry_list_tools():
    registry = ToolRegistry()

    registry.register(DatasetInspectionTool())
    registry.register(DetectionTool())
    registry.register(QualityAnalysisTool())

    tools = registry.list_tools()

    assert tools == [
        "inspect_dataset",
        "detect_issues",
        "analyze_quality",
    ]


def test_tool_registry_unknown_tool():
    registry = ToolRegistry()

    try:
        registry.get("unknown_tool")
    except KeyError as exc:
        assert "unknown_tool" in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown tool.")


def test_tool_registry_duplicate_registration():
    registry = ToolRegistry()

    registry.register(DatasetInspectionTool())

    try:
        registry.register(DatasetInspectionTool())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate registration.")


def test_tool_registry_execute():
    registry = ToolRegistry()

    registry.register(DatasetInspectionTool())

    df = pd.DataFrame(
        {
            "age": [25, 30, None],
            "income": [40000, 50000, 60000],
        }
    )

    result = registry.execute(
        "inspect_dataset",
        df=df,
    )

    assert result["success"] is True
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_values"]["age"] == 1


def test_tool_registry_clear():
    registry = ToolRegistry()

    registry.register(DatasetInspectionTool())
    registry.register(DetectionTool())

    assert len(registry.list_tools()) == 2

    registry.clear()

    assert registry.list_tools() == []
    assert registry.has("inspect_dataset") is False
    assert registry.has("detect_issues") is False


# ---------------------------------------------------------------------------
# Agent Planner Tests
# ---------------------------------------------------------------------------


def test_agent_planner_missing_values():
    issue = DataQualityIssue(
        issue_type="missing_values",
        severity="high",
        description="Income contains missing values.",
        column="income",
        affected_count=10,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "missing_value_imputation"
    assert actions[0].column == "income"
    assert actions[0].parameters["strategy"] == "median"
    assert actions[0].approved is False


def test_agent_planner_outliers():
    issue = DataQualityIssue(
        issue_type="outliers",
        severity="medium",
        description="Income contains extreme values.",
        column="income",
        affected_count=4,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "outlier_treatment"
    assert actions[0].column == "income"
    assert actions[0].parameters["strategy"] == "iqr_capping"


def test_agent_planner_constant_feature():
    issue = DataQualityIssue(
        issue_type="constant_feature",
        severity="medium",
        description="Feature has a single unique value.",
        column="constant_col",
        affected_count=100,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "drop_column"
    assert actions[0].column == "constant_col"
    assert actions[0].parameters["reason"] == "constant_feature"


def test_agent_planner_categorical_inconsistency():
    issue = DataQualityIssue(
        issue_type="categorical_inconsistency",
        severity="medium",
        description="Categorical labels are inconsistent.",
        column="city",
        affected_count=5,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "categorical_normalization"
    assert actions[0].column == "city"
    assert actions[0].parameters["strip_whitespace"] is True
    assert actions[0].parameters["normalize_case"] is True


def test_agent_planner_high_cardinality():
    issue = DataQualityIssue(
        issue_type="high_cardinality",
        severity="medium",
        description="Categorical feature has excessive unique values.",
        column="zipcode",
        affected_count=100,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "high_cardinality_encoding"
    assert actions[0].column == "zipcode"
    assert actions[0].parameters["strategy"] == "frequency_encoding"


def test_agent_planner_skewness():
    issue = DataQualityIssue(
        issue_type="skewed_distribution",
        severity="medium",
        description="Feature is highly right-skewed.",
        column="income",
        affected_count=0,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "distribution_transformation"
    assert actions[0].column == "income"
    assert actions[0].parameters["strategy"] == "log1p"


def test_agent_planner_datetime():
    issue = DataQualityIssue(
        issue_type="datetime",
        severity="low",
        description="Datetime column detected.",
        column="created_at",
        affected_count=0,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "datetime_feature_extraction"
    assert actions[0].column == "created_at"
    assert "year" in actions[0].parameters["features"]
    assert "month" in actions[0].parameters["features"]


def test_agent_planner_data_leakage_requires_review():
    issue = DataQualityIssue(
        issue_type="data_leakage",
        severity="critical",
        description="Potential target leakage detected.",
        column="future_value",
        affected_count=20,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "manual_review"
    assert actions[0].column == "future_value"
    assert actions[0].parameters["requires_approval"] is True
    assert actions[0].approved is False


def test_agent_planner_unknown_issue():
    issue = DataQualityIssue(
        issue_type="unknown_issue",
        severity="low",
        description="Unknown issue type.",
        column="feature",
        affected_count=1,
    )

    state = AgentState(issues=[issue])

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 1
    assert actions[0].action_type == "manual_review"
    assert actions[0].column == "feature"


def test_agent_planner_no_issues():
    state = AgentState()

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert actions == []


def test_agent_planner_multiple_issues():
    issues = [
        DataQualityIssue(
            issue_type="missing_values",
            severity="high",
            description="Missing income values.",
            column="income",
            affected_count=10,
        ),
        DataQualityIssue(
            issue_type="outliers",
            severity="medium",
            description="Income has outliers.",
            column="income",
            affected_count=3,
        ),
        DataQualityIssue(
            issue_type="constant_feature",
            severity="low",
            description="Constant feature.",
            column="constant",
            affected_count=100,
        ),
    ]

    state = AgentState(issues=issues)

    planner = AgentPlanner()
    actions = planner.plan(state)

    assert len(actions) == 3
    assert actions[0].action_type == "missing_value_imputation"
    assert actions[1].action_type == "outlier_treatment"
    assert actions[2].action_type == "drop_column"
