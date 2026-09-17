from app.reporting.issue_prioritizer import IssuePrioritizer
from app.reporting.quality_score import QualityScoreCalculator
from app.services.quality_service import QualityService


def test_quality_score_with_no_issues():
    calculator = QualityScoreCalculator()

    result = calculator.calculate([])

    assert result["score"] == 100
    assert result["issue_count"] == 0
    assert result["quality_level"] == "excellent"


def test_quality_score_deducts_for_issues():
    calculator = QualityScoreCalculator()

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

    result = calculator.calculate(issues)

    assert result["score"] == 85
    assert result["issue_count"] == 2
    assert result["severity_counts"]["high"] == 1
    assert result["severity_counts"]["medium"] == 1


def test_quality_score_never_goes_below_zero():
    calculator = QualityScoreCalculator(
        weights={
            "critical": 100,
            "high": 50,
            "medium": 25,
            "low": 10,
        }
    )

    issues = [
        {
            "type": "leakage",
            "severity": "critical",
        },
        {
            "type": "missing_values",
            "severity": "high",
        },
        {
            "type": "outliers",
            "severity": "high",
        },
    ]

    result = calculator.calculate(issues)

    assert result["score"] == 0
    assert result["quality_level"] == "critical"


def test_quality_levels():
    calculator = QualityScoreCalculator()

    assert calculator._quality_level(95) == "excellent"
    assert calculator._quality_level(80) == "good"
    assert calculator._quality_level(60) == "fair"
    assert calculator._quality_level(30) == "poor"
    assert calculator._quality_level(10) == "critical"


def test_issue_prioritizer_orders_by_severity():
    prioritizer = IssuePrioritizer()

    issues = [
        {
            "type": "duplicates",
            "severity": "low",
        },
        {
            "type": "outliers",
            "severity": "medium",
        },
        {
            "type": "leakage",
            "severity": "critical",
        },
        {
            "type": "missing_values",
            "severity": "high",
        },
    ]

    result = prioritizer.prioritize(issues)

    assert result[0]["severity"] == "critical"
    assert result[1]["severity"] == "high"
    assert result[2]["severity"] == "medium"
    assert result[3]["severity"] == "low"


def test_issue_prioritizer_uses_impact():
    prioritizer = IssuePrioritizer()

    issues = [
        {
            "type": "outliers",
            "severity": "high",
            "impact": 10,
        },
        {
            "type": "missing_values",
            "severity": "high",
            "impact": 80,
        },
    ]

    result = prioritizer.prioritize(issues)

    assert result[0]["type"] == "missing_values"
    assert result[1]["type"] == "outliers"


def test_quality_service():
    service = QualityService()

    issues = [
        {
            "type": "missing_values",
            "severity": "high",
            "impact": 50,
        },
        {
            "type": "duplicates",
            "severity": "low",
            "impact": 5,
        },
    ]

    result = service.analyze(issues)

    assert result["quality"]["score"] == 88
    assert result["quality"]["issue_count"] == 2
    assert len(result["issues"]) == 2
    assert result["issues"][0]["type"] == "missing_values"
    assert result["issues"][0]["priority_rank"] == 1
