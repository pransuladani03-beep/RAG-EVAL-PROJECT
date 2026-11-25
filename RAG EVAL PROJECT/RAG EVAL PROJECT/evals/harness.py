import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union


def load_goldens(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load golden dataset from a JSON file.
    Handles relative paths and case differences (e.g. Goldens vs goldens).
    """
    file_path = Path(path)

    if not file_path.exists():
        # Check alternative casings (e.g. Goldens/ vs goldens/)
        parts = list(file_path.parts)
        if parts:
            if parts[0].lower() == "goldens":
                alt_dir = "Goldens" if parts[0] == "goldens" else "goldens"
                alt_path = Path(alt_dir).joinpath(*parts[1:])
                if alt_path.exists():
                    file_path = alt_path

    if not file_path.exists():
        raise FileNotFoundError(f"Golden dataset file not found at: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result: Any) -> Dict[str, Any]:
    """
    Aggregate DeepEval EvaluationResult test cases grouped by metric name.
    Calculates mean score, total evaluated, passed count, and pass rate.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    test_results = getattr(result, "test_results", []) or []

    for tr in test_results:
        metrics_data = getattr(tr, "metrics_data", []) or []
        for m in metrics_data:
            name = getattr(m, "name", "Unknown Metric")
            if name not in summary:
                summary[name] = {
                    "total": 0,
                    "passed": 0,
                    "scores": [],
                    "threshold": getattr(m, "threshold", None),
                }
            summary[name]["total"] += 1
            if getattr(m, "success", False):
                summary[name]["passed"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                summary[name]["scores"].append(score)

    summary_metrics: Dict[str, Any] = {}
    for name, data in summary.items():
        scores = data["scores"]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = (data["passed"] / data["total"] * 100) if data["total"] else 0.0
        summary_metrics[name] = {
            "mean_score": round(mean_score, 3),
            "pass_rate": f"{pass_rate:.1f}%",
            "passed": data["passed"],
            "total": data["total"],
            "threshold": data["threshold"],
        }

    return summary_metrics


def print_summary(eval_name: str, summary: Dict[str, Any]) -> None:
    """
    Print a formatted summary table of evaluation metrics.
    """
    header = f"EVAL SUMMARY: {eval_name.upper()}"
    width = max(50, len(header) + 8)
    border = "=" * width

    print(f"\n{border}")
    print(f" {header} ".center(width))
    print(f"{border}\n")

    if not summary:
        print("No evaluation metrics found.\n")
        print(border)
        return

    col_metric = "Metric"
    col_mean = "Mean Score"
    col_pass = "Pass Rate"
    col_passed = "Passed / Total"
    col_thresh = "Threshold"

    header_line = f"{col_metric:<28} | {col_mean:<10} | {col_pass:<10} | {col_passed:<14} | {col_thresh:<10}"
    print(header_line)
    print("-" * len(header_line))

    for name, stats in summary.items():
        if isinstance(stats, dict):
            mean_str = str(stats.get("mean_score", "-"))
            pass_str = str(stats.get("pass_rate", "-"))
            count_str = f"{stats.get('passed', 0)}/{stats.get('total', 0)}"
            thresh_str = str(stats.get("threshold", "-"))
            print(f"{name:<28} | {mean_str:<10} | {pass_str:<10} | {count_str:<14} | {thresh_str:<10}")
        else:
            print(f"{name:<28} | {stats}")

    print(f"\n{border}\n")
