import argparse
import sys
from pathlib import Path

try:
    from .io_utils import (
        find_baseline_table_path,
        get_shape,
        infer_baseline_table_metadata,
        infer_prediction_metadata,
        normalize_split_name,
        read_baseline_table,
        read_numeric_array,
        resolve_reference_path,
        validate_same_shape,
    )
    from .metrics import compute_metrics, load_metric_names
except ImportError:
    from io_utils import (
        find_baseline_table_path,
        get_shape,
        infer_baseline_table_metadata,
        infer_prediction_metadata,
        normalize_split_name,
        read_baseline_table,
        read_numeric_array,
        resolve_reference_path,
        validate_same_shape,
    )
    from metrics import compute_metrics, load_metric_names


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare a prediction file with bundled NLI, AV, or ED reference data."
    )
    parser.add_argument(
        "--task",
        choices=("nli", "av", "ed"),
        help="Task whose bundled reference should be used.",
    )
    parser.add_argument(
        "--prediction",
        help="Path to the numeric prediction file to score. If omitted, the CLI scores baseline methods for --task.",
    )
    parser.add_argument(
        "--split",
        choices=("dev",),
        help="Reference split to use. Only `dev` is supported.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="baseline",
        help="Directory containing task baseline CSV files like 25_DEV_NLI.csv.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.prediction:
        if not args.task:
            print(
                "Error: Baseline scoring requires --task (nli, av, or ed).",
                file=sys.stderr,
            )
            return 1
        try:
            report = score_task_baselines(
                task=args.task,
                baseline_dir=args.baseline_dir,
                split=args.split,
            )
        except Exception as exc:
            print("Error: {}".format(exc), file=sys.stderr)
            return 1
        print_report(report)
        return 0

    try:
        report = score_prediction(
            prediction_path=args.prediction,
            task=args.task,
            split=args.split,
        )
        print_report(report)
        return 0
    except Exception as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1


def score_prediction(prediction_path, task=None, split=None):
    prediction_path = Path(prediction_path).expanduser().resolve()
    try:
        inferred = infer_prediction_metadata(prediction_path)
    except ValueError:
        inferred = None

    chosen_task = task or (inferred["task"] if inferred else None)
    chosen_split = split or (inferred["split"] if inferred else "dev")
    if chosen_split:
        chosen_split = normalize_split_name(chosen_split)

    if not chosen_task:
        raise ValueError(
            "Provide --task or use a prediction filename like METHOD_TASK_SPLIT.csv."
        )

    reference_path = resolve_reference_path(chosen_task)
    solution = read_numeric_array(reference_path)
    prediction = read_numeric_array(prediction_path)
    validate_same_shape(solution, prediction, reference_path, prediction_path)

    metric_names = load_metric_names()
    scores = compute_metrics(solution, prediction, metric_names)
    shape = get_shape(solution)

    report = {
        "prediction": prediction_path,
        "reference": reference_path,
        "rows": shape[0],
        "shape": shape,
        "scores": scores,
    }
    if chosen_task:
        report["task"] = chosen_task
    if chosen_split:
        report["split"] = chosen_split
    if inferred and inferred.get("method"):
        report["method"] = inferred["method"]
    return report


def score_task_baselines(task, baseline_dir, split=None):
    chosen_split = normalize_split_name(split or "dev")
    baseline_file = find_baseline_table_path(baseline_dir, task, chosen_split)
    baseline_metadata = infer_baseline_table_metadata(baseline_file)

    reference_path = resolve_reference_path(task)
    solution = read_numeric_array(reference_path)
    baseline_table = read_baseline_table(baseline_file)
    validate_same_shape(solution, baseline_table["reference"], reference_path, baseline_file)

    if baseline_table["reference"] != solution:
        raise ValueError(
            "Reference column in {} does not match bundled reference {}.".format(
                baseline_file, reference_path
            )
        )

    metric_names = load_metric_names()
    method_scores = []
    for method_name, prediction in baseline_table["methods"]:
        validate_same_shape(solution, prediction, reference_path, baseline_file)
        method_scores.append(
            {
                "method": method_name,
                "scores": compute_metrics(solution, prediction, metric_names),
            }
        )

    shape = get_shape(solution)
    return {
        "type": "baseline_task",
        "baseline_directory": Path(baseline_dir).expanduser().resolve(),
        "baseline_file": baseline_file,
        "baseline_label": baseline_metadata["label"],
        "task": baseline_metadata["task"],
        "split": baseline_metadata["split"],
        "reference": reference_path,
        "rows": shape[0],
        "shape": shape,
        "method_scores": method_scores,
    }


def print_report(report):
    if report.get("type") == "baseline_task":
        print_baseline_task_report(report)
        return

    if "baseline_directory" in report:
        print("Baseline directory: {}".format(report["baseline_directory"]))
    if report.get("method"):
        print("Method: {}".format(report["method"]))
    if report.get("task"):
        print("Task: {}".format(report["task"]))
    if report.get("split"):
        print("Split: {}".format(report["split"]))
    if report.get("reference"):
        print("Reference: {}".format(report["reference"]))
    print("Prediction: {}".format(report["prediction"]))
    if report.get("error"):
        print("Error: {}".format(report["error"]))
        return
    print("Rows: {}".format(report["rows"]))
    print("Shape: {}".format(report["shape"]))
    for metric_name, score in report["scores"]:
        print("{}: {:.14f}".format(metric_name, score))


def print_baseline_task_report(report):
    print("Baseline directory: {}".format(report["baseline_directory"]))
    print("Baseline file: {}".format(report["baseline_file"]))
    print("Baseline label: {}".format(report["baseline_label"]))
    print("Task: {}".format(report["task"]))
    print("Split: {}".format(report["split"]))
    print("Reference: {}".format(report["reference"]))
    print("Rows: {}".format(report["rows"]))
    print("Shape: {}".format(report["shape"]))
    method_names = [entry["method"] for entry in report["method_scores"]]
    print("Methods: {}".format(", ".join(method_names)))
    print("")
    for line in _format_baseline_metric_table(report["method_scores"]):
        print(line)


def _format_baseline_metric_table(method_scores):
    if not method_scores:
        return ["No baseline methods found."]

    method_names = [entry["method"] for entry in method_scores]
    metric_names = [metric_name for metric_name, _ in method_scores[0]["scores"]]

    table_rows = []
    for metric_name in metric_names:
        values = []
        for method_entry in method_scores:
            score_map = dict(method_entry["scores"])
            values.append("{:.6f}".format(score_map[metric_name]))
        table_rows.append([metric_name] + values)

    metric_width = max(len("Metric"), max(len(row[0]) for row in table_rows))
    value_widths = []
    for index, method_name in enumerate(method_names, start=1):
        width = max(len(method_name), max(len(row[index]) for row in table_rows))
        value_widths.append(width)

    header = "  ".join(
        [("Metric").ljust(metric_width)]
        + [name.rjust(width) for name, width in zip(method_names, value_widths)]
    )
    separator = "  ".join(
        ["-" * metric_width] + ["-" * width for width in value_widths]
    )
    lines = [header, separator]
    for row in table_rows:
        metric_name = row[0].ljust(metric_width)
        values = [value.rjust(width) for value, width in zip(row[1:], value_widths)]
        lines.append("  ".join([metric_name] + values))
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
