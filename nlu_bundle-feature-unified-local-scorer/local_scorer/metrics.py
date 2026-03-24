from pathlib import Path

try:
    import numpy as np
    from sklearn import metrics as sklearn_metrics
except ImportError:
    np = None
    sklearn_metrics = None


DEFAULT_METRIC_FILE = (
    Path(__file__).resolve().parent / "metric.txt"
)


def load_metric_names(metric_file=None):
    metric_path = Path(metric_file) if metric_file else DEFAULT_METRIC_FILE
    with metric_path.open("r", encoding="utf-8") as handle:
        metric_names = [line.strip() for line in handle if line.strip()]

    missing = [name for name in metric_names if name not in METRIC_FUNCTIONS]
    if missing:
        raise KeyError("Unsupported metrics in {}: {}".format(metric_path, ", ".join(missing)))

    return metric_names


def compute_metrics(solution, prediction, metric_names=None):
    names = metric_names or load_metric_names()
    return [(name, float(METRIC_FUNCTIONS[name](solution, prediction))) for name in names]


if sklearn_metrics is not None and np is not None:
    def accuracy_score(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.accuracy_score(y_true, y_pred)


    def macro_precision(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )


    def macro_recall(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )


    def macro_f1(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )


    def weighted_macro_precision(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )


    def weighted_macro_recall(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )


    def weighted_mmacro_f1(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )


    def matthews_corrcoef(solution, prediction):
        y_true, y_pred = _prepare_sklearn_arrays(solution, prediction)
        return sklearn_metrics.matthews_corrcoef(y_true, y_pred)


    def _prepare_sklearn_arrays(solution, prediction):
        y_true = np.asarray(solution, dtype=float)
        y_pred = np.asarray(prediction, dtype=float)
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.ravel()
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
        return y_true, y_pred
else:
    from math import sqrt

    def accuracy_score(solution, prediction):
        y_true, y_pred, _ = _prepare_labels(solution, prediction)
        matches = sum(1 for truth, guess in zip(y_true, y_pred) if truth == guess)
        return matches / float(len(y_true))


    def macro_precision(solution, prediction):
        return _average_class_metric(solution, prediction, "precision", "macro")


    def macro_recall(solution, prediction):
        return _average_class_metric(solution, prediction, "recall", "macro")


    def macro_f1(solution, prediction):
        return _average_class_metric(solution, prediction, "f1", "macro")


    def weighted_macro_precision(solution, prediction):
        return _average_class_metric(solution, prediction, "precision", "weighted")


    def weighted_macro_recall(solution, prediction):
        return _average_class_metric(solution, prediction, "recall", "weighted")


    def weighted_mmacro_f1(solution, prediction):
        return _average_class_metric(solution, prediction, "f1", "weighted")


    def matthews_corrcoef(solution, prediction):
        y_true, y_pred, labels = _prepare_labels(solution, prediction)
        label_index = {label: index for index, label in enumerate(labels)}
        size = len(labels)
        confusion = [[0 for _ in range(size)] for _ in range(size)]

        for truth, guess in zip(y_true, y_pred):
            confusion[label_index[truth]][label_index[guess]] += 1

        samples = float(len(y_true))
        correct = sum(confusion[index][index] for index in range(size))
        predicted_totals = [sum(confusion[row][col] for row in range(size)) for col in range(size)]
        true_totals = [sum(row) for row in confusion]

        numerator = correct * samples - sum(
            predicted_totals[index] * true_totals[index] for index in range(size)
        )
        denominator_left = samples * samples - sum(total * total for total in predicted_totals)
        denominator_right = samples * samples - sum(total * total for total in true_totals)

        if denominator_left <= 0 or denominator_right <= 0:
            return 0.0

        return numerator / sqrt(denominator_left * denominator_right)


    def _average_class_metric(solution, prediction, metric_name, average):
        y_true, y_pred, labels = _prepare_labels(solution, prediction)
        per_label_scores = []
        supports = []

        for label in labels:
            stats = _label_stats(y_true, y_pred, label)
            support = stats["tp"] + stats["fn"]
            supports.append(support)
            per_label_scores.append(_score_from_stats(stats, metric_name))

        if average == "macro":
            return sum(per_label_scores) / float(len(per_label_scores))

        total_support = float(sum(supports))
        if total_support == 0:
            return 0.0

        weighted_total = sum(score * support for score, support in zip(per_label_scores, supports))
        return weighted_total / total_support


    def _score_from_stats(stats, metric_name):
        precision = _safe_divide(stats["tp"], stats["tp"] + stats["fp"])
        recall = _safe_divide(stats["tp"], stats["tp"] + stats["fn"])

        if metric_name == "precision":
            return precision
        if metric_name == "recall":
            return recall
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


    def _label_stats(y_true, y_pred, label):
        tp = fp = fn = 0
        for truth, guess in zip(y_true, y_pred):
            if truth == label and guess == label:
                tp += 1
            elif truth != label and guess == label:
                fp += 1
            elif truth == label and guess != label:
                fn += 1
        return {"tp": tp, "fp": fp, "fn": fn}


    def _prepare_labels(solution, prediction):
        y_true = [_normalise_row(row) for row in solution]
        y_pred = [_normalise_row(row) for row in prediction]
        labels = sorted(set(y_true) | set(y_pred))
        return y_true, y_pred, labels


    def _normalise_row(row):
        if len(row) == 1:
            return row[0]
        return tuple(row)


    def _safe_divide(numerator, denominator):
        if denominator == 0:
            return 0.0
        return numerator / float(denominator)


METRIC_FUNCTIONS = {
    "accuracy_score": accuracy_score,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_macro_precision": weighted_macro_precision,
    "weighted_macro_recall": weighted_macro_recall,
    "weighted_mmacro_f1": weighted_mmacro_f1,
    "matthews_corrcoef": matthews_corrcoef,
}
