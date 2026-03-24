import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_scorer.io_utils import get_shape, read_numeric_array


ROOT_DIR = Path(__file__).resolve().parents[1]
SCORER_MAIN = ROOT_DIR / "local_scorer" / "main.py"
REFERENCE_DIR = ROOT_DIR / "local_scorer" / "reference_data"
REFERENCE_FILES = {
    "nli": "NLU_SharedTask_NLI_dev.solution",
    "av": "NLU_SharedTask_AV_dev.solution",
    "ed": "NLU_SharedTask_ED_dev.solution",
}
EXPECTED_ROWS = {
    task: get_shape(read_numeric_array(REFERENCE_DIR / reference_name))[0]
    for task, reference_name in REFERENCE_FILES.items()
}
METRIC_NAMES = [
    "accuracy_score",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_macro_precision",
    "weighted_macro_recall",
    "weighted_mmacro_f1",
    "matthews_corrcoef",
]


class LocalScorerCliTests(unittest.TestCase):
    def test_perfect_match_scores_all_metrics_at_one(self):
        for task, reference_name in REFERENCE_FILES.items():
            with self.subTest(task=task):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    prediction_path = Path(tmp_dir) / "prediction.txt"
                    shutil.copyfile(REFERENCE_DIR / reference_name, prediction_path)

                    result = run_cli("--task", task, "--prediction", str(prediction_path))

                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("Rows: {}".format(EXPECTED_ROWS[task]), result.stdout)
                scores = parse_metric_output(result.stdout)
                self.assertEqual(METRIC_NAMES, list(scores.keys()))
                for score in scores.values():
                    self.assertAlmostEqual(score, 1.0, places=12)

    def test_shape_mismatch_returns_non_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "bad_prediction.txt"
            prediction_path.write_text("1\n0\n1\n", encoding="utf-8")

            result = run_cli("--task", "nli", "--prediction", str(prediction_path))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match reference shape", result.stderr)

    def test_header_line_is_ignored(self):
        reference_path = REFERENCE_DIR / REFERENCE_FILES["nli"]
        reference_body = reference_path.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "prediction_with_header.txt"
            prediction_path.write_text("label\n{}".format(reference_body), encoding="utf-8")

            result = run_cli("--task", "nli", "--prediction", str(prediction_path))

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        scores = parse_metric_output(result.stdout)
        for score in scores.values():
            self.assertAlmostEqual(score, 1.0, places=12)

    def test_csv_input_is_accepted_with_task_selection(self):
        reference_path = REFERENCE_DIR / REFERENCE_FILES["av"]
        reference_body = reference_path.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "prediction.csv"
            prediction_path.write_text("prediction\n{}".format(reference_body), encoding="utf-8")

            result = run_cli("--task", "av", "--prediction", str(prediction_path))

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Task: av", result.stdout)
        self.assertIn("Rows: {}".format(EXPECTED_ROWS["av"]), result.stdout)
        scores = parse_metric_output(result.stdout)
        for score in scores.values():
            self.assertAlmostEqual(score, 1.0, places=12)

    def test_task_selection_uses_distinct_bundled_references(self):
        for task, reference_name in REFERENCE_FILES.items():
            with self.subTest(task=task):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    prediction_path = Path(tmp_dir) / "{}_prediction.txt".format(task)
                    shutil.copyfile(REFERENCE_DIR / reference_name, prediction_path)

                    result = run_cli("--task", task, "--prediction", str(prediction_path))

                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("Task: {}".format(task), result.stdout)
                self.assertIn(reference_name, result.stdout)
                self.assertIn("Rows: {}".format(EXPECTED_ROWS[task]), result.stdout)

    def test_prediction_filename_can_infer_task_and_split(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prediction_path = Path(tmp_dir) / "BERT_NLI_dev.csv"
            shutil.copyfile(
                REFERENCE_DIR / REFERENCE_FILES["nli"],
                prediction_path,
            )

            result = run_cli("--prediction", str(prediction_path))

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Method: BERT", result.stdout)
        self.assertIn("Task: nli", result.stdout)
        self.assertIn("Split: dev", result.stdout)
        self.assertIn("Rows: {}".format(EXPECTED_ROWS["nli"]), result.stdout)

    def test_baseline_scoring_requires_task_flag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_dir = Path(tmp_dir) / "baseline"
            baseline_dir.mkdir()

            result = run_cli("--baseline-dir", str(baseline_dir))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Baseline scoring requires --task", result.stderr)

    def test_task_baseline_file_is_scored_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_dir = Path(tmp_dir) / "baseline"
            baseline_dir.mkdir()
            reference_rows = read_csv_rows(REFERENCE_DIR / REFERENCE_FILES["nli"])
            write_task_baseline_file(
                baseline_dir / "25_DEV_NLI.csv",
                reference_rows,
                {
                    "SVM": reference_rows,
                    "LSTM": reference_rows,
                    "BERT": reference_rows,
                },
            )

            result = run_cli("--task", "nli", "--baseline-dir", str(baseline_dir))

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Baseline file:", result.stdout)
        self.assertIn("Task: nli", result.stdout)
        self.assertIn("Methods: SVM, LSTM, BERT", result.stdout)
        self.assertIn("Metric", result.stdout)
        self.assertIn("accuracy_score", result.stdout)
        self.assertIn("1.000000", result.stdout)

    def test_task_baseline_reference_column_must_match_bundled_reference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_dir = Path(tmp_dir) / "baseline"
            baseline_dir.mkdir()
            reference_rows = read_csv_rows(REFERENCE_DIR / REFERENCE_FILES["av"])
            bad_reference_rows = [[1.0 - row[0]] for row in reference_rows]
            write_task_baseline_file(
                baseline_dir / "25_DEV_AV.csv",
                bad_reference_rows,
                {"SVM": reference_rows, "LSTM": reference_rows, "BERT": reference_rows},
            )

            result = run_cli("--task", "av", "--baseline-dir", str(baseline_dir))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Reference column", result.stderr)
        self.assertIn("does not match bundled reference", result.stderr)


def parse_metric_output(stdout):
    scores = {}
    for line in stdout.splitlines():
        if ": " not in line:
            continue
        name, value = line.split(": ", 1)
        if name in METRIC_NAMES:
            scores[name] = float(value)
    return scores


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCORER_MAIN)] + list(args),
        cwd=str(ROOT_DIR),
        check=False,
        capture_output=True,
        text=True,
    )


def read_csv_rows(path):
    return read_numeric_array(path)


def write_task_baseline_file(path, reference_rows, method_rows):
    method_names = list(method_rows.keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "reference"] + method_names)
        for index, reference_row in enumerate(reference_rows):
            row = [index, int(reference_row[0])]
            for method_name in method_names:
                row.append(int(method_rows[method_name][index][0]))
            writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
