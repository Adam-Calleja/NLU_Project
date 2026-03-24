# NLU Local Scorer

This repository exposes a standalone Python scorer for the bundled `NLI`, `AV`, and `ED` dev references. The CLI supports two workflows:

- score one prediction file against a bundled task reference
- score the bundled baseline methods for one task from `baseline/`

## Project Layout

- `local_scorer/main.py`: CLI entrypoint.
- `local_scorer/reference_data/`: embedded `NLI`, `AV`, and `ED` dev solutions.
- `local_scorer/metric.txt`: metric order used in the CLI output.
- `tests/test_local_scorer.py`: regression tests for task selection, header parsing, shape validation, and perfect-match scoring.

## Install With Conda

Run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate nlu-local-scorer
```

If you change dependencies later, update the environment with:

```bash
conda env update -f environment.yml --prune
```

## Dependencies

- `environment.yml` is the source of truth for the install.
- The conda environment installs Python, `pip`, `setuptools`, `wheel`, `numpy`, and `scikit-learn`.
- The package itself is installed from the repo with `pip -e .` during `conda env create`.
- If you add any extra package later, put it in `environment.yml` and keep this README in sync.

## Scoring Backend

- The scorer now prefers `numpy` and `scikit-learn` for metric computation.
- If those packages are unavailable in a bare Python, the code still falls back to the built-in implementation so the repo remains usable.

## Usage

### 1. Score One Prediction File

Use the same command pattern for all three tasks. Only the `--task` value and prediction path change.

General form:

```bash
nlu-score --task {nli|av|ed} --prediction path/to/predictions.txt
```

Examples:

```bash
nlu-score --task nli --prediction path/to/nli_predictions.txt
nlu-score --task av --prediction path/to/av_predictions.txt
nlu-score --task ed --prediction path/to/ed_predictions.txt
```

These three commands behave the same way. The only difference is which bundled dev reference is loaded:

- `nli` -> `NLU_SharedTask_NLI_dev.solution`
- `av` -> `NLU_SharedTask_AV_dev.solution`
- `ed` -> `NLU_SharedTask_ED_dev.solution`

You can also let the scorer infer task and split from a filename like `BERT_NLI_dev.csv`:

```bash
nlu-score --prediction path/to/BERT_NLI_dev.csv
```

`--split dev` is optional because only the dev split is supported right now:

```bash
nlu-score --task av --split dev --prediction path/to/BERT_AV_dev.csv
```

Task row counts must match the selected bundled reference:

- `nli`: 6736 rows
- `av`: 5993 rows
- `ed`: 5926 rows

### 2. Score Bundled Baseline Methods

If `--prediction` is omitted, the CLI scores the bundled baseline table for the selected task.

Examples:

```bash
nlu-score --task nli
nlu-score --task av
nlu-score --task ed
```

This mode loads the matching file from `baseline/`, such as:

- `baseline/25_DEV_NLI.csv`
- `baseline/25_DEV_AV.csv`
- `baseline/25_DEV_ED.csv`

Each baseline file contains:

- one `reference` column
- one column per baseline method, such as `SVM`, `LSTM`, and `BERT`

The scorer validates the `reference` column against the bundled dev reference, then prints a comparison table of metrics for all methods in that task.

## Input Rules

- The prediction file must contain numeric values in plain text.
- One non-numeric header line is allowed and will be skipped automatically.
- One-column and multi-column numeric tables are accepted.
- Comma-separated `.csv` files are accepted.
- The scorer always uses the bundled dev reference data for `NLI`, `AV`, and `ED`.
- `--task` only changes which bundled reference file is used.
- The parsed prediction shape must exactly match the selected reference shape.
- If `--prediction` is omitted, you must provide `--task`; the CLI then loads the matching task baseline file from `baseline/`.
- Baseline files should be named like `LABEL_SPLIT_TASK.csv`, for example `25_DEV_NLI.csv`.
- Task baseline files must contain one `reference` column plus one or more baseline-method columns such as `SVM`, `LSTM`, and `BERT`.
- In task baseline mode, the scorer validates that the CSV `reference` column matches the bundled reference before reporting metrics for each baseline method.

## Run Tests

After the environment is ready, run:

```bash
python -m unittest tests.test_local_scorer -v
```
