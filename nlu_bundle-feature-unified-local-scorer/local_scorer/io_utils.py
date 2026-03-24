import csv
from pathlib import Path


REFERENCE_FILENAMES = {
    "nli": "NLU_SharedTask_NLI_dev.solution",
    "av": "NLU_SharedTask_AV_dev.solution",
    "ed": "NLU_SharedTask_ED_dev.solution",
}

REFERENCE_DIR = Path(__file__).resolve().parent / "reference_data"


def resolve_reference_path(task):
    if task:
        normalized_task = normalize_task_name(task)
        path = REFERENCE_DIR / REFERENCE_FILENAMES[normalized_task]
    else:
        raise ValueError("A task is required to resolve the bundled reference data.")

    if not path.exists():
        raise FileNotFoundError("Reference file not found: {}".format(path))
    return path.resolve()


def normalize_task_name(task):
    normalized = str(task).strip().lower()
    if normalized not in REFERENCE_FILENAMES:
        raise ValueError("Unsupported task: {}".format(task))
    return normalized


def normalize_split_name(split):
    normalized = str(split).strip().lower()
    if normalized != "dev":
        raise ValueError("Only the dev split is supported: {}".format(split))
    return normalized


def infer_prediction_metadata(path):
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            "Could not infer method, task, and split from filename {}.".format(Path(path).name)
        )

    task = normalize_task_name(parts[-2])
    split = normalize_split_name(parts[-1])
    method = "_".join(parts[:-2]).strip()
    if not method:
        raise ValueError("Could not infer method name from filename {}.".format(Path(path).name))

    return {"method": method, "task": task, "split": split}


def infer_baseline_table_metadata(path):
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            "Could not infer split and task from baseline filename {}.".format(Path(path).name)
        )

    split = normalize_split_name(parts[-2])
    task = normalize_task_name(parts[-1])
    label = "_".join(parts[:-2]).strip()
    if not label:
        raise ValueError(
            "Could not infer baseline label from filename {}.".format(Path(path).name)
        )

    return {"label": label, "task": task, "split": split}


def find_baseline_table_path(baseline_dir, task, split="dev"):
    baseline_path = Path(baseline_dir).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline directory not found: {}".format(baseline_path))
    if not baseline_path.is_dir():
        raise ValueError("Baseline path is not a directory: {}".format(baseline_path))

    chosen_task = normalize_task_name(task)
    chosen_split = normalize_split_name(split)
    matches = []

    for path in sorted(baseline_path.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        try:
            metadata = infer_baseline_table_metadata(path)
        except ValueError:
            continue
        if metadata["task"] == chosen_task and metadata["split"] == chosen_split:
            matches.append(path.resolve())

    if not matches:
        raise FileNotFoundError(
            "No baseline file found for task {} and split {} in {}.".format(
                chosen_task, chosen_split, baseline_path
            )
        )
    if len(matches) > 1:
        raise ValueError(
            "Found multiple baseline files for task {} and split {}: {}".format(
                chosen_task, chosen_split, ", ".join(str(path) for path in matches)
            )
        )
    return matches[0]


def read_baseline_table(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("File not found: {}".format(file_path))

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("File is empty: {}".format(file_path))

        column_specs = _resolve_baseline_columns(header, file_path)
        columns = {name: [] for _, name in column_specs}

        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    "Inconsistent column count in {} at line {}.".format(
                        file_path, line_number
                    )
                )

            row_count += 1
            for index, name in column_specs:
                cell = row[index].strip()
                try:
                    value = float(cell)
                except ValueError:
                    raise ValueError(
                        "Unable to parse numeric data in {} at line {}, column '{}'.".format(
                            file_path, line_number, name
                        )
                    )
                columns[name].append([value])

    if row_count == 0:
        raise ValueError("No numeric data found in {}".format(file_path))

    reference_name = None
    method_names = []
    for _, name in column_specs:
        if name.lower() == "reference":
            reference_name = name
        else:
            method_names.append(name)

    return {
        "reference_name": reference_name,
        "reference": columns[reference_name],
        "methods": [(name, columns[name]) for name in method_names],
    }


def read_numeric_array(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("File not found: {}".format(file_path))

    data_lines = _load_data_lines(file_path)
    rows = []
    column_count = None

    for line_number, line in data_lines:
        values = _parse_numeric_line(line, file_path, line_number)
        if column_count is None:
            column_count = len(values)
        elif len(values) != column_count:
            raise ValueError(
                "Inconsistent column count in {} at line {}.".format(
                    file_path, line_number
                )
            )
        rows.append(values)

    if not rows:
        raise ValueError("No numeric data found in {}".format(file_path))

    return rows


def get_shape(rows):
    if not rows:
        return (0, 0)
    return (len(rows), len(rows[0]))


def validate_same_shape(solution, prediction, solution_path, prediction_path):
    solution_shape = get_shape(solution)
    prediction_shape = get_shape(prediction)
    if solution_shape != prediction_shape:
        raise ValueError(
            "Prediction shape {} from {} does not match reference shape {} from {}.".format(
                prediction_shape, prediction_path, solution_shape, solution_path
            )
        )


def _load_data_lines(path):
    with path.open("r", encoding="utf-8") as handle:
        raw_lines = [line.rstrip("\n") for line in handle]

    if not raw_lines:
        raise ValueError("File is empty: {}".format(path))

    first_content_index = None
    first_content_line = None
    for index, line in enumerate(raw_lines):
        stripped = line.strip()
        if stripped:
            first_content_index = index
            first_content_line = stripped
            break

    if first_content_index is None:
        raise ValueError("File contains no numeric data: {}".format(path))

    if _is_numeric_line(first_content_line):
        start_index = first_content_index
    else:
        start_index = first_content_index + 1

    data_lines = []
    for offset, raw_line in enumerate(raw_lines[start_index:], start=start_index + 1):
        stripped = raw_line.strip()
        if stripped:
            data_lines.append((offset, stripped))

    if not data_lines:
        raise ValueError("No numeric data found after header in {}".format(path))

    if not _is_numeric_line(data_lines[0][1]):
        raise ValueError("Unable to parse numeric data from {}".format(path))

    return data_lines


def _is_numeric_line(line):
    try:
        _parse_numeric_tokens(line)
    except ValueError:
        return False
    return True


def _parse_numeric_line(line, path, line_number):
    try:
        return _parse_numeric_tokens(line)
    except ValueError:
        raise ValueError(
            "Unable to parse numeric data in {} at line {}.".format(path, line_number)
        )


def _parse_numeric_tokens(line):
    delimiter = _detect_delimiter(line)
    if delimiter:
        tokens = [token.strip() for token in line.split(delimiter)]
    else:
        tokens = line.split()
    if not tokens:
        raise ValueError("Blank line")
    return [float(token) for token in tokens]


def _detect_delimiter(line):
    if "," in line:
        return ","
    return None


def _resolve_baseline_columns(header, path):
    if not header:
        raise ValueError("Missing header row in {}".format(path))

    column_specs = []
    for index, raw_name in enumerate(header):
        name = raw_name.strip()
        if index == 0 and name.lower() in ("", "index", "id"):
            continue
        if not name:
            raise ValueError(
                "Unnamed baseline column in {} at header position {}.".format(path, index + 1)
            )
        column_specs.append((index, name))

    if not column_specs:
        raise ValueError("No usable baseline columns found in {}".format(path))

    reference_columns = [name for _, name in column_specs if name.lower() == "reference"]
    if len(reference_columns) != 1:
        raise ValueError(
            "Baseline file {} must contain exactly one 'reference' column.".format(path)
        )

    method_names = [name for _, name in column_specs if name.lower() != "reference"]
    if not method_names:
        raise ValueError(
            "Baseline file {} must contain at least one method column.".format(path)
        )

    return column_specs
