from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DAY_SHEET_PATTERN = re.compile(r"\d{1,2}$")
KNOWN_DATETIME_FORMATS = (
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
)
EXCLUDE_PATH_TOKENS = (
    "sample dcr",
    "sampale dcr",
    "sample",
    "blank dcr",
    "blank",
    "formate",
    "format",
    "calibr",
    "celibr",
    "ripor",
    "cecb monthly dcr - copy",
    "~$",
)
SUPPORTED_WORKBOOK_SUFFIXES = {".xlsx", ".xls", ".xlsb"}
CANONICAL_MEASUREMENT_PREFIXES = [
    "date_time",
    "pm2_5",
    "pm10",
    "benz",
    "so2",
    "no2",
    "nox",
    "nh3",
    "temp",
    "hum",
    "sr",
    "rg",
    "co",
    "o3",
    "wd",
    "ws",
    "no",
]
REGION_ORDER = ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DCR station zip files, normalize workbook sheets, and build one "
            "combined all-region air-quality dataset."
        )
    )
    parser.add_argument(
        "--zip",
        dest="zip_specs",
        action="append",
        default=[],
        help=(
            "Path to a station zip. Optionally prefix with REGION=, for example "
            "AIIMS=/content/DCR_AIIMS.zip. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--region-root",
        action="append",
        default=[],
        help=(
            "Already-extracted station folder, optionally REGION=path. Useful when "
            "data/raw already contains station folders."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--write-excel",
        action="store_true",
        help="Also write XLSX outputs. CSV is written by default and is faster for Colab.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def to_snake_case(text: str) -> str:
    text = text.strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = text.replace("&", " and ")
    text = text.replace("%", " percent ")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def slugify_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def infer_region_label(value: str | Path) -> str:
    text = str(value).lower()
    if "aiims" in text:
        return "AIIMS"
    if "bhatagaon" in text:
        return "BHATAGAON"
    if "igkv" in text:
        return "IGKV"
    if "siltara" in text:
        return "SILTARA"
    return slugify_name(Path(str(value)).stem).upper()


def normalize_column_name(name: Any) -> str | None:
    text = clean_text(name)
    if not text:
        return None

    normalized = to_snake_case(text)
    normalized = normalized.replace("date_and_time", "date_time")
    normalized = normalized.replace("date_time_stamp", "date_time")
    normalized = normalized.replace("pm_10", "pm10")
    normalized = normalized.replace("pm_2_5", "pm2_5")
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    rename_map = {
        "date_time": "date_time",
        "date_and_time": "date_time",
        "pm2_5": "pm2_5",
        "pm_2_5": "pm2_5",
    }
    if normalized in rename_map:
        return rename_map[normalized]

    for prefix in CANONICAL_MEASUREMENT_PREFIXES:
        if normalized == prefix:
            return prefix
        if normalized.startswith(f"{prefix}_"):
            return prefix
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :]
            if suffix and suffix.replace("_", "").isalpha():
                return prefix
    return normalized


def get_day_sheets(sheet_names: list[str]) -> list[str]:
    day_sheets: list[str] = []
    for sheet in sheet_names:
        text = str(sheet).strip()
        text = re.sub(r"\.0$", "", text)
        if DAY_SHEET_PATTERN.fullmatch(text) and 1 <= int(text) <= 31:
            day_sheets.append(sheet)
    return day_sheets


def detect_header_row(raw_df: pd.DataFrame) -> int:
    max_rows = min(len(raw_df), 20)
    canonical_set = set(CANONICAL_MEASUREMENT_PREFIXES)
    for idx in range(max_rows):
        raw_values = [clean_text(value) for value in raw_df.iloc[idx].tolist()]
        normalized_values = [normalize_column_name(value) for value in raw_values]
        filtered = [value for value in normalized_values if value]
        metric_hits = sum(1 for value in filtered if value in canonical_set)
        raw_joined = " | ".join(value.lower() for value in raw_values if value)
        has_datetime = "date_time" in filtered or ("date" in raw_joined and "time" in raw_joined)
        if has_datetime and metric_hits >= 4:
            return idx
    raise ValueError("Could not find the data header row.")


def classify_granularity(time_base: str | None, source_name: str) -> str:
    token = f"{time_base or ''} {source_name}".lower()
    if "15" in token or "quat" in token or "quarter" in token or "qua" in token:
        return "quarter_hourly"
    if "hour" in token or "hou" in token:
        return "hourly"
    return "unknown"


def parse_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    text_series = series.astype("string")
    for date_format in KNOWN_DATETIME_FORMATS:
        remaining_mask = parsed.isna()
        if not remaining_mask.any():
            break
        parsed.loc[remaining_mask] = pd.to_datetime(
            text_series.loc[remaining_mask],
            format=date_format,
            errors="coerce",
        )

    remaining_mask = parsed.isna()
    if remaining_mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed.loc[remaining_mask] = pd.to_datetime(
                text_series.loc[remaining_mask],
                errors="coerce",
                dayfirst=True,
            )
    return parsed


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def align_datetime_to_granularity(data_df: pd.DataFrame) -> pd.DataFrame:
    aligned = data_df.copy()
    aligned["date_time"] = coerce_datetime_series(aligned["date_time"])
    aligned = aligned[aligned["date_time"].notna()].copy()

    frequency_by_granularity = {
        "hourly": "h",
        "quarter_hourly": "15min",
    }
    for granularity, frequency in frequency_by_granularity.items():
        mask = aligned["granularity"] == granularity
        if mask.any():
            aligned.loc[mask, "date_time"] = aligned.loc[mask, "date_time"].dt.round(frequency)
    return aligned


def workbook_year(path: Path) -> int | None:
    for part in path.parts:
        match = re.search(r"(20\d{2})", part)
        if match:
            return int(match.group(1))
    return None


def should_exclude_workbook(path: Path) -> tuple[bool, str | None]:
    lowered = str(path).lower()
    for token in EXCLUDE_PATH_TOKENS:
        if token in lowered:
            return True, f"Excluded by path token: {token}"
    return False, None


def candidate_engines(path: Path) -> list[str | None]:
    suffix = path.suffix.lower()
    preferred = {
        ".xlsx": "openpyxl",
        ".xls": "xlrd",
        ".xlsb": "pyxlsb",
    }.get(suffix)
    engines: list[str | None] = []
    if preferred:
        engines.append(preferred)
    engines.append(None)
    for engine in ("openpyxl", "xlrd", "pyxlsb"):
        if engine not in engines:
            engines.append(engine)
    return engines


def open_excel_file(workbook_path: Path) -> tuple[pd.ExcelFile, str | None]:
    failures: list[str] = []
    for engine in candidate_engines(workbook_path):
        try:
            return pd.ExcelFile(workbook_path, engine=engine), engine
        except Exception as exc:
            label = engine or "auto"
            failures.append(f"{label}: {exc}")
    raise RuntimeError("; ".join(failures))


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def unique_join(series: pd.Series) -> Any:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna().tolist():
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            values.append(text)
    if not values:
        return pd.NA
    return " | ".join(values)


def deduplicate_workbook_paths(workbook_paths: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    hash_to_paths: defaultdict[str, list[Path]] = defaultdict(list)
    skipped_duplicates: list[dict[str, str]] = []

    for path in workbook_paths:
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
        hash_to_paths[file_hash].append(path)

    unique_paths: list[Path] = []
    for group in hash_to_paths.values():
        ordered_group = sorted(group, key=lambda item: str(item).lower())
        unique_paths.append(ordered_group[0])
        for duplicate_path in ordered_group[1:]:
            skipped_duplicates.append(
                {
                    "workbook": str(duplicate_path),
                    "reason": f"Exact duplicate of {ordered_group[0]}",
                }
            )
    return sorted(unique_paths, key=lambda item: str(item).lower()), skipped_duplicates


def load_sheet_frame(
    workbook_path: Path,
    excel_file: pd.ExcelFile,
    sheet_name: str,
    dataset_root: Path,
    column_occurrences: Counter,
    column_mapping: dict[str, str],
) -> pd.DataFrame:
    raw_df = excel_file.parse(sheet_name=sheet_name, header=None)
    header_row = detect_header_row(raw_df)

    raw_headers = raw_df.iloc[header_row].tolist()
    normalized_headers: list[str | None] = []
    name_counter: Counter[str] = Counter()

    for header in raw_headers:
        normalized = normalize_column_name(header)
        if normalized:
            name_counter[normalized] += 1
            if name_counter[normalized] > 1:
                normalized = f"{normalized}_{name_counter[normalized]}"
            column_occurrences[normalized] += 1
            column_mapping[str(header)] = normalized
        normalized_headers.append(normalized)

    selected_indices = [idx for idx, value in enumerate(normalized_headers) if value]
    selected_columns = [normalized_headers[idx] for idx in selected_indices]

    data_df = raw_df.iloc[header_row + 2 :, selected_indices].copy()
    data_df.columns = selected_columns
    data_df = data_df.dropna(how="all")

    if "date_time" not in data_df.columns:
        raise ValueError(f"Sheet {sheet_name!r} is missing a normalized date_time column.")

    data_df["date_time"] = parse_datetime_series(data_df["date_time"])
    data_df = data_df[data_df["date_time"].notna()].copy()

    measurement_columns = [column for column in data_df.columns if column != "date_time"]
    for column in measurement_columns:
        data_df[column] = pd.to_numeric(data_df[column], errors="coerce")

    station_name = clean_text(raw_df.iat[0, 1] if raw_df.shape[1] > 1 else None)
    report_type = clean_text(raw_df.iat[2, 1] if raw_df.shape[1] > 1 and raw_df.shape[0] > 2 else None)
    time_base = clean_text(raw_df.iat[3, 1] if raw_df.shape[1] > 1 and raw_df.shape[0] > 3 else None)

    data_df["granularity"] = classify_granularity(time_base, workbook_path.name)
    data_df["station_name"] = station_name
    data_df["report_type"] = report_type
    data_df["time_base"] = time_base
    data_df["source_workbook"] = workbook_path.name
    data_df["source_relative_path"] = str(workbook_path.relative_to(dataset_root))
    data_df["source_sheet"] = str(sheet_name)
    data_df["source_month_folder"] = workbook_path.parent.name
    data_df["source_year"] = workbook_year(workbook_path)

    return data_df


def load_workbooks(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    discovered_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_WORKBOOK_SUFFIXES
        and not path.name.startswith("~$")
    )

    skipped_files: list[dict[str, str]] = []
    sheet_failures: list[dict[str, str]] = []
    column_occurrences: Counter = Counter()
    column_mapping: dict[str, str] = {}
    engine_counts: Counter = Counter()
    frames: list[pd.DataFrame] = []

    candidate_paths: list[Path] = []
    for workbook_path in discovered_paths:
        should_exclude, reason = should_exclude_workbook(workbook_path)
        if should_exclude:
            skipped_files.append({"workbook": str(workbook_path), "reason": reason or "Excluded"})
            continue
        candidate_paths.append(workbook_path)

    workbook_paths, skipped_duplicates = deduplicate_workbook_paths(candidate_paths)
    skipped_files.extend(skipped_duplicates)

    for workbook_path in workbook_paths:
        excel_file: pd.ExcelFile | None = None
        try:
            excel_file, engine = open_excel_file(workbook_path)
            engine_counts[engine or "auto"] += 1
            day_sheets = get_day_sheets(excel_file.sheet_names)
            if not day_sheets:
                skipped_files.append(
                    {
                        "workbook": str(workbook_path),
                        "reason": "No daily numeric sheets found.",
                    }
                )
                continue

            for sheet_name in day_sheets:
                try:
                    frame = load_sheet_frame(
                        workbook_path=workbook_path,
                        excel_file=excel_file,
                        sheet_name=sheet_name,
                        dataset_root=dataset_root,
                        column_occurrences=column_occurrences,
                        column_mapping=column_mapping,
                    )
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    sheet_failures.append(
                        {
                            "workbook": str(workbook_path),
                            "sheet": str(sheet_name),
                            "reason": str(exc),
                        }
                    )
        except Exception as exc:
            skipped_files.append({"workbook": str(workbook_path), "reason": str(exc)})
        finally:
            if excel_file is not None:
                excel_file.close()

    if not frames:
        raise RuntimeError("No data could be read from the provided DCR workbooks.")

    combined_df = pd.concat(frames, ignore_index=True, sort=False)

    diagnostics = {
        "files_discovered": len(discovered_paths),
        "files_considered": len(candidate_paths),
        "files_processed": len(workbook_paths),
        "engine_counts": dict(sorted(engine_counts.items())),
        "skipped_files": skipped_files,
        "sheet_failures": sheet_failures,
        "column_occurrences": dict(sorted(column_occurrences.items())),
        "column_mapping": dict(sorted(column_mapping.items())),
    }
    return combined_df, diagnostics


def merge_duplicate_rows(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_df = align_datetime_to_granularity(data_df)
    key_columns = ["granularity", "date_time"]
    metric_columns = [
        column
        for column in data_df.columns
        if column
        not in {
            "granularity",
            "date_time",
            "station_name",
            "report_type",
            "time_base",
            "source_workbook",
            "source_relative_path",
            "source_sheet",
            "source_month_folder",
            "source_year",
        }
    ]

    ranked_df = data_df.copy()
    ranked_df["_non_null_score"] = ranked_df[metric_columns].notna().sum(axis=1)
    ranked_df = ranked_df.sort_values(
        by=key_columns + ["_non_null_score", "source_relative_path", "source_sheet"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )

    selected_df = (
        ranked_df.drop_duplicates(subset=key_columns, keep="first")
        .copy()
        .rename(columns={"_non_null_score": "selected_row_non_null_score"})
    )

    duplicate_summary = (
        ranked_df.groupby(key_columns, as_index=False)
        .agg(
            source_row_count=("source_relative_path", "size"),
            source_file_count=("source_relative_path", lambda series: series.nunique()),
            source_workbook=("source_workbook", unique_join),
            source_relative_path=("source_relative_path", unique_join),
            source_sheet=("source_sheet", unique_join),
            source_month_folder=("source_month_folder", unique_join),
            source_year=("source_year", first_non_null),
        )
        .sort_values(key_columns, kind="stable")
    )

    merged_df = selected_df.drop(
        columns=[
            "source_workbook",
            "source_relative_path",
            "source_sheet",
            "source_month_folder",
            "source_year",
        ]
    ).merge(duplicate_summary, on=key_columns, how="left")
    merged_df["year"] = merged_df["date_time"].dt.year.astype("Int64")
    merged_df["month"] = merged_df["date_time"].dt.month.astype("Int64")
    merged_df["day"] = merged_df["date_time"].dt.day.astype("Int64")
    merged_df["hour"] = merged_df["date_time"].dt.hour.astype("Int64")
    merged_df["minute"] = merged_df["date_time"].dt.minute.astype("Int64")

    ordered_columns = [
        "date_time",
        "granularity",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        *sorted(metric_columns),
        "station_name",
        "report_type",
        "time_base",
        "source_row_count",
        "source_file_count",
        "selected_row_non_null_score",
        "source_workbook",
        "source_relative_path",
        "source_sheet",
        "source_month_folder",
        "source_year",
    ]
    return merged_df[ordered_columns].sort_values(["date_time", "granularity"], kind="stable"), duplicate_summary


def write_region_outputs(
    region: str,
    merged_df: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    output_dir: Path,
    write_excel: bool,
    dataset_root: Path,
) -> Path:
    region_dir = output_dir / "regions" / region.lower()
    region_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = slugify_name(dataset_root.name)

    combined_csv_path = region_dir / f"{dataset_slug}_combined.csv"
    duplicate_csv_path = region_dir / "duplicate_key_summary.csv"
    report_json_path = region_dir / "merge_report.json"
    column_map_csv_path = region_dir / "column_mapping.csv"

    merged_df.to_csv(combined_csv_path, index=False)
    duplicate_summary.to_csv(duplicate_csv_path, index=False)

    mapping_df = pd.DataFrame(
        [
            {"raw_column_name": raw_name, "normalized_column_name": normalized_name}
            for raw_name, normalized_name in diagnostics["column_mapping"].items()
        ]
    ).sort_values(["normalized_column_name", "raw_column_name"], kind="stable")
    mapping_df.to_csv(column_map_csv_path, index=False)

    output_paths = {
        "combined_csv": str(combined_csv_path),
        "duplicate_summary_csv": str(duplicate_csv_path),
        "column_mapping_csv": str(column_map_csv_path),
        "merge_report_json": str(report_json_path),
    }
    if write_excel:
        combined_xlsx_path = region_dir / f"{dataset_slug}_combined.xlsx"
        merged_df.to_excel(combined_xlsx_path, index=False)
        output_paths["combined_xlsx"] = str(combined_xlsx_path)

    duplicate_groups = int((duplicate_summary["source_row_count"] > 1).sum())
    report_payload = {
        "region": region,
        "dataset_root": str(dataset_root),
        "rows_after_merge": int(len(merged_df)),
        "granularity_counts": {
            str(key): int(value)
            for key, value in merged_df["granularity"].value_counts(dropna=False).to_dict().items()
        },
        "duplicate_timestamp_groups": duplicate_groups,
        "max_rows_combined_per_timestamp": int(duplicate_summary["source_row_count"].max()),
        "outputs": output_paths,
        **diagnostics,
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return combined_csv_path


def safe_extract_zip(zip_path: Path, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_top_levels: set[Path] = set()
    raw_dir_resolved = raw_dir.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if not member_path.parts:
                continue
            target_path = (raw_dir / member_path).resolve()
            if raw_dir_resolved not in target_path.parents and target_path != raw_dir_resolved:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            archive.extract(member, raw_dir)
            extracted_top_levels.add(raw_dir / member_path.parts[0])
    return sorted(extracted_top_levels, key=lambda item: str(item).lower())


def parse_path_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        region, path = spec.split("=", 1)
        return region.strip().upper(), Path(path.strip().strip('"'))
    return None, Path(spec.strip().strip('"'))


def discover_region_roots(args: argparse.Namespace) -> dict[str, Path]:
    region_roots: dict[str, Path] = {}

    for zip_spec in args.zip_specs:
        explicit_region, zip_path = parse_path_spec(zip_spec)
        zip_path = zip_path.expanduser().resolve()
        top_levels = safe_extract_zip(zip_path, args.raw_dir)
        for root in top_levels:
            region = explicit_region or infer_region_label(root)
            region_roots[region] = root.resolve()

    for root_spec in args.region_root:
        explicit_region, root_path = parse_path_spec(root_spec)
        root_path = root_path.expanduser().resolve()
        region = explicit_region or infer_region_label(root_path)
        region_roots[region] = root_path

    if not region_roots and args.raw_dir.exists():
        for child in args.raw_dir.iterdir():
            if child.is_dir():
                region_roots[infer_region_label(child)] = child.resolve()

    if not region_roots:
        raise RuntimeError(
            "No station roots found. Pass --zip for the raw DCR archives or --region-root for extracted folders."
        )

    return dict(sorted(region_roots.items(), key=lambda item: REGION_ORDER.index(item[0]) if item[0] in REGION_ORDER else 99))


def combine_region_csvs(region_csvs: dict[str, Path], output_dir: Path, write_excel: bool) -> Path:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for region, csv_path in region_csvs.items():
        df = pd.read_csv(csv_path, parse_dates=["date_time"], low_memory=False)
        df["date_time"] = coerce_datetime_series(df["date_time"])
        df = align_datetime_to_granularity(df)
        df["region"] = region
        frames.append(df)
        granularity_counts = df["granularity"].value_counts(dropna=False).to_dict()
        summary_rows.append(
            {
                "region": region,
                "rows_after_merge": int(len(df)),
                "hourly_rows": int(granularity_counts.get("hourly", 0)),
                "quarter_hourly_rows": int(granularity_counts.get("quarter_hourly", 0)),
                "unknown_rows": int(granularity_counts.get("unknown", 0)),
                "source_csv": str(csv_path),
            }
        )

    combined_df = pd.concat(frames, ignore_index=True, sort=False)
    ordered_columns = ["region"] + [column for column in combined_df.columns if column != "region"]
    combined_df = combined_df[ordered_columns].sort_values(
        ["region", "date_time", "granularity"], kind="stable"
    )
    summary_df = pd.DataFrame(summary_rows).sort_values("region", kind="stable")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = output_dir / "all_regions_combined.csv"
    summary_csv = output_dir / "all_regions_dataset_summary.csv"
    report_json = output_dir / "all_regions_combined_report.json"

    combined_df.to_csv(combined_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    if write_excel:
        combined_df.to_excel(output_dir / "all_regions_combined.xlsx", index=False)

    report_payload = {
        "rows_after_merge": int(len(combined_df)),
        "regions": sorted(region_csvs),
        "granularity_counts": {
            str(key): int(value)
            for key, value in combined_df["granularity"].value_counts(dropna=False).to_dict().items()
        },
        "region_summary": summary_df.to_dict(orient="records"),
        "combined_csv": str(combined_csv),
        "summary_csv": str(summary_csv),
    }
    report_json.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return combined_csv


def main() -> int:
    args = parse_args()
    region_roots = discover_region_roots(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    region_csvs: dict[str, Path] = {}
    for region, root in region_roots.items():
        print(f"\nPreparing {region}: {root}")
        combined_df, diagnostics = load_workbooks(root)
        merged_df, duplicate_summary = merge_duplicate_rows(combined_df)
        region_csv = write_region_outputs(
            region=region,
            merged_df=merged_df,
            duplicate_summary=duplicate_summary,
            diagnostics=diagnostics,
            output_dir=args.output_dir,
            write_excel=args.write_excel,
            dataset_root=root,
        )
        region_csvs[region] = region_csv
        print(
            f"{region}: {len(merged_df)} rows, "
            f"{diagnostics['files_processed']} files processed, "
            f"{len(diagnostics['sheet_failures'])} sheet failures"
        )

    combined_csv = combine_region_csvs(region_csvs, args.output_dir, write_excel=args.write_excel)
    print(f"\nCombined dataset: {combined_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
