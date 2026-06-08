from __future__ import annotations

from pathlib import Path


def discover_station_archives(raw_dir: Path) -> list[Path]:
    return sorted(Path(raw_dir).glob("*.zip"), key=lambda path: path.name.lower())


def discover_processed_dataset(processed_dir: Path) -> Path:
    dataset_path = Path(processed_dir) / "all_regions_combined.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
    return dataset_path
