#!/usr/bin/env python3
"""Generate versionable BirdCLEF metadata-quality reports before model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def first_present(frame: pd.DataFrame, choices: list[str]) -> str | None:
    return next((name for name in choices if name in frame.columns), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = frame.primary_label.value_counts().rename_axis("species").reset_index(name="recordings")
    counts.to_csv(args.output / "species_counts.csv", index=False)
    source_column = first_present(frame, ["author", "recordist", "recodist"])
    summary = {
        "recordings": len(frame),
        "species": int(frame.primary_label.nunique()),
        "median_recordings_per_species": float(counts.recordings.median()),
        "species_with_fewer_than_25_recordings": int((counts.recordings < 25).sum()),
        "metadata_columns": frame.columns.tolist(),
        "source_column": source_column,
        "unique_sources": int(frame[source_column].nunique()) if source_column else None,
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items() if value},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    top = counts.head(30).sort_values("recordings")
    plt.figure(figsize=(10, 8))
    plt.barh(top.species, top.recordings)
    plt.title("30 species with most focal recordings")
    plt.xlabel("Recordings")
    plt.tight_layout()
    plt.savefig(args.output / "top_species.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
