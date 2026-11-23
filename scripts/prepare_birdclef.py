#!/usr/bin/env python3
"""Create a lightweight, versionable manifest from the BirdCLEF 2021 download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.input / "train_metadata.csv")
    required = {"primary_label", "filename"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    metadata["audio_path"] = metadata.apply(
        lambda row: str(Path(row.primary_label) / row.filename), axis=1
    )
    labels = {label: index for index, label in enumerate(sorted(metadata.primary_label.unique()))}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output / "metadata.csv", index=False)
    (args.output / "label_map.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    print(f"Wrote {len(metadata):,} rows and {len(labels)} species")


if __name__ == "__main__":
    main()
