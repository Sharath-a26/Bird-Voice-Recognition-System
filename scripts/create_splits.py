#!/usr/bin/env python3
"""Create reproducible BirdCLEF development or source-held-out splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def source_column(frame: pd.DataFrame) -> str | None:
    return next((name for name in ("author", "recordist", "recodist") if name in frame), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--strategy", choices=("stratified", "source-held-out"), default="stratified")
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata)
    if args.strategy == "stratified":
        train, validation = train_test_split(
            frame, test_size=args.validation_fraction, random_state=args.seed, stratify=frame.primary_label
        )
        split_column = None
    else:
        split_column = source_column(frame)
        if not split_column:
            raise ValueError("No author/recordist column exists for source-held-out splitting")
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
        train_index, validation_index = next(splitter.split(frame, groups=frame[split_column].fillna("unknown")))
        train, validation = frame.iloc[train_index], frame.iloc[validation_index]
    args.output.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output / "train.csv", index=False)
    validation.to_csv(args.output / "validation.csv", index=False)
    manifest = {
        "strategy": args.strategy, "seed": args.seed, "validation_fraction": args.validation_fraction,
        "source_column": split_column, "train_rows": len(train), "validation_rows": len(validation),
        "train_species": int(train.primary_label.nunique()), "validation_species": int(validation.primary_label.nunique()),
    }
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
