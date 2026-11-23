#!/usr/bin/env python3
"""Estimate robust per-species pitch priors from training audio only."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio

from birdvoice.audio import crop_or_pad, dominant_pitch_hz, mono_resample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-species", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    args = parser.parse_args()
    rows = pd.read_csv(args.metadata).groupby("primary_label", group_keys=False).head(args.max_per_species)
    results = []
    for label, group in rows.groupby("primary_label"):
        pitches = []
        for row in group.itertuples():
            waveform, rate = torchaudio.load(args.audio_root / row.audio_path)
            waveform = mono_resample(waveform, rate, args.sample_rate)
            waveform = crop_or_pad(waveform, int(args.sample_rate * args.clip_seconds), train=False)
            pitch = dominant_pitch_hz(waveform, args.sample_rate)
            if not torch.isnan(torch.tensor(pitch)):
                pitches.append(pitch)
        if pitches:
            series = pd.Series(pitches)
            median = float(series.median())
            mad = float((series - median).abs().median())
            results.append({"primary_label": label, "median_pitch_hz": median, "mad_pitch_hz": mad, "samples": len(pitches)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
