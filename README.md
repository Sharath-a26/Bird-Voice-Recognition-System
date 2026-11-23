# Bird Voice Recognition

A reproducible bioacoustics project built around a historically appropriate snapshot: **BirdCLEF 2021**. The benchmark was released in 2021 and contains more than 60,000 focal recordings from 397 bird species, plus labelled soundscapes. That makes it a credible foundation for a project begun in a second-year timeframe without claiming use of a later dataset.

## Honest project narrative

The original idea and prototype began in second year. This repository preserves that original scope, then documents a current, more rigorous rebuild. Do not describe current experiments, metrics, or deployment work as historical results.

## Method

Field audio → 32 kHz mono waveform → 5-second windows → log-mel spectrogram → residual CNN species classifier. An autocorrelation pitch estimator creates a species-specific pitch-consistency score for error analysis and calibration experiments. It never overrides the CNN with a simplistic frequency rule.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
kaggle competitions download -c birdclef-2021 -p data/raw/birdclef-2021
unzip data/raw/birdclef-2021/birdclef-2021.zip -d data/raw/birdclef-2021
python scripts/prepare_birdclef.py --input data/raw/birdclef-2021 --output data/processed/birdclef-2021
birdvoice-train --config configs/birdclef2021_baseline.yaml
```

Kaggle requires acceptance of its competition rules. Never commit audio, credentials, checkpoints, or location data for sensitive species.

## Layout

`configs/` stores experiments. `docs/` contains the research protocol and roadmap. `scripts/` prepares data. `src/birdvoice/` contains reusable audio and model code. `tests/` verifies numerical primitives.

Read [the implementation guide](docs/implementation.md) for the expected BirdCLEF layout, exact command sequence, leakage-aware splits, and the policy against reporting unrun experiments as results.
