# Implementation guide

## Expected BirdCLEF 2021 layout

The code expects the original Kaggle directory structure:

```text
data/raw/birdclef-2021/
├── train_metadata.csv
├── train_short_audio/<species-code>/<file>.ogg
└── train_soundscapes/
```

No audio, credentials, predictions, model weights, or fabricated reports belong in version control.

## Reproducible run order

```bash
python scripts/prepare_birdclef.py --input data/raw/birdclef-2021 --output data/processed/birdclef-2021
python scripts/eda.py --metadata data/processed/birdclef-2021/metadata.csv --output artifacts/eda
python scripts/create_splits.py --metadata data/processed/birdclef-2021/metadata.csv --output artifacts/splits --strategy stratified
birdvoice-train --config configs/birdclef2021_baseline.yaml
python scripts/build_pitch_priors.py --metadata artifacts/splits/train.csv --audio-root data/raw/birdclef-2021/train_short_audio --output artifacts/pitch_priors.csv
```

Use the source-held-out split as the more demanding robustness check. It can omit rare classes from one side, so report coverage and do not compare it directly to the stratified-development score.

## Interpretation policy

Until commands run on the original dataset, `artifacts/` must contain no claimed accuracy, F1, calibration, or ecological findings. The generated code only specifies how these quantities will be measured.
