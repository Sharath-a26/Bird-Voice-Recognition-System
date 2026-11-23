# Research protocol

## Dataset decision

The primary dataset is BirdCLEF 2021, released in 2021. It contains over 60,000 focal recordings spanning 397 species, and labelled soundscapes in five-second windows. This lets the study test both isolated recordings and overlapping, noisy field audio. The Kaggle download is 42.18 GB and subject to competition rules, so this repository stores only scripts and derived manifests, never the dataset.

## Evaluation

Use species-stratified development splits for fast iteration and an audio-recorder or location-held-out split for the final report. Recordings from the same source can otherwise inflate performance. Report macro F1, micro F1 for soundscape labels, PR-AUC, per-class recall, expected calibration error, confusion matrices, rare-class performance, and bootstrap confidence intervals.

## Algorithm

Resample each clip to 32 kHz, normalize amplitude, select a five-second window, and compute a 128-bin log-mel spectrogram. A residual CNN predicts the class. Time and frequency masking improve robustness to partial calls and background interference. An autocorrelation estimator measures dominant periodic frequency; robust per-species pitch ranges, fit only on training data, give an interpretable consistency score. This supports calibration and ablations rather than a hard pitch-matching decision.

## Scope limits

Species recognition is the evaluated outcome. Call type and sex labels are not available at sufficient quality in this benchmark, so the project will not make those claims. A local population trend requires repeated sampling, recorder effort, site metadata, and detection-corrected ecological modelling; classifier counts alone do not estimate population.
