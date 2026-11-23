# Roadmap

1. Prepare the frozen BirdCLEF 2021 manifest and run data-quality checks.
2. Train the CNN baseline and publish a full reproducibility record: split IDs, seed, config, environment, metrics, and model checksum.
3. Test pitch-consistency scoring with an ablation against the CNN-only model and report failures, especially rare species and overlapping vocalizations.
4. Validate on a small, manually reviewed campus recording set. Publish precision, recall, calibration, and abstention behaviour before making any ecological statement.
5. Collect repeated fixed-site recordings with weather, time, equipment, and effort logs. Only then investigate occupancy or occurrence, with sensitive locations generalized.
6. Benchmark an ONNX/TFLite export on a target phone before describing the application as real-time.

For an SOP, the credible framing is: “I began with a second-year conservation-ML idea and rebuilt it as a reproducible study that separates audio classification from ecological inference.” Attach measured results only after they exist.
