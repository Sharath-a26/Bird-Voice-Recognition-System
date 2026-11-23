"""Run a trained BirdCNN checkpoint on one audio recording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from .audio import crop_or_pad, dominant_pitch_hz, log_mel, mono_resample
from .model import BirdCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    label_map = json.loads(args.labels.read_text())
    labels = {index: label for label, index in label_map.items()}
    model = BirdCNN(len(labels), **config["model"])
    model.load_state_dict(saved["model"]); model.eval()
    waveform, source_rate = torchaudio.load(args.audio)
    data, audio = config["data"], config["audio"]
    waveform = mono_resample(waveform, source_rate, data["sample_rate"])
    waveform = crop_or_pad(waveform, int(data["sample_rate"] * data["clip_seconds"]), train=False)
    mel = log_mel(waveform, data["sample_rate"], **audio)
    with torch.no_grad():
        logits, _ = model(mel.unsqueeze(0)); probability = torch.softmax(logits, 1).squeeze(0)
    values, indexes = torch.topk(probability, min(args.top_k, len(labels)))
    output = {
        "dominant_pitch_hz": dominant_pitch_hz(waveform, data["sample_rate"]),
        "predictions": [{"species": labels[int(index)], "confidence": float(value)} for value, index in zip(values, indexes)],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
