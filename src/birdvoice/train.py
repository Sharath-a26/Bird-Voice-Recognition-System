"""Minimal, reproducible training entry point for the BirdCLEF focal-recording baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import crop_or_pad, log_mel, mono_resample
from .metrics import expected_calibration_error
from .model import BirdCNN


class FocalRecordingDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, root: Path, labels: dict[str, int], config: dict, train: bool) -> None:
        self.rows, self.root, self.labels, self.config, self.train = rows.reset_index(drop=True), root, labels, config, train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, data, audio = self.rows.iloc[index], self.config["data"], self.config["audio"]
        waveform, rate = torchaudio.load(self.root / row.audio_path)
        waveform = mono_resample(waveform, rate, data["sample_rate"])
        waveform = crop_or_pad(waveform, int(data["sample_rate"] * data["clip_seconds"]), self.train)
        mel = log_mel(waveform, data["sample_rate"], **audio)
        if self.train:
            mel = torchaudio.transforms.FrequencyMasking(12)(mel)
            mel = torchaudio.transforms.TimeMasking(24)(mel)
        return mel, torch.tensor(self.labels[row.primary_label])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval(); actual, predicted, probabilities = [], [], []
    with torch.no_grad():
        for mel, label in loader:
            logits, _ = model(mel.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            actual.extend(label.tolist()); predicted.extend(logits.argmax(1).cpu().tolist())
    probability_array, labels = np.concatenate(probabilities), np.asarray(actual)
    one_hot = np.eye(probability_array.shape[1])[labels]
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "top1_accuracy": float(np.mean(labels == np.asarray(predicted))),
        "macro_pr_auc": float(average_precision_score(one_hot, probability_array, average="macro")),
        "expected_calibration_error": expected_calibration_error(probability_array, labels),
    }, labels, probability_array


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()); seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = config["data"]; training = config["training"]
    rows = pd.read_csv(data["metadata_csv"]); labels = json.loads(Path(data["label_map"]).read_text())
    train_rows, valid_rows = train_test_split(rows, test_size=data["validation_fraction"], random_state=seed, stratify=rows.primary_label)
    root = Path(data["audio_root"])
    train_loader = DataLoader(FocalRecordingDataset(train_rows, root, labels, config, True), batch_size=training["batch_size"], shuffle=True, num_workers=training["num_workers"])
    valid_loader = DataLoader(FocalRecordingDataset(valid_rows, root, labels, config, False), batch_size=training["batch_size"], num_workers=training["num_workers"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = BirdCNN(len(labels), **config["model"]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    output = Path(training["output_dir"]); output.mkdir(parents=True, exist_ok=True); best = -1.0
    train_rows.to_csv(output / "train_split.csv", index=False); valid_rows.to_csv(output / "validation_split.csv", index=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    for epoch in range(training["epochs"]):
        model.train()
        for mel, label in train_loader:
            logits, _ = model(mel.to(device)); loss = nn.functional.cross_entropy(logits, label.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        metrics, _, _ = evaluate(model, valid_loader, device); score = metrics["macro_f1"]
        print(f"epoch={epoch + 1} macro_f1={score:.4f} ece={metrics['expected_calibration_error']:.4f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": config, "metrics": metrics}, output / "best.pt")
            (output / "metrics.json").write_text(json.dumps({**metrics, "seed": seed}, indent=2))


if __name__ == "__main__":
    main()
