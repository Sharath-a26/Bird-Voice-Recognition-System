from __future__ import annotations

import math

import torch
import torchaudio


def mono_resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Convert [channels, samples] audio to normalized mono."""
    if waveform.ndim != 2:
        raise ValueError("Expected [channels, samples]")
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return waveform / waveform.abs().max().clamp_min(1e-8)


def crop_or_pad(waveform: torch.Tensor, samples: int, train: bool) -> torch.Tensor:
    if waveform.shape[-1] < samples:
        return torch.nn.functional.pad(waveform, (0, samples - waveform.shape[-1]))
    maximum = waveform.shape[-1] - samples
    start = int(torch.randint(maximum + 1, (1,)).item()) if train else maximum // 2
    return waveform[..., start : start + samples]


def log_mel(waveform: torch.Tensor, sample_rate: int, *, n_fft: int, hop_length: int,
            n_mels: int, f_min: float, f_max: float) -> torch.Tensor:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        f_min=f_min, f_max=f_max, power=2.0,
    ).to(waveform.device)
    return torch.log(transform(waveform).clamp_min(1e-10))


def dominant_pitch_hz(waveform: torch.Tensor, sample_rate: int, low_hz: float = 80,
                      high_hz: float = 2000) -> float:
    """Estimate the strongest autocorrelation period; NaN signals no usable periodicity."""
    signal = waveform.flatten().float()
    signal -= signal.mean()
    if signal.norm() < 1e-6:
        return math.nan
    autocorrelation = torch.fft.irfft(torch.fft.rfft(signal, n=2 * len(signal)).abs().square())[:len(signal)]
    min_lag, max_lag = int(sample_rate / high_hz), int(sample_rate / low_hz)
    if max_lag >= len(autocorrelation):
        return math.nan
    autocorrelation[0] = 0
    lag = int(torch.argmax(autocorrelation[min_lag:max_lag + 1])) + min_lag
    return sample_rate / lag


def pitch_consistency(pitch_hz: float, center_hz: float, spread_hz: float) -> float:
    """Return a bounded diagnostic score relative to a robust species pitch prior."""
    if math.isnan(pitch_hz) or math.isnan(center_hz) or spread_hz <= 0:
        return math.nan
    z_score = abs(pitch_hz - center_hz) / spread_hz
    return float(math.exp(-0.5 * z_score * z_score))
