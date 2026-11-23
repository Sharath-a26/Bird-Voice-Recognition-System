import math

import torch

from birdvoice.audio import crop_or_pad, dominant_pitch_hz


def test_crop_or_pad_length() -> None:
    assert crop_or_pad(torch.ones(1, 10), 16, False).shape[-1] == 16
    assert crop_or_pad(torch.ones(1, 32), 16, False).shape[-1] == 16


def test_pitch_estimate_for_sine() -> None:
    rate, expected = 32000, 440
    time = torch.arange(rate) / rate
    estimate = dominant_pitch_hz(torch.sin(2 * math.pi * expected * time).unsqueeze(0), rate)
    assert abs(estimate - expected) < 5
