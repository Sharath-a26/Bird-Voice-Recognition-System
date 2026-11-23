import torch

from birdvoice.model import BirdCNN


def test_model_output_shapes() -> None:
    logits, pitch = BirdCNN(classes=397)(torch.rand(2, 1, 128, 313))
    assert logits.shape == (2, 397)
    assert pitch.shape == (2,)
