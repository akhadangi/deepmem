import json
from pathlib import Path
import pytest
import torch

from deepmem.model import (LosslessSubbandByteLadderConfig, LosslessSubbandByteLadderLM,
                           count_parameters, chunked_gated_linear_recurrence)

ROOT = Path(__file__).resolve().parents[1]


def tiny(**kwargs):
    data = json.loads((ROOT / "configs/tiny_smoke.json").read_text())
    data.update(kwargs)
    return LosslessSubbandByteLadderConfig(**data)


def test_reference_shapes_and_parameters():
    cfg = LosslessSubbandByteLadderConfig(**json.loads((ROOT / "configs/paper_r2.json").read_text()))
    # This test instantiates ~400 MB of parameters; run only as an explicit smoke check.
    assert cfg.lengths == [2048, 1024, 512, 256, 128, 64, 32, 16]
    assert cfg.sem_widths == [1024, 512, 256, 128, 128, 128, 128, 128]


def test_carrier_invertibility():
    torch.manual_seed(7)
    model = LosslessSubbandByteLadderLM(tiny())
    x = torch.randint(0, 256, (1, 32))
    c0 = model.carrier(x)
    c = c0
    for down in model.downs:
        c, _ = down.carrier_lift(c)
    assert torch.allclose(model.reconstruct_carrier_from_bottom(c), c0, atol=1e-6)


def test_next_byte_and_causality():
    torch.manual_seed(7)
    model = LosslessSubbandByteLadderLM(tiny()).eval()
    x = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        out = model(x, x)
        assert out["logits"].shape == (1, 32, 256)
        assert torch.isfinite(out["loss"])
        alt = x.clone()
        alt[:, 16:] = (alt[:, 16:] + 17) % 256
        pred = model(alt)["logits"]
        assert torch.allclose(pred[:, :16], out["logits"][:, :16], atol=2e-5)


def test_scalar_recurrence_matches_stepwise():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    g = -torch.rand(1, 2, 8) * 0.3
    actual = chunked_gated_linear_recurrence(q, k, v, g, 4)
    states = torch.zeros(1, 2, 4, 4)
    expected = []
    for i in range(8):
        states = g[..., i].exp()[..., None, None] * states + torch.einsum('bhd,bhe->bhde', k[..., i, :], v[..., i, :])
        expected.append(torch.einsum('bhde,bhd->bhe', states, q[..., i, :]))
    assert torch.allclose(actual, torch.stack(expected, dim=2), atol=2e-5)


def test_vector_recurrence_matches_stepwise():
    torch.manual_seed(1)
    q, k, v = [torch.randn(1, 2, 8, 4) for _ in range(3)]
    g = -torch.rand(1, 2, 8, 4) * 0.1
    actual = chunked_gated_linear_recurrence(q, k, v, g, 4)
    states = torch.zeros(1, 2, 4, 4)
    expected = []
    for i in range(8):
        states = g[..., i, :].exp()[..., None] * states + torch.einsum('bhd,bhe->bhde', k[..., i, :], v[..., i, :])
        expected.append(torch.einsum('bhde,bhd->bhe', states, q[..., i, :]))
    assert torch.allclose(actual, torch.stack(expected, dim=2), atol=2e-5)


def test_gradient_finite():
    model = LosslessSubbandByteLadderLM(tiny())
    x = torch.randint(0, 256, (1, 32))
    model(x, x)["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
