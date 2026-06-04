"""Tests for the Accelerate-friendly ModelEMA (porting item 6).

ModelEMA keeps an exponential moving average of all floating-point state
(parameters + buffers). The decay follows an exponential ramp
``decay * (1 - exp(-updates / warmups))`` so early updates move faster, and a
``start`` offset can skip the very first updates. The averaged module is the
weight set used for evaluation / checkpointing when EMA is enabled.

Reference: /workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78
"""

import math

import torch
import torch.nn as nn

from models.ema import ModelEMA


def _tiny_model(value: float) -> nn.Module:
    m = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


def test_decay_fn_ramps_with_warmups() -> None:
    ema = ModelEMA(_tiny_model(0.0), decay=0.9999, warmups=1000)
    # Early on, the effective decay is much smaller than the asymptotic decay.
    assert ema.decay_fn(1) == 0.9999 * (1 - math.exp(-1 / 1000))
    assert ema.decay_fn(1) < ema.decay_fn(1000) < 0.9999
    # As updates -> inf, decay_fn -> decay.
    assert ema.decay_fn(10_000_000) == 0.9999 * (1 - math.exp(-10_000_000 / 1000))
    assert abs(ema.decay_fn(10_000_000) - 0.9999) < 1e-9


def test_decay_fn_constant_when_no_warmups() -> None:
    ema = ModelEMA(_tiny_model(0.0), decay=0.99, warmups=0)
    assert ema.decay_fn(1) == 0.99
    assert ema.decay_fn(123) == 0.99


def test_update_moves_ema_weights_toward_model_and_counts() -> None:
    ema_model = _tiny_model(0.0)      # EMA starts at all-zeros
    ema = ModelEMA(ema_model, decay=0.5, warmups=0)   # constant decay 0.5
    assert ema.updates == 0

    online = _tiny_model(1.0)         # online model is all-ones
    ema.update(online)

    assert ema.updates == 1
    # new_ema = 0.5 * 0 + (1 - 0.5) * 1 = 0.5
    w = ema.module.weight
    assert torch.allclose(w, torch.full_like(w, 0.5))

    # A second update moves it further: 0.5*0.5 + 0.5*1 = 0.75
    ema.update(online)
    assert ema.updates == 2
    assert torch.allclose(w, torch.full_like(w, 0.75))


def test_start_offset_skips_initial_updates() -> None:
    ema = ModelEMA(_tiny_model(0.0), decay=0.5, warmups=0, start=2)
    online = _tiny_model(1.0)
    # First two updates are skipped (no averaging, no count increment).
    ema.update(online)
    ema.update(online)
    assert ema.updates == 0
    assert torch.allclose(ema.module.weight, torch.zeros_like(ema.module.weight))
    # Third update applies.
    ema.update(online)
    assert ema.updates == 1
    assert torch.allclose(ema.module.weight, torch.full_like(ema.module.weight, 0.5))


def test_ema_module_params_do_not_require_grad() -> None:
    ema = ModelEMA(_tiny_model(0.3), decay=0.9, warmups=10)
    assert all(not p.requires_grad for p in ema.module.parameters())


def test_state_dict_roundtrip() -> None:
    ema = ModelEMA(_tiny_model(0.0), decay=0.5, warmups=0)
    ema.update(_tiny_model(1.0))
    state = ema.state_dict()
    assert set(state.keys()) == {"module", "updates"}
    assert state["updates"] == 1

    restored = ModelEMA(_tiny_model(9.0), decay=0.5, warmups=0)
    restored.load_state_dict(state)
    assert restored.updates == 1
    assert torch.allclose(restored.module.weight, ema.module.weight)


def test_buffers_are_averaged_too() -> None:
    # BatchNorm has float running buffers that must be averaged.
    class Net(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.bn = nn.BatchNorm1d(2)
            with torch.no_grad():
                self.bn.running_mean.fill_(v)

    ema = ModelEMA(Net(0.0), decay=0.5, warmups=0)
    ema.update(Net(1.0))
    assert torch.allclose(ema.module.bn.running_mean, torch.full((2,), 0.5))
