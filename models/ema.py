# Copyright (c) Ruopeng Gao. All Rights Reserved.
"""Exponential Moving Average of model weights (Accelerate-friendly).

Adapted for MOTIP's Accelerate-based training from the DEIM reference
``/workspace/Project/DEIM_sandbox/DEIM/engine/optim/ema.py:33-78``.

Key differences from the reference:
  - This class does NOT call any distributed de-parallel helper internally.
    The caller is responsible for passing an *unwrapped* module to ``update``
    (i.e. ``ema.update(accelerator.unwrap_model(model))``), which is the
    Accelerate-correct way to reach the real weights under DDP / AMP wrappers.
  - The constructor takes an already-unwrapped module to mirror.

A smoothed copy of every floating-point entry in ``state_dict`` (parameters
*and* buffers) is kept; the decay follows an exponential ramp
``decay * (1 - exp(-updates / warmups))`` so early updates move faster.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA"]


class ModelEMA(object):
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 1000, start: int = 0):
        """Create an EMA mirror of ``model``.

        Args:
            model: The (already unwrapped) module to mirror. Deep-copied and put
                in eval mode; its parameters do not require grad.
            decay: Asymptotic decay applied once warmups are exhausted.
            warmups: Exponential-ramp horizon. ``0`` means constant ``decay``.
            start: Number of initial ``update`` calls to skip before averaging.
        """
        super().__init__()
        self.module = deepcopy(model).eval()
        self.decay = decay
        self.warmups = warmups
        self.start = start
        self.before_start = 0
        self.updates = 0  # number of EMA updates applied
        if warmups == 0:
            self.decay_fn = lambda x: decay
        else:
            self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))

        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Move the EMA weights toward ``model`` (which must be unwrapped).

        The first ``start`` calls are skipped (no averaging, no count bump).
        Only floating-point state is averaged; integer buffers (e.g.
        BatchNorm ``num_batches_tracked``) are left untouched.
        """
        if self.before_start < self.start:
            self.before_start += 1
            return
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach().to(v.device)

    def to(self, *args, **kwargs) -> "ModelEMA":
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.module.load_state_dict(state["module"], strict=strict)
        if "updates" in state:
            self.updates = state["updates"]

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"
