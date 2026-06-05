"""Optional optimizers ported from DEIM for MOTIP experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


def _as_param_list(params: Iterable[Tensor] | Iterable[dict[str, Any]]) -> list[Tensor]:
    param_items = list(params)
    if not param_items:
        return []

    flat_params: list[Tensor] = []
    if isinstance(param_items[0], dict):
        for group in param_items:
            flat_params.extend(list(group["params"]))
    else:
        flat_params.extend(param_items)  # type: ignore[arg-type]

    return [param for param in flat_params if param.requires_grad]


class AutoMuonWithAuxAdam(Optimizer):
    """Muon for matrix-like parameters plus AdamW-style updates for the rest.

    Ported from `/workspace/Project/DEIM_sandbox/DEIM/engine/optim/optim.py`.
    The incoming MOTIP param groups are flattened before Muon/Adam split, matching
    DEIM's implementation. This means MOTIP's per-name LR scales are not preserved
    in the Muon branch; use AdamW/ScheduleFree when those scales are required.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.01,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_lr: float = 0.00025,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-10,
        adam_weight_decay: float | None = None,
        **_kwargs,
    ) -> None:
        trainable_params = _as_param_list(params)
        muon_params = [p for p in trainable_params if p.ndim in (2, 4)]
        adam_params = [p for p in trainable_params if p.ndim not in (2, 4)]

        if adam_weight_decay is None:
            adam_weight_decay = weight_decay

        param_groups: list[dict[str, Any]] = []
        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    use_muon=True,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                )
            )
        if adam_params:
            param_groups.append(
                dict(
                    params=adam_params,
                    use_muon=False,
                    lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    betas=adam_betas,
                    eps=adam_eps,
                )
            )
        if not param_groups:
            raise ValueError("AutoMuonWithAuxAdam got no trainable parameters")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=adam_betas,
            eps=adam_eps,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)

        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        try:
            from muon import muon_update
        except ImportError as exc:
            raise ImportError("AutoMuonWithAuxAdam requires muon-optimizer") from exc

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("AutoMuonWithAuxAdam does not support sparse gradients")

            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)

            update = muon_update(
                grad,
                state["momentum_buffer"],
                beta=group["momentum"],
                ns_steps=group["ns_steps"],
                nesterov=group["nesterov"],
            )
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update.reshape_as(param), alpha=-lr)

    def _step_adam_group(self, group: dict[str, Any]) -> None:
        try:
            from muon import adam_update
        except ImportError as exc:
            raise ImportError("AutoMuonWithAuxAdam requires muon-optimizer") from exc

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        betas = group["betas"]
        eps = group["eps"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("AutoMuonWithAuxAdam does not support sparse gradients")

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            update = adam_update(
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                betas,
                eps,
            )
            if weight_decay:
                param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)


def build_optimizer(params: list[dict], config: dict) -> Optimizer:
    optimizer_type = config.get("OPTIMIZER_TYPE", "AdamW")
    if optimizer_type == "AdamW":
        return torch.optim.AdamW(
            params=params,
            lr=config["LR"],
            weight_decay=config["WEIGHT_DECAY"],
        )

    if optimizer_type == "AdamWScheduleFree":
        try:
            import schedulefree
        except ImportError as exc:
            raise ImportError("AdamWScheduleFree requires schedulefree") from exc
        return schedulefree.AdamWScheduleFree(
            params=params,
            lr=config.get("SCHEDULEFREE_LR", config["LR"]),
            betas=tuple(config.get("SCHEDULEFREE_BETAS", [0.9, 0.999])),
            eps=config.get("SCHEDULEFREE_EPS", 1e-8),
            weight_decay=config.get("SCHEDULEFREE_WEIGHT_DECAY", config["WEIGHT_DECAY"]),
            warmup_steps=config.get("SCHEDULEFREE_WARMUP_STEPS", 0),
            r=config.get("SCHEDULEFREE_R", 0.0),
            weight_lr_power=config.get("SCHEDULEFREE_WEIGHT_LR_POWER", 2.0),
            foreach=config.get("SCHEDULEFREE_FOREACH", True),
        )

    if optimizer_type == "AutoMuonWithAuxAdam":
        return AutoMuonWithAuxAdam(
            params=params,
            lr=config.get("MUON_LR", 0.005),
            weight_decay=config.get("MUON_WEIGHT_DECAY", 0.01),
            momentum=config.get("MUON_MOMENTUM", 0.95),
            nesterov=config.get("MUON_NESTEROV", True),
            ns_steps=config.get("MUON_NS_STEPS", 5),
            adam_lr=config.get("MUON_ADAM_LR", 0.00025),
            adam_betas=tuple(config.get("MUON_ADAM_BETAS", [0.9, 0.95])),
            adam_eps=config.get("MUON_ADAM_EPS", 1e-10),
            adam_weight_decay=config.get("MUON_ADAM_WEIGHT_DECAY", config.get("MUON_WEIGHT_DECAY", 0.01)),
        )

    raise ValueError(f"Unsupported OPTIMIZER_TYPE={optimizer_type!r}")


def set_optimizer_mode(optimizer, mode: str) -> None:
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and inner is not optimizer:
        set_optimizer_mode(inner, mode)
    mode_fn = getattr(optimizer, mode, None)
    if callable(mode_fn):
        mode_fn()
