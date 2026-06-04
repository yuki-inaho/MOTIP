"""Tests for config-driven early stopping (porting item 4).

The decision is a pure function over the history of monitored values (epoch
average loss; lower is better). It must:
  - never stop before ``start_epoch`` worth of history exists;
  - reset its patience counter when the metric improves by more than
    ``min_delta``;
  - stop once ``patience`` consecutive non-improving epochs have elapsed.

Disabling early stop (EARLY_STOP=False) is handled by the caller (the judge is
simply never consulted), so the default behaviour stays unchanged.
"""

from train import should_early_stop


def test_no_stop_with_short_history() -> None:
    # patience=2 needs at least 3 epochs (1 baseline + 2 non-improving) to fire.
    assert should_early_stop([1.0], patience=2, min_delta=0.0, start_epoch=0) is False
    assert should_early_stop([1.0, 0.9], patience=2, min_delta=0.0, start_epoch=0) is False


def test_stop_after_patience_non_improving() -> None:
    # best=1.0 at epoch0; epochs 1,2 do not improve -> stop at epoch2 (patience=2).
    assert should_early_stop([1.0, 1.0, 1.0], patience=2, min_delta=0.0, start_epoch=0) is True


def test_improvement_resets_patience() -> None:
    # 1.0, 1.0(no), 0.5(improve -> reset), 0.5(no) -> only 1 non-improving since reset.
    assert should_early_stop([1.0, 1.0, 0.5, 0.5], patience=2, min_delta=0.0, start_epoch=0) is False
    # one more non-improving epoch trips patience=2.
    assert should_early_stop([1.0, 1.0, 0.5, 0.5, 0.5], patience=2, min_delta=0.0, start_epoch=0) is True


def test_min_delta_requires_meaningful_improvement() -> None:
    # Tiny improvements below min_delta do not count as improvement.
    history = [1.0, 0.999, 0.998]   # each step improves by 0.001 < min_delta=0.01
    assert should_early_stop(history, patience=2, min_delta=0.01, start_epoch=0) is True
    # A real improvement (>= min_delta) resets.
    history2 = [1.0, 0.5, 0.499]    # 0.5 is a big improvement, then tiny
    assert should_early_stop(history2, patience=2, min_delta=0.01, start_epoch=0) is False


def test_start_epoch_delays_monitoring() -> None:
    # With start_epoch=2, epochs before index 2 are ignored for patience.
    # history index: 0,1 ignored; monitoring effectively starts at epoch 2.
    assert should_early_stop([5.0, 4.0, 1.0, 1.0], patience=2, min_delta=0.0, start_epoch=2) is False
    assert should_early_stop([5.0, 4.0, 1.0, 1.0, 1.0], patience=2, min_delta=0.0, start_epoch=2) is True


def test_patience_zero_stops_on_first_non_improvement() -> None:
    assert should_early_stop([1.0], patience=0, min_delta=0.0, start_epoch=0) is False
    assert should_early_stop([1.0, 1.0], patience=0, min_delta=0.0, start_epoch=0) is True
