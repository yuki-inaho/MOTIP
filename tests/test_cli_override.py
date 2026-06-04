"""Tests for the generic ``-u KEY=VALUE`` config override (porting item 5).

Contract:
  - Values are interpreted as YAML (``yaml.safe_load``), so int / float / bool /
    list / str all round-trip with the correct type.
  - A matching (possibly nested) key is updated in place.
  - An unknown key is NOT silently ignored: it raises / warns explicitly
    (no silent fallback).
"""

import pytest

from configs.util import apply_cli_updates, update_config_with_kv


def test_dotted_update_sets_top_level_int() -> None:
    cfg = {"EPOCHS": 10, "LR": 0.0001}
    out = apply_cli_updates(cfg, ["EPOCHS=3"])
    assert out["EPOCHS"] == 3
    assert isinstance(out["EPOCHS"], int)


def test_value_is_yaml_typed() -> None:
    cfg = {"LR": 0.0001, "EPOCHS": 10, "AMP_DTYPE": "no", "EARLY_STOP": False}
    # NOTE: YAML floats need a decimal point or a signed exponent. "1e-3" is a
    # *string* under PyYAML (YAML 1.1); use "0.001" or "1.0e-3" for a float.
    out = apply_cli_updates(cfg, ["LR=0.001", "EARLY_STOP=True", "AMP_DTYPE=fp16"])
    assert out["LR"] == 0.001 and isinstance(out["LR"], float)
    assert out["EARLY_STOP"] is True
    assert out["AMP_DTYPE"] == "fp16"


def test_list_value_is_parsed() -> None:
    cfg = {"SAMPLE_LENGTHS": [2]}
    out = apply_cli_updates(cfg, ["SAMPLE_LENGTHS=[10, 20]"])
    assert out["SAMPLE_LENGTHS"] == [10, 20]


def test_nested_key_is_updated() -> None:
    cfg = {"MODEL": {"BACKBONE": {"DEPTH": 18}}}
    out = apply_cli_updates(cfg, ["DEPTH=50"])
    assert out["MODEL"]["BACKBONE"]["DEPTH"] == 50


def test_unknown_key_is_not_silently_ignored() -> None:
    cfg = {"EPOCHS": 10}
    with pytest.raises(KeyError) as excinfo:
        apply_cli_updates(cfg, ["DOES_NOT_EXIST=1"])
    assert "DOES_NOT_EXIST" in str(excinfo.value)


def test_malformed_update_without_equals_raises() -> None:
    cfg = {"EPOCHS": 10}
    with pytest.raises(ValueError):
        apply_cli_updates(cfg, ["EPOCHS"])  # missing '='


def test_multiple_updates_applied_in_order() -> None:
    cfg = {"EPOCHS": 10, "BATCH_SIZE": 1}
    out = apply_cli_updates(cfg, ["EPOCHS=5", "BATCH_SIZE=2"])
    assert out["EPOCHS"] == 5 and out["BATCH_SIZE"] == 2


def test_existing_update_config_with_kv_still_works() -> None:
    # The legacy helper is unchanged and still resolves nested keys.
    cfg = {"A": {"B": 1}}
    hit, out = update_config_with_kv(cfg, k="B", v="True")
    assert hit is True
    assert out["A"]["B"] is True
