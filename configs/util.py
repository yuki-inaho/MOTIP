# Copyright (c) Ruopeng Gao. All Rights Reserved.

import argparse
import warnings

import yaml

from utils.misc import yaml_to_dict


def update_config_with_kv(config: dict, k: str, v) -> [bool, dict]:
    """
    Update config with a pair of K and V from options.

    Args:
        config: Current config.
        k: A key from options.
        v: A value from options.

    Returns:
        [New config dict, Hit or Not]
    """
    hit = False
    for config_k in config.keys():
        if isinstance(config[config_k], dict):
            hit, config[config_k] = update_config_with_kv(config=config[config_k], k=k, v=v)
            if hit:
                break
        elif config_k == k.upper():
            if v == "True":
                config[config_k] = True
            elif v == "False":
                config[config_k] = False
            else:
                config[config_k] = v
            hit = True
            break
    return hit, config


def _set_existing_key(config: dict, key: str, value) -> bool:
    """Set ``key`` (matched case-insensitively, possibly nested) to ``value``.

    Returns True if a matching key existed and was updated, False otherwise.
    Does NOT create new keys (no silent fallback on typos).
    """
    target = key.upper()
    for config_k in config.keys():
        if config_k == target:
            config[config_k] = value
            return True
    for config_k in config.keys():
        if isinstance(config[config_k], dict):
            if _set_existing_key(config[config_k], key, value):
                return True
    return False


def apply_cli_updates(config: dict, updates: list[str] | None) -> dict:
    """Apply generic ``KEY=VALUE`` overrides to ``config`` (in place).

    Each update is ``KEY=VALUE`` where VALUE is parsed as YAML
    (``yaml.safe_load``), so ints/floats/bools/lists/strings keep their types.
    The KEY is matched case-insensitively against existing (possibly nested)
    config keys. Unknown keys are NOT silently ignored: a KeyError is raised
    (no silent fallback on typos / stale keys).

    Args:
        config: The config dict to mutate.
        updates: List like ``["EPOCHS=3", "AMP_DTYPE=fp16"]`` (or None).

    Returns:
        The same config dict, updated.
    """
    if not updates:
        return config
    for item in updates:
        if "=" not in item:
            raise ValueError(f"Invalid -u override '{item}'; expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = yaml.safe_load(raw_value)   # YAML typing: int/float/bool/list/str
        if not _set_existing_key(config, key, value):
            raise KeyError(
                f"Unknown -u override key '{key}'; it does not exist in the config "
                f"(no silent fallback). Check the spelling against the .yaml config."
            )
        warnings.warn(f"Config overridden via -u: {key.upper()} = {value!r}", stacklevel=2)
    return config


def update_config(config: dict, option: argparse.Namespace) -> dict:
    """
    Update current config with an option parser.

    Args:
        config: Current config.
        option: Option parser.

    Returns:
        New config dict.
    """
    # v2.0 DO NOT need to check uniqueness
    # if is_unique(config)[0] is False:
    #     raise RuntimeError("Config's key is not unique, Please check the config file.")

    # "update" is the generic -u override; it is applied separately via
    # apply_cli_updates (not a typed config key), so skip it here.
    for option_k, option_v in vars(option).items():
        if option_k not in ("config_path", "update") and option_v is not None:  # except --config-path / -u
            # v2.0 remove hierarchical config setting, using plain config setting.
            # hit, config = update_config_with_kv(config=config, k=option_k, v=option_v)
            config_k = option_k.upper()
            if config_k in config:
                if option_v == "True":
                    config[config_k] = True
                elif option_v == "False":
                    config[config_k] = False
                else:
                    config[config_k] = option_v
            else:
                raise RuntimeError(f"The option '{option_k}' is not appeared in .yaml config file.")
    return config


def is_unique(config: dict, keys_set: set = None) -> [bool, set]:
    """
    Check whether the keys in config are unique.

    Args:
        config: Config dict.
        keys_set: Current keys set.

    Returns:
        [Whether the keys are unique, Current keys set]
    """
    if keys_set is None:
        keys_set = set()

    for k in config.keys():
        if k in keys_set:
            return False, keys_set
        else:
            keys_set.add(k)
        if isinstance(config[k], dict):
            hit, keys_set = is_unique(config[k], keys_set=keys_set)
            if hit is False:
                return False, keys_set

    return True, keys_set


def load_super_config(config: dict, super_config_path: str | None):
    if super_config_path is None:
        return config
    else:
        super_config = yaml_to_dict(super_config_path)
        super_config = load_super_config(super_config, super_config["SUPER_CONFIG_PATH"])
        super_config.update(config)
        return super_config
