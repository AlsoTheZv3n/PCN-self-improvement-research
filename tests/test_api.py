"""Tests for the train_and_eval config contract (docs/02, docs/13 M1/M2).

These exercise only `_resolve_config`, so they need no MNIST download or torchvision
(the data loaders are imported lazily inside `_mnist_loaders`).
"""
from __future__ import annotations

import pytest

from pcn.api import DEFAULT_CONFIG, _resolve_config


def test_config_aliasing_maps_spec_names():
    cfg = _resolve_config({"eta_x": 0.2, "eta_w": 0.03})
    assert cfg["lr_state"] == 0.2
    assert cfg["lr_weight"] == 0.03


def test_unknown_config_key_warns_and_is_ignored():
    with pytest.warns(UserWarning):
        cfg = _resolve_config({"nonsense_key": 123})
    assert "nonsense_key" not in cfg


def test_defaults_preserved_when_no_config():
    cfg = _resolve_config(None)
    assert cfg["lr_weight"] == DEFAULT_CONFIG["lr_weight"]
    assert cfg["precision_schedule"] == "isotropic"
    assert cfg["update_variant"] == "standard"


def test_reserved_precision_schedule_raises():
    # Non-isotropic precision is reserved for M2 and must fail loudly, not silently no-op.
    with pytest.raises(NotImplementedError):
        _resolve_config({"precision_schedule": "spiking"})


def test_update_variant_ipc_accepted_unknown_raises():
    # iPC is now implemented (Salvatori et al. 2024) -> accepted; an unknown variant still
    # fails loudly rather than silently no-op'ing.
    assert _resolve_config({"update_variant": "ipc"})["update_variant"] == "ipc"
    with pytest.raises(NotImplementedError):
        _resolve_config({"update_variant": "bogus"})
