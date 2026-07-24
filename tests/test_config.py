"""Unit tests for shrike.config: the model registry and the SHRIKE_MODEL_DIR
override (pure Python, no model/GPU)."""

import importlib

import shrike.config as config
from tests import conftest


def test_registry_maps_known_models_to_qwen2():
    # every registered snapshot builds the qwen2 architecture today
    assert set(config.MODEL_REGISTRY.values()) == {"qwen2"}
    assert "Qwen/Qwen2.5-0.5B-Instruct" in config.MODEL_REGISTRY


def test_default_model_dir_is_a_nonempty_string():
    assert isinstance(config.DEFAULT_MODEL_DIR, str)
    assert config.DEFAULT_MODEL_DIR


def test_env_override_takes_effect_on_import(monkeypatch):
    monkeypatch.setenv("SHRIKE_MODEL_DIR", "/some/custom/snapshot")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DEFAULT_MODEL_DIR == "/some/custom/snapshot"
    finally:
        # restore the module to its unpatched state for other tests
        monkeypatch.delenv("SHRIKE_MODEL_DIR", raising=False)
        importlib.reload(config)


def test_model_available_helper_reports_missing_snapshot(tmp_path):
    # model_available() checks for a config.json inside conftest.MODEL_DIR
    original = conftest.MODEL_DIR
    try:
        conftest.MODEL_DIR = str(tmp_path)
        assert conftest.model_available() is False  # empty dir, no config.json
        (tmp_path / "config.json").write_text("{}")
        assert conftest.model_available() is True
    finally:
        conftest.MODEL_DIR = original
