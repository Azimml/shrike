"""Unit tests for the top-level package surface (no model/GPU)."""

import shrike


def test_version_is_a_string():
    assert isinstance(shrike.__version__, str)
    assert shrike.__version__  # non-empty


def test_sampling_params_reexported():
    # importing the package must not require torch, and SamplingParams must be
    # the same class as the one in the engine module
    from shrike.engine.request import SamplingParams

    assert shrike.SamplingParams is SamplingParams


def test_public_all():
    assert set(shrike.__all__) == {"SamplingParams", "__version__"}
