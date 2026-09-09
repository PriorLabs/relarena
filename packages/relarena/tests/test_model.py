"""Tests for infrastructure stored on the base model contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from relarena.cache import CacheConfig
from relarena.model import RelArenaModel


class _Model(RelArenaModel):
    name = "test"

    def fit(self, *args: object, **kwargs: object) -> None:
        pass

    def predict(self, *args: object, **kwargs: object) -> np.ndarray:
        return np.array([])


def test__model_init__unconfigured_direct_call__uses_private_compute() -> None:
    assert _Model({}).cache == CacheConfig(None, "compute")


def test__model_init__explicit_cache__stores_same_immutable_value(
    tmp_path: Path,
) -> None:
    cache = CacheConfig(tmp_path, "fill")
    assert _Model({}, cache=cache).cache is cache
