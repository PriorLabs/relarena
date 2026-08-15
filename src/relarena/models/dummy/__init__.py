"""Constant (optimal-constant) baselines.

Importing this package registers `constant-global` (one global constant) and
`constant-per-entity` (each entity`s own constant). See `model`.
"""

from relarena.models.dummy.model import DummyBaseline, DummyPerEntityBaseline

__all__ = ["DummyBaseline", "DummyPerEntityBaseline"]
