"""Shared input and output contract for relational feature backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

CUTOFF_COLUMN = "__nori_rel_cutoff_time"


class RelationalData(Protocol):
    """The snapshot fields needed by feature generation, without a connector."""

    @property
    def tables(self) -> Mapping[str, pd.DataFrame]:
        """Return the bounded source tables."""
        ...

    @property
    def primary_keys(self) -> Mapping[str, Sequence[str]]:
        """Return each table's primary-key columns."""
        ...

    @property
    def foreign_keys(self) -> Sequence[tuple[str, str, str, str]]:
        """Return child table/key and parent table/key relationships."""
        ...

    @property
    def time_columns(self) -> Mapping[str, str]:
        """Return the availability timestamp for each temporal table."""
        ...


@dataclass(frozen=True)
class FeatureMatrices:
    """Aligned context/query matrices with the same ordered feature columns."""

    train: pd.DataFrame
    query: pd.DataFrame
    feature_names: list[str]


class FeatureBackend(ABC):
    """Materialize relational features without changing prediction row order.

    Deep feature synthesis backends and agent-written SQL programs both
    implement this; the runtime only depends on ``compute``.
    """

    @abstractmethod
    def compute(
        self,
        snapshot: RelationalData,
        train_rows: pd.DataFrame,
        query_rows: pd.DataFrame,
        entity_table: str,
        *,
        row_id_column: str,
    ) -> FeatureMatrices:
        """Build features for entity keys, cutoffs and unique row IDs.

        Callers supply label-free rows with the entity key, ``CUTOFF_COLUMN``
        and a unique ``row_id_column``. Each backend retains its own search
        policy; this interface aligns inputs and outputs, not feature values.
        """
        ...


__all__ = ["CUTOFF_COLUMN", "FeatureBackend", "FeatureMatrices", "RelationalData"]
