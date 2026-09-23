"""Snapshot-to-matrix interface around the existing NoriDFS planner."""

from __future__ import annotations

import pandas as pd

from .._basedfs import CUTOFF_COLUMN, BaseDFS, FeatureMatrices, RelationalData
from .contracts import AnchorDefinition, DatabaseView, NativeFeatureConfig
from .engine import NativeRelationalFeaturizer


class NoriDFS(BaseDFS):
    """Use the native planner through the common DFS matrix interface.

    Callers that need the fitted plan itself can use
    ``NativeRelationalFeaturizer`` directly.
    """

    def __init__(
        self,
        config: NativeFeatureConfig | None = None,
        *,
        horizon: str | pd.Timedelta | None = None,
    ) -> None:
        """Set the native feature budget and optional aggregation-window unit."""
        self.config = config or NativeFeatureConfig()
        self.horizon = horizon

    def compute(
        self,
        snapshot: RelationalData,
        train_rows: pd.DataFrame,
        query_rows: pd.DataFrame,
        entity_table: str,
        *,
        row_id_column: str,
    ) -> FeatureMatrices:
        """Plan once and return features in the supplied context/query order."""
        database = DatabaseView.from_parts(
            tables=snapshot.tables,
            primary_keys=snapshot.primary_keys,
            foreign_keys=snapshot.foreign_keys,
            time_columns=snapshot.time_columns,
        )
        if entity_table not in database.primary_keys:
            raise ValueError("NoriDFS requires a single-column entity primary key")
        entity_column = database.primary_keys[entity_table]
        combined = pd.concat([train_rows, query_rows], ignore_index=True)
        if row_id_column not in combined or combined[row_id_column].isna().any():
            raise ValueError("feature rows require non-null row identifiers")
        if combined[row_id_column].duplicated().any():
            raise ValueError("feature rows require unique row identifiers")
        definition = AnchorDefinition.create(
            entity_table=entity_table,
            entity_column=entity_column,
            cutoff_column=CUTOFF_COLUMN,
            horizon=self.horizon,
        )
        featurizer = NativeRelationalFeaturizer.fit(database, definition, self.config)
        anchors = combined[[entity_column, CUTOFF_COLUMN]]
        frame = featurizer.transform(database, anchors).frame
        split = len(train_rows)
        return FeatureMatrices(
            train=frame.iloc[:split].reset_index(drop=True),
            query=frame.iloc[split:].reset_index(drop=True),
            feature_names=list(frame.columns),
        )
