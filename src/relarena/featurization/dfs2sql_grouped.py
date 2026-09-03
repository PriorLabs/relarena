"""A dfs2sql engine that builds each set of cutoff-time temp tables once.

fastdfs' stock `dfs2sql` engine names its intermediate cutoff-time tables per
(table, depth), not per feature, so every feature on the same join path emits the
same `CREATE TABLE AS` statements and drops them again right after its `SELECT`.
On a wide task that is thousands of statements where a few hundred suffice.

The engine here runs the stock SQL generator unchanged and regroups its output:
each distinct set of temp tables is created once, every feature over it is
selected, and only then is it dropped. Feature values and column order match the
stock engine; only the statement order and count differ.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import featuretools as ft
import pandas as pd
from fastdfs import RDB
from fastdfs.dfs import DFSConfig, assemble_dfs2sql_feature_frames, dfs_engine
from fastdfs.dfs.dfs2sql_engine import DFS2SQLEngine
from fastdfs.dfs.duckdb_database import DuckDBBuilder
from fastdfs.dfs.gen_sqls import decode_column_from_sql, features2sql
from sqlglot.expressions import Create, Drop, Expression, Select

#: Engine name to pass as `DFSConfig(engine=...)`.
GROUPED_DFS2SQL_ENGINE = "dfs2sql-grouped"


def group_temp_table_statements(statements: list[Expression]) -> list[Expression]:
    """Reorder stock dfs2sql statements so each set of temp tables is built once.

    The stock generator emits `[CREATE..., SELECT, DROP...]` per feature. Features
    whose CREATE statements render to the same SQL are moved next to each other and
    share one CREATE/DROP pair. Selects without a CREATE (the no-cutoff path) pass
    through in their original order. The `Select` objects are returned as is, so
    callers can map results back by identity.
    """
    groups: dict[tuple[str, ...], dict[str, list[Expression]]] = {}
    pending_creates: list[Expression] = []
    current: dict[str, list[Expression]] | None = None
    current_is_new = False
    for statement in statements:
        if isinstance(statement, Create):
            pending_creates.append(statement)
        elif isinstance(statement, Select):
            key = tuple(create.sql() for create in pending_creates)
            current_is_new = key not in groups
            current = groups.setdefault(
                key, {"creates": pending_creates, "selects": [], "drops": []}
            )
            current["selects"].append(statement)
            pending_creates = []
        elif isinstance(statement, Drop):
            if current is None:
                raise ValueError("DROP before any SELECT in dfs2sql statements")
            if current_is_new:
                current["drops"].append(statement)
        else:
            raise TypeError(f"unexpected dfs2sql statement: {type(statement).__name__}")
    if pending_creates:
        raise ValueError("trailing CREATE without a SELECT in dfs2sql statements")

    ordered: list[Expression] = []
    for group in groups.values():
        ordered.extend(group["creates"])
        ordered.extend(group["selects"])
        ordered.extend(group["drops"])
    return ordered


@dfs_engine
class GroupedDFS2SQLEngine(DFS2SQLEngine):
    """The stock dfs2sql engine with temp-table statements grouped per join path."""

    name = GROUPED_DFS2SQL_ENGINE

    def compute_feature_matrix(
        self,
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: dict[str, str],
        cutoff_time_column: str | None,
        features: list[ft.FeatureBase],
        config: DFSConfig,
    ) -> pd.DataFrame:
        """Run the grouped statements and assemble the frames in feature order."""
        target_index = "__target_index__"
        engine_path = config.engine_path
        if engine_path is None:
            engine_path = str(
                Path(tempfile.gettempdir()) / f"fastdfs_{uuid.uuid4()}.db"
            )
        builder = DuckDBBuilder(Path(engine_path))
        self._build_database_tables(
            builder, rdb, target_dataframe, target_index, cutoff_time_column
        )

        has_cutoff_time = config.use_cutoff_time and cutoff_time_column is not None
        cutoff_time_col_name = builder.cutoff_time_col_name if has_cutoff_time else None
        statements = features2sql(
            features,
            target_index,
            has_cutoff_time=has_cutoff_time,
            cutoff_time_table_name=(
                builder.cutoff_time_table_name if has_cutoff_time else None
            ),
            cutoff_time_col_name=cutoff_time_col_name,
            time_col_mapping=builder.time_columns if has_cutoff_time else None,
            column_type_map=self._build_column_type_map(rdb, target_dataframe),
            include_cutoff_time=config.include_cutoff_time,
        )

        # Grouping reorders the selects; restore the generator's feature order so
        # the matrix columns come out as the stock engine emits them.
        position = {id(s): i for i, s in enumerate(statements) if isinstance(s, Select)}
        frames: list[tuple[int, pd.DataFrame]] = []
        for statement in group_temp_table_statements(statements):
            result = builder.db.sql(statement.sql())
            if result is None:  # CREATE and DROP return nothing
                continue
            frame = result.df()
            if cutoff_time_col_name in frame.columns:
                frame = frame.drop(columns=[cutoff_time_col_name])
            frame = frame.rename(columns=decode_column_from_sql)
            frames.append((position[id(statement)], frame))
        frames.sort(key=lambda item: item[0])

        if not frames:
            return pd.DataFrame({target_index: target_dataframe[target_index]})
        canonical_index = pd.Index(
            target_dataframe[target_index].values, name=target_index
        )
        merged = assemble_dfs2sql_feature_frames(
            [frame for _, frame in frames],
            target_index,
            canonical_index,
            concat_chunk_size=config.dfs2sql_concat_chunk_size,
        )
        passthrough = set(target_dataframe.columns) - {target_index}
        return merged[[c for c in merged.columns if c not in passthrough]]
