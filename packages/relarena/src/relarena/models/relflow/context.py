"""Derive and materialize nested Arrow observations from harness-supplied data.

Profiling always uses the validation-censored database. The frozen experiment's
outer refit keeps validation anchors' histories frozen at that same boundary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from relbench.base import Database, EntityTask, Table, TaskType

from ._vendor import derive, raw


@dataclass
class Context:
    """Owned DuckDB connection with stable ordinals and both cutoff views."""

    connection: duckdb.DuckDBPyConnection
    schema: dict[str, dict[str, Any]]
    sources: dict[str, str]
    cutoff: Any
    registered: dict[str, str]
    decimals: dict[str, frozenset[str]]

    @classmethod
    def open(
        cls,
        db: Database,
        cutoff: Any,
        directory: Path,
        *,
        threads: int = 8,
        memory_limit: str = "16GB",
        max_temp_directory_size: str = "64GB",
    ) -> Context:
        """Register only provided frames; never consult RelBench's raw cache."""
        if type(threads) is not int or threads < 1:
            raise ValueError("relflow DuckDB threads must be a positive integer.")
        connection = duckdb.connect(
            config={
                "threads": threads,
                "memory_limit": memory_limit,
                "max_temp_directory_size": max_temp_directory_size,
                "temp_directory": str(directory / "spill"),
                "preserve_insertion_order": True,
            }
        )
        schema, sources, registered, decimals = {}, {}, {}, {}
        for index, (name, table) in enumerate(sorted(db.table_dict.items())):
            if "__source_row_id" in table.df.columns:
                connection.close()
                raise ValueError("Input column '__source_row_id' is reserved.")
            relation = f"__relflow_source_{index}"
            # RelBench pandas categoricals otherwise become DuckDB ENUMs, while
            # the frozen Parquet path sees VARCHAR. Preserve values and nulls.
            categorical = [
                name
                for name, dtype in table.df.dtypes.items()
                if isinstance(dtype, pd.CategoricalDtype)
            ]
            frame = table.df.copy(deep=False)
            for column in categorical:
                frame[column] = frame[column].astype(object)
            frame["__source_row_id"] = np.arange(len(frame), dtype=np.int64)
            connection.register(relation, frame)
            registered[name] = relation
            decimals[name] = frozenset(
                column
                for column, dtype, *_ in connection.execute(
                    f"DESCRIBE {relation}"
                ).fetchall()
                if dtype.startswith("DECIMAL(")
            )
            q = derive.quote
            connection.execute(f"CREATE VIEW {q(name)} AS SELECT * FROM {relation}")
            schema[name] = {
                "pkey": table.pkey_col,
                "time_col": table.time_col,
                # The harness cleaner can remove all-null columns while retaining
                # their original foreign-key metadata. Derive only existing links.
                "fkeys": {
                    column: target
                    for column, target in table.fkey_col_to_pkey_table.items()
                    if column in frame.columns
                },
            }
            if table.time_col:
                value = str(cutoff).replace("'", "''")
                sources[name] = (
                    f"(SELECT * FROM {q(name)} WHERE {q(table.time_col)} "
                    f"<= TIMESTAMP '{value}')"
                )
            else:
                sources[name] = q(name)
        return cls(connection, schema, sources, cutoff, registered, decimals)

    def freeze(self) -> None:
        """Replace timed views with their validation-censored equivalents."""
        value = str(self.cutoff).replace("'", "''")
        for name, spec in self.schema.items():
            if spec["time_col"]:
                self.connection.execute(
                    f"CREATE OR REPLACE VIEW {derive.quote(name)} AS "
                    f"SELECT * FROM {self.registered[name]} "
                    f"WHERE {derive.quote(spec['time_col'])} <= TIMESTAMP '{value}'"
                )

    def close(self) -> None:
        """Release registered frame references and temporary DuckDB storage."""
        self.connection.close()


def infer(
    context: Context,
    task: EntityTask,
    config: dict[str, Any],
    *,
    namespace: str = "database",
) -> raw.Task:
    """Freeze structure from keys and validation-time column measurements."""
    task_type = (
        "binary_classification"
        if task.task_type == TaskType.BINARY_CLASSIFICATION
        else "regression"
    )
    # Namespace hashes by related table, never by task identity.
    meta = derive.Meta(
        dataset=namespace,
        task="task",
        task_type=task_type,
        entity_table=task.entity_table,
        entity_col=task.entity_col,
        time_col=task.time_col,
        target_col=task.target_col,
        training_rows=config["training_rows"],
    )
    tables = derive.profile(context.connection, context.sources, context.schema)
    return derive.derive(context.connection, tables, meta)


def materialize(
    context: Context,
    structure: raw.Task,
    task: EntityTask,
    table: Table,
    output: Path,
    *,
    labelled: bool,
    deadline: float | None = None,
) -> Path:
    """Write bounded chunks in the supplied task-table order, preserving nulls."""
    frame = table.df
    required = [task.entity_col, task.time_col]
    if labelled:
        required.append(task.target_col)
    if any(name not in frame for name in required):
        raise ValueError(f"Task table requires columns {required!r}.")
    if frame[required].isna().any().any():
        raise ValueError("Anchor entity, timestamp and training labels cannot be null.")
    anchors = pa.table(
        {
            "__row_position": np.arange(len(frame), dtype=np.int64),
            "__entity_key": pa.Array.from_pandas(frame[task.entity_col]),
            "__as_of": pa.Array.from_pandas(frame[task.time_col]),
        }
    )
    context.connection.register("__relflow_anchors", anchors)
    query = raw.query(structure, history_overscan=4, decimals=context.decimals).replace(
        "read_parquet($anchors)", "__relflow_anchors"
    )
    targets = pa.Array.from_pandas(frame[task.target_col]) if labelled else None
    if targets is not None:
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            invalid = pc.and_(pc.not_equal(targets, 0), pc.not_equal(targets, 1))
            if pc.any(invalid).as_py():
                raise ValueError("Binary training labels must be 0 or 1.")
            targets = pc.cast(targets, pa.bool_())
        else:
            targets = pc.cast(targets, pa.float64())
    writer = None
    try:
        for start in range(0, max(len(frame), 1), 8192):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("relflow exhausted its fit budget in preparation.")
            result = context.connection.execute(
                query, {"start": start, "stop": min(start + 8192, len(frame))}
            ).fetch_arrow_table()
            if targets is not None:
                result = result.append_column(
                    "target", pc.take(targets, result["__row_position"])
                )
            if writer is None:
                writer = pq.ParquetWriter(output, result.schema)
            writer.write_table(result)
    finally:
        if writer is not None:
            writer.close()
        context.connection.unregister("__relflow_anchors")
    return output
