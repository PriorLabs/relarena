"""Materialize supplied temporal splits as ordered, nested Arrow observations.

``snapshot`` projects one supplied, censored database per phase. ``prepare``
uses that snapshot and the supplied task table; this module alone attaches labels.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa
    from relbench.base import Database, EntityTask, Table


def decimal_type(datatype: pa.DataType) -> pa.DataType:
    """Replace decimal leaves while preserving nested Arrow container structure."""
    import pyarrow as pa

    if pa.types.is_decimal(datatype):
        return pa.float64()
    if pa.types.is_struct(datatype):
        return pa.struct(
            [field.with_type(decimal_type(field.type)) for field in datatype]
        )
    if pa.types.is_list(datatype):
        return pa.list_(
            datatype.value_field.with_type(decimal_type(datatype.value_type))
        )
    if pa.types.is_large_list(datatype):
        return pa.large_list(
            datatype.value_field.with_type(decimal_type(datatype.value_type))
        )
    if pa.types.is_fixed_size_list(datatype):
        return pa.list_(
            datatype.value_field.with_type(decimal_type(datatype.value_type)),
            datatype.list_size,
        )
    return datatype


def configuration(directory: Path) -> dict[str, Any]:
    """Leave room for resident input frames and spill within the phase directory."""
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    available = total
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    limit = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    if limit.is_file() and current.is_file():
        bound = limit.read_text().strip()
        if bound.isdigit():
            available = min(available, max(1, int(bound) - int(current.read_text())))
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    spill = directory / "spill"
    spill.mkdir(exist_ok=True)
    spill_bytes = max(1, int(shutil.disk_usage(spill).free * 0.8))
    return {
        "threads": cpus,
        "memory_limit": f"{max(1, int(available * 0.5))}B",
        "temp_directory": str(spill),
        "max_temp_directory_size": f"{spill_bytes}B",
        "preserve_insertion_order": False,
    }


def snapshot(
    db: Database,
    directory: Path,
    *,
    sources: dict[str, Any] | None = None,
) -> Path:
    """Persist one supplied database projection for reuse within its own phase.

    Requested columns and indexes come from the packaged recipe. Only present
    columns are retained: the harness may remove unused, entirely-null columns.
    A recipe may explicitly name audited all-null columns and their SQL types
    under ``nullable_columns``; only these are restored as typed nulls. SQL
    binding rejects any other missing column required by the actual query.
    """
    import duckdb
    import numpy as np
    import pandas as pd

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "snapshot.duckdb"
    if output.exists():
        raise FileExistsError(f"Snapshot already exists: {output}")
    sources = (
        sources
        if sources is not None
        else {
            name: {"columns": list(table.df.columns), "indexes": []}
            for name, table in db.table_dict.items()
        }
    )
    with duckdb.connect(str(output), config=configuration(directory)) as connection:
        for name, definition in sources.items():
            if name not in db.table_dict:
                raise ValueError(f"Censored database is missing source table {name!r}")
            frame = db.table_dict[name].df
            declared = definition["columns"]
            columns = [column for column in declared if column in frame.columns]
            nullable = definition.get("nullable_columns", {})
            allowed = {"VARCHAR", "DOUBLE", "BOOLEAN", "TIMESTAMP", "BIGINT"}
            if not set(nullable) <= set(declared) or any(
                kind not in allowed for kind in nullable.values()
            ):
                raise ValueError(
                    f"Invalid audited nullable columns for source {name!r}"
                )
            if "__source_row_id" in frame.columns:
                raise ValueError(f"Source table {name!r} has reserved __source_row_id")
            source = pd.DataFrame(
                {
                    **{column: frame[column] for column in columns},
                    "__source_row_id": np.arange(len(frame), dtype=np.int64),
                },
                copy=False,
            )
            connection.register("__relflow_source", source)
            quoted = '"' + name.replace('"', '""') + '"'
            projection = "*"
            for column, kind in nullable.items():
                if column not in columns:
                    column_sql = '"' + column.replace('"', '""') + '"'
                    projection += f", CAST(NULL AS {kind}) AS {column_sql}"
                    columns.append(column)
            try:
                connection.execute(
                    f"CREATE TABLE {quoted} AS SELECT {projection} "
                    "FROM __relflow_source"
                )
            finally:
                connection.unregister("__relflow_source")
            for number, index in enumerate(definition.get("indexes", [])):
                if not index or not set(index) <= set(declared):
                    raise ValueError(f"Source index {name!r}/{number} lacks columns")
                if not set(index) <= set(columns):
                    continue
                index_columns = ", ".join(
                    '"' + column.replace('"', '""') + '"' for column in index
                )
                connection.execute(
                    f'CREATE INDEX {quoted[:-1]}_{number}" ON {quoted} '
                    f"({index_columns})"
                )
    return output


def prepare(
    snapshot_path: Path,
    table: Table,
    task: EntityTask,
    query: str,
    directory: Path,
    *,
    include_target: bool,
    fetch_size: int = 16_384,
) -> Path:
    """Write one observation per supplied anchor without loading other splits.

    Args:
        snapshot_path: This phase's snapshot of its supplied censored database.
        table: The supplied training, validation, or masked prediction table.
        task: Task metadata identifying the entity, time, and target columns.
        query: Feature-only SELECT with ``$anchors``, ``$start``, and ``$stop``.
        directory: A fresh, phase-specific working directory.
        include_target: Attach supervised labels; false requires a masked table.
        fetch_size: Maximum anchor rows materialized by each query execution.

    Returns:
        The completed, ordered observations Parquet path.
    """
    import duckdb
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from relbench.base import TaskType

    if not isinstance(include_target, bool):
        raise TypeError("include_target must be a boolean")
    if (
        isinstance(fetch_size, bool)
        or not isinstance(fetch_size, int)
        or fetch_size < 1
    ):
        raise ValueError("fetch_size must be a positive integer")
    frame = table.df
    columns = [task.entity_col, task.time_col]
    if include_target:
        columns.append(task.target_col)
    elif task.target_col in frame.columns:
        raise ValueError("Prediction anchors must be masked: remove the target column")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Task anchors are missing columns: {sorted(missing)}")
    if frame[columns].isna().any().any():
        raise ValueError("Task anchor entities, timestamps, and labels must be nonnull")
    if include_target:
        labels = frame[task.target_col].to_numpy(dtype=np.float64)
        if not np.isfinite(labels).all():
            raise ValueError("Task labels must be finite")
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            if not np.isin(labels, (0, 1)).all():
                raise ValueError("Binary task labels must be zero or one")

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    anchors = directory / "anchors.parquet"
    values: dict[str, Any] = {
        "__row_position": np.arange(len(frame), dtype=np.int64),
        "__entity_key": frame[task.entity_col],
        "__as_of": frame[task.time_col],
    }
    pq.write_table(pa.Table.from_pydict(values), anchors, compression="zstd")
    labels_path = directory / "labels.parquet"
    if include_target:
        pq.write_table(
            pa.table(
                {
                    "__row_position": values["__row_position"],
                    "__target": frame[task.target_col],
                }
            ),
            labels_path,
            compression="zstd",
        )

    settings = configuration(directory)
    snapshot_path = Path(snapshot_path).resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Phase snapshot does not exist: {snapshot_path}")
    query = query.strip().removesuffix(";").strip()
    with duckdb.connect(
        str(snapshot_path), read_only=True, config=settings
    ) as connection:
        statements = connection.extract_statements(query)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("Feature SQL must contain exactly one SELECT")
        projection, join = "features.*", ""
        if include_target:
            target_type = (
                "BOOLEAN"
                if task.task_type == TaskType.BINARY_CLASSIFICATION
                else "DOUBLE"
            )
            projection += f", CAST(labels.__target AS {target_type}) AS target"
            join = (
                " LEFT JOIN (SELECT __row_position, __target FROM "
                "read_parquet($labels) WHERE __row_position >= $start "
                "AND __row_position < $stop) AS labels "
                "ON features.__row_position = labels.__row_position"
            )
        statement = (
            f"SELECT {projection} FROM ({query}) AS features{join} "
            "ORDER BY features.__row_position"
        )
        pending = directory / "observations.pending.parquet"
        writer = None
        schema = None
        try:
            for start in range(0, max(len(frame), 1), fetch_size):
                stop = min(start + fetch_size, len(frame))
                parameters = {"anchors": str(anchors), "start": start, "stop": stop}
                if include_target:
                    parameters["labels"] = str(labels_path)
                with connection.execute(statement, parameters).to_arrow_reader(
                    fetch_size
                ) as reader:
                    current = reader.schema
                    if (
                        current.names.count("__row_position") != 1
                        or current.names.count("target") != int(include_target)
                        or "__target" in current.names
                    ):
                        raise ValueError(
                            "Feature SQL must return one __row_position and no targets"
                        )
                    if current.field("__row_position").type != pa.int64():
                        raise ValueError("Observation __row_position must be int64")
                    current = pa.schema(
                        [field.with_type(decimal_type(field.type)) for field in current]
                    )
                    if schema is None:
                        schema = current
                        writer = pq.ParquetWriter(pending, schema, compression="zstd")
                    elif not schema.equals(current):
                        raise ValueError(
                            "Observation schema changed between anchor chunks"
                        )
                    position = start
                    for batch in reader:
                        positions = batch.column("__row_position")
                        end = position + batch.num_rows
                        expected = pa.array(np.arange(position, end, dtype=np.int64))
                        if (
                            end > stop
                            or positions.null_count
                            or pc.all(pc.equal(positions, expected)).as_py() is False
                        ):
                            raise ValueError(
                                "Feature SQL must preserve exactly one ordered row "
                                "per anchor; found missing or duplicate positions"
                            )
                        if include_target and batch.column("target").null_count:
                            raise ValueError("Observation targets must be nonnull")
                        assert writer is not None
                        writer.write_batch(batch.cast(schema))
                        position = end
                    if position != stop:
                        raise ValueError(
                            "Feature SQL must preserve exactly one row per anchor; "
                            f"expected {stop - start}, got {position - start}"
                        )
        finally:
            if writer is not None:
                writer.close()
    output = directory / "observations.parquet"
    pending.replace(output)
    return output
