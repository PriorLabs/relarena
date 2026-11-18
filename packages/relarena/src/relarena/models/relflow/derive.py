"""Infer nested model structure from keys and label-free column measurements.

``profile`` measures the supplied, censored tables; ``derive`` chooses typed
fields and bounded relational branches for the original training-set size.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace

import duckdb

from . import raw
from .raw import Branch, Col, Lookup, Task

SAMPLE = 200_000
# Every structural budget is fixed by the original training-anchor count.
TIERS = ((20_000, "small"), (500_000, "medium"), (None, "large"))
BUDGETS = {
    "small": {"branch": 128, "nested": 16, "fields": 48, "root": 64},
    "medium": {"branch": 64, "nested": 16, "fields": 40, "root": 48},
    "large": {"branch": 32, "nested": 8, "fields": 24, "root": 32},
}
NESTED_FANOUT = 32  # p90 rows per key above which a child table is not nested
LINK_COLUMNS = 2  # own columns at most for a branch to count as a link table
ALIGNED = (
    0.99,
    0.5,
)  # unique shared key, and a majority of the sibling's keys covered by the branch
SOURCE_ROW_ID = "__source_row_id"
RESERVED = {"as_of", "target", "occurred_at", "age_days"}
# Table aliases the rendered query already uses; a lookup alias must not shadow them.
RESERVED_ALIASES = {
    "anchor",
    "anchors",
    "root",
    "parent",
    "event",
    "linked",
    "labels",
    "features",
}
IDENTIFIER = re.compile(
    r"(?i:(^|[^a-z])(id|ids|code|type|status|kind|class)([^a-z]|$))|[A-Za-z](Id|ID)$"
)
INTEGERS = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
}
VECTOR_BLOCK = re.compile(r"^(.*?)_?(\d+)$")


@dataclass(frozen=True)
class Column:
    """Measurements of one column on a sample of its table."""

    name: str
    dtype: str
    n: int
    nonnull: int
    distinct: int
    mean_len: float | None = None
    wordy: float | None = None
    spaced: float | None = None
    fixed: bool | None = None
    integral: float | None = None
    daytime: float | None = None  # share of timestamps carrying a time of day


@dataclass(frozen=True)
class Table:
    """Schema and column measurements for one supplied relational table."""

    name: str
    # SQL relation the profiler reads, such as read_parquet('...') or a view name
    source: str
    pkey: str | None
    time: str | None
    fkeys: dict[str, str]
    rows: int
    columns: dict[str, Column]


@dataclass(frozen=True)
class Meta:
    """Task columns and original training size used to derive model structure."""

    dataset: str
    task: str
    task_type: str
    entity_table: str
    entity_col: str  # the task table's entity column
    time_col: str  # the task table's timestamp column
    target_col: str
    training_rows: int | None = (
        None  # anchors in the training split; selects the size tier
    )


def tier(training_rows: int | None) -> str:
    """The size tier of a task; unknown sizes take the smallest tier's budgets."""
    for limit, name in TIERS:
        if limit is None or training_rows is None or training_rows < limit:
            return name
    raise AssertionError("the last tier is unbounded")


def quote(name: str) -> str:
    """Quote a SQL identifier, escaping embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


def profile(
    con: duckdb.DuckDBPyConnection,
    sources: Mapping[str, str],
    schema: Mapping[str, dict],
) -> dict[str, Table]:
    """Measure every column of every table on one sample per table."""
    tables = {}
    for index, (name, source) in enumerate(sources.items()):
        spec = schema[name]
        sample = f"sample_{index}"
        con.execute(
            (
                f"CREATE OR REPLACE TEMP TABLE {sample} AS SELECT * FROM {source} "
                f"USING SAMPLE reservoir({SAMPLE} ROWS) REPEATABLE (0)"
            )
        )
        rows = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        dtypes = con.execute(f"DESCRIBE {sample}").fetchall()
        columns = {}
        for column, dtype, *_ in dtypes:
            if column == SOURCE_ROW_ID:
                continue
            v = quote(column)
            n, nonnull, distinct = con.execute(
                f"SELECT count(*), count({v}), count(DISTINCT {v}) FROM {sample}"
            ).fetchone()
            measured = Column(column, dtype, n, nonnull, distinct)
            if dtype.startswith("VARCHAR") and not dtype.endswith("[]"):
                row = con.execute(
                    f"SELECT avg(length({v})), avg(CASE WHEN "
                    f"regexp_matches({v}, '\\p{{L}}{{2,}}') THEN 1 ELSE 0 END), "
                    f"avg(CASE WHEN {v} LIKE '% %' THEN 1 ELSE 0 END), "
                    f"min(length({v})) = max(length({v})) "
                    f"FROM {sample} WHERE {v} IS NOT NULL"
                ).fetchone()
                measured = Column(
                    column,
                    dtype,
                    n,
                    nonnull,
                    distinct,
                    mean_len=row[0],
                    wordy=row[1],
                    spaced=row[2],
                    fixed=row[3],
                )
            elif dtype in ("DOUBLE", "FLOAT") or dtype.startswith("DECIMAL"):
                integral = con.execute(
                    (
                        f"SELECT avg(CASE WHEN {v} = floor({v}) THEN 1 ELSE 0 END) "
                        f"FROM {sample} WHERE {v} IS NOT NULL AND isfinite({v})"
                    )
                ).fetchone()[0]
                measured = Column(
                    column, dtype, n, nonnull, distinct, integral=integral
                )
            elif dtype.startswith("TIMESTAMP"):
                daytime = con.execute(
                    (
                        f"SELECT avg(CASE WHEN date_trunc('day', {v}) <> {v} THEN 1 "
                        f"ELSE 0 END) FROM {sample} WHERE {v} IS NOT NULL"
                    )
                ).fetchone()[0]
                measured = Column(column, dtype, n, nonnull, distinct, daytime=daytime)
            columns[column] = measured
        tables[name] = Table(
            name,
            source,
            spec.get("pkey"),
            spec.get("time_col"),
            dict(spec.get("fkeys") or {}),
            rows,
            columns,
        )
    return tables


def fanout(
    con: duckdb.DuckDBPyConnection, table: Table, keys: tuple[str, ...]
) -> float:
    """The ninetieth percentile of rows per key tuple."""
    group = ", ".join(quote(key) for key in keys)
    present = " AND ".join(f"{quote(key)} IS NOT NULL" for key in keys)
    value = con.execute(
        (
            f"SELECT quantile_cont(n, 0.9) FROM (SELECT count(*) AS n FROM "
            f"{table.source} WHERE {present} GROUP BY {group})"
        )
    ).fetchone()[0]
    return float(value or 0)


def peak(con: duckdb.DuckDBPyConnection, table: Table, keys: tuple[str, ...]) -> int:
    """Return the maximum number of rows sharing a key tuple."""
    group = ", ".join(quote(key) for key in keys)
    present = " AND ".join(f"{quote(key)} IS NOT NULL" for key in keys)
    value = con.execute(
        (
            f"SELECT max(n) FROM (SELECT count(*) AS n FROM {table.source} WHERE "
            f"{present} GROUP BY {group})"
        )
    ).fetchone()[0]
    return int(value or 0)


def alignment(
    con: duckdb.DuckDBPyConnection,
    sibling: Table,
    branch: Table,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[float, float]:
    """Measure sibling uniqueness and overlap on shared keys."""
    own = ", ".join(quote(s) for s, _ in pairs)
    other = ", ".join(f"{quote(b)} AS {quote(s)}" for s, b in pairs)
    own_present = " AND ".join(f"{quote(s)} IS NOT NULL" for s, _ in pairs)
    other_present = " AND ".join(f"{quote(b)} IS NOT NULL" for _, b in pairs)
    unique = con.execute(
        (
            f"SELECT count(DISTINCT ({own})) / greatest(count(*), 1) FROM "
            f"{sibling.source} WHERE {own_present}"
        )
    ).fetchone()[0]
    coverage = con.execute(
        f"WITH s AS (SELECT DISTINCT {own} FROM {sibling.source} WHERE {own_present}), "
        f"b AS (SELECT DISTINCT {other} FROM {branch.source} WHERE {other_present}) "
        f"SELECT (SELECT count(*) FROM s JOIN b USING ({own})) / "
        f"greatest((SELECT count(*) FROM s), 1)"
    ).fetchone()[0]
    return float(unique or 0), float(coverage or 0)


def references(tables: Mapping[str, Table], name: str) -> int:
    """Count incoming references to rows of this table."""
    return sum(
        t.rows for t in tables.values() for target in t.fkeys.values() if target == name
    )


def kind(
    column: Column, table: Table, tables: Mapping[str, Table], *, root: bool
) -> str | None:
    """The tensorfield for one measured column, or None to leave it out."""
    d = column.dtype
    if column.nonnull == 0 or (column.distinct <= 1 and column.nonnull == column.n):
        return None  # constant with no null signal
    if column.distinct <= 1:
        return "category"  # one value or null: a flag
    if column.name == table.pkey:
        return "hash" if not root and references(tables, table.name) else None
    if column.name in table.fkeys:
        return "hash"
    if d.startswith(("TIMESTAMP", "DATE")):
        return "dateparts"
    if d == "BOOLEAN":
        return "boolean"
    if d.endswith("[]"):
        return "set" if d.startswith("VARCHAR") else "vector"
    if d.startswith(("STRUCT", "MAP")):
        return None
    numeric = d in INTEGERS or d in ("DOUBLE", "FLOAT") or d.startswith("DECIMAL")
    if numeric:
        integral = d in INTEGERS or (column.integral or 0) >= 0.999
        if integral and column.distinct <= 3:
            return "category"  # binary and ternary flags
        if integral and IDENTIFIER.search(column.name):
            return "category" if column.distinct <= 20_000 else "hash"
        return "number"
    if d.startswith("VARCHAR"):
        weight = max(1.0, references(tables, table.name) / max(table.rows, 1))
        ratio = column.distinct / max(column.nonnull, 1)
        wordy = column.wordy or 0
        if (
            column.fixed
            and not column.spaced
            and ratio >= 0.5
            and (column.mean_len or 0) >= 8
        ):
            return "hash"  # opaque fixed-width identifiers
        if (column.mean_len or 0) >= 40 and wordy >= 0.3:
            return "text"  # prose, however often it recurs
        if column.distinct <= 25_000 and ratio / weight <= 0.25:
            return "category"
        if wordy >= 0.3:
            return "text"
        return "hash"
    return None


def budget(mean_len: float | None) -> int:
    """Choose the bounded token length for a text field."""
    length = mean_len or 0
    return 16 if length < 40 else 32 if length < 160 else 64 if length < 700 else 96


def same_time(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    column: str,
    row_time: str,
    dimension: Table,
    on: str | None = None,
) -> bool:
    """Check whether a dimension repeats the row timestamp."""
    rate = con.execute(
        f"SELECT avg(CASE WHEN r.{quote(row_time)} = d.{quote(dimension.time)} "
        f"THEN 1 ELSE 0 END) FROM (SELECT {quote(column)}, {quote(row_time)} "
        f"FROM {table.source} USING SAMPLE reservoir({SAMPLE} ROWS) "
        f"REPEATABLE (0)) r JOIN {dimension.source} d "
        f"ON r.{quote(column)} = d.{quote(on or dimension.pkey)} "
        f"WHERE r.{quote(row_time)} IS NOT NULL "
        f"AND d.{quote(dimension.time)} IS NOT NULL"
    ).fetchone()[0]
    return (rate or 0) >= 0.99


def trim(
    columns: tuple[Col, ...], groups: list[list[Lookup]], budget: int
) -> tuple[tuple[Col, ...], list[Lookup]]:
    """Keep own columns first, then lookup groups, within the field budget."""
    own = columns[:budget]
    remaining = budget - len(own)
    kept: list[Lookup] = []
    for group in groups:
        for lookup in group:
            if remaining <= 0:
                break
            take = lookup.columns[:remaining]
            if take:
                kept.append(replace(lookup, columns=take))
                remaining -= len(take)
    return own, kept


def snake(name: str) -> str:
    """Convert a field name to lowercase underscore-separated words."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text or text[0].isdigit():
        text = "c_" + text
    return text


def stem(column: str) -> str:
    """Derive a lookup alias from its foreign-key column."""
    name = re.sub(r"(?i)[_ ]?id$", "", column) or column
    return snake(name)


class Names:
    """Unique field names within one node; reserved and internal names get a prefix."""

    def __init__(self, reserved: frozenset[str] = frozenset(RESERVED)) -> None:
        """Reserve existing names before allocating derived field names."""
        self.used: set[str] = set()
        self.reserved = reserved

    def take(self, name: str) -> str:
        """Allocate a unique, non-reserved field or SQL alias name."""
        name = snake(name)
        if name in self.reserved or name.startswith("__"):
            name = "col_" + name.lstrip("_")
        candidate, index = name, 2
        while candidate in self.used:
            candidate = f"{name}_{index}"
            index += 1
        self.used.add(candidate)
        return candidate


def vector_blocks(table: Table) -> dict[str, list[str]]:
    """Group runs of at least eight numbered numeric columns into vectors."""
    groups: dict[str, list[str]] = {}
    for column in table.columns.values():
        match = VECTOR_BLOCK.match(column.name)
        if (
            match
            and match.group(1)
            and (column.dtype in INTEGERS or column.dtype in ("DOUBLE", "FLOAT"))
        ):
            groups.setdefault(match.group(1), []).append(column.name)
    return {prefix: members for prefix, members in groups.items() if len(members) >= 8}


def typed(
    table: Table,
    tables: Mapping[str, Table],
    dataset: str,
    names: Names,
    *,
    prefix: str = "",
    exclude: set[str] = frozenset(),
    root: bool = False,
    text: bool = True,
) -> tuple[Col, ...]:
    """Describe table columns in order, optionally including text."""
    blocks = vector_blocks(table)
    member_of = {
        member: block for block, members in blocks.items() for member in members
    }
    emitted: set[str] = set()
    result: list[Col] = []
    for column in table.columns.values():
        if column.name in exclude:
            continue
        if column.name in member_of:
            block = member_of[column.name]
            if block in emitted:
                continue
            emitted.add(block)
            members = blocks[block]
            values = ", ".join(
                f"coalesce({{t}}.{quote(member)}, 0)" for member in members
            )
            field = names.take(prefix + block)
            result.append(
                Col(
                    field,
                    f"CAST(list_value({values}) AS FLOAT[{len(members)}])",
                    "vector",
                    {"n_dim": len(members)},
                    source=tuple(members),
                )
            )
            continue
        k = kind(column, table, tables, root=root)
        if k == "text" and not text:
            k = "category" if column.distinct <= 25_000 else None
        if k is None:
            continue
        field = names.take(prefix + column.name)
        if k == "dateparts":
            parts = raw.when(column.name, field)
            if (column.daytime or 0) >= 0.01:
                parts[0] = replace(
                    parts[0], options={**parts[0].options, "intraday": True}
                )
            result.extend(parts)
        elif k == "hash":
            namespace = f"{dataset}:{table.fkeys.get(column.name, table.name)}"
            result.append(raw.hashed(column.name, namespace, field))
        elif k == "text":
            result.append(raw.text(column.name, field, budget(column.mean_len)))
        elif k == "number":
            result.append(raw.number(column.name, field))
        elif k == "category":
            result.append(raw.category(column.name, field))
        elif k == "boolean":
            result.append(raw.boolean(column.name, field))
        elif k == "set":
            result.append(raw.setvalued(column.name, field))
    return tuple(result)


def lookups_for(
    con: duckdb.DuckDBPyConnection,
    table: Table,
    tables: Mapping[str, Table],
    dataset: str,
    names: Names,
    *,
    exclude: set[str] = frozenset(),
    hops: int = 2,
    text: bool = True,
    row_time: str | None = None,
) -> tuple[list[Lookup], list[Lookup]]:
    """Follow foreign keys, with a second hop only into vocabulary tables."""
    aliases = Names(frozenset(RESERVED_ALIASES))
    first: list[Lookup] = []
    second: list[Lookup] = []
    for column, target in table.fkeys.items():
        if column in exclude or target not in tables or tables[target].pkey is None:
            continue
        dimension = tables[target]
        alias = aliases.take(stem(column))
        skip = {dimension.pkey}
        if (
            row_time
            and dimension.time
            and same_time(con, table, column, row_time, dimension)
        ):
            skip.add(dimension.time)
        columns = typed(
            dimension,
            tables,
            dataset,
            names,
            prefix=alias + "_",
            exclude=skip,
            text=text,
        )
        onward = [
            (column2, target2)
            for column2, target2 in dimension.fkeys.items()
            if hops >= 2
            and target2 in tables
            and tables[target2].pkey is not None
            and vocabulary(tables[target2])
        ]
        # The second hop joins through the first hop's columns, so the snapshot keeps
        # them whatever the budget keeps.
        # They are not part of the key: a composite index over them is useless and
        # aborts DuckDB on wide tables.
        first.append(
            Lookup(
                alias,
                target,
                f"{{t}}.{quote(dimension.pkey)} = {{row}}.{quote(column)}",
                columns,
                time=dimension.time,
                key=(dimension.pkey,),
                via=(column,),
                reads=tuple(column2 for column2, _ in onward),
            )
        )
        for column2, target2 in onward:
            dimension2 = tables[target2]
            alias2 = aliases.take(f"{alias}_{stem(column2)}")
            columns2 = typed(
                dimension2,
                tables,
                dataset,
                names,
                prefix=alias2 + "_",
                exclude={dimension2.pkey},
                text=False,
            )
            second.append(
                Lookup(
                    alias2,
                    target2,
                    f"{{t}}.{quote(dimension2.pkey)} = {alias}.{quote(column2)}",
                    columns2,
                    time=dimension2.time,
                    key=(dimension2.pkey,),
                )
            )
    return first, second


def shared_keys(sibling: Table, branch: Table) -> tuple[tuple[str, str], ...]:
    """Pair sibling and branch columns pointing to shared foreign-key targets."""
    pairs = []
    seen = set()
    for s_column, target in sibling.fkeys.items():
        for b_column, b_target in branch.fkeys.items():
            if target == b_target and target not in seen:
                pairs.append((s_column, b_column))
                seen.add(target)
    return tuple(pairs)


def vocabulary(table: Table) -> bool:
    """Identify a static dimension without foreign keys."""
    return table.time is None and not table.fkeys


def belongs_solely(child: Table, dimension: str, tables: Mapping[str, Table]) -> bool:
    """Check whether other child foreign keys lead only to vocabulary tables."""
    return all(
        target == dimension or (target in tables and vocabulary(tables[target]))
        for target in child.fkeys.values()
    )


def intraday(table: Table) -> bool:
    """Whether a table's time column carries a time of day worth an hour part."""
    column = table.columns.get(table.time) if table.time else None
    return bool(column is not None and (column.daytime or 0) >= 0.01)


def own_columns(table: Table) -> int:
    """Columns that are not keys or the time column."""
    return sum(
        1
        for name in table.columns
        if name != table.pkey and name != table.time and name not in table.fkeys
    )


def nested_for(
    con: duckdb.DuckDBPyConnection,
    branch: Table,
    tables: Mapping[str, Table],
    dataset: str,
    *,
    entity: str,
    absorbed: set[str],
    budgets: Mapping[str, int],
) -> tuple[str | None, tuple[Branch, ...], tuple[tuple[Table, str], ...]]:
    """Build nested branches and one-to-one extensions of branch rows."""
    options: dict[str, list[tuple[Table, str]]] = {}
    link = own_columns(branch) <= LINK_COLUMNS
    for name, candidate in tables.items():
        if name in absorbed:
            continue
        for column, target in candidate.fkeys.items():
            if target == branch.name and branch.pkey is not None:
                options.setdefault(branch.pkey, []).append((candidate, column))
            elif link and target != entity and target in set(branch.fkeys.values()):
                if candidate.name == branch.name or not belongs_solely(
                    candidate, target, tables
                ):
                    continue
                for via, via_target in branch.fkeys.items():
                    if via_target == target:
                        options.setdefault(via, []).append((candidate, column))
    best: tuple[str | None, list[Branch], list[tuple[Table, str]]] = (None, [], [])
    for child_key, candidates in options.items():
        branches: list[Branch] = []
        folded: list[tuple[Table, str]] = []
        taken = Names()
        for candidate, column in candidates:
            rows_per_key = fanout(con, candidate, (column,))
            if rows_per_key > NESTED_FANOUT:
                continue
            if rows_per_key <= 1 and peak(con, candidate, (column,)) <= 1:
                folded.append((candidate, column))
                continue
            names = Names()
            columns = typed(
                candidate,
                tables,
                dataset,
                names,
                exclude={column, *((candidate.time,) if candidate.time else ())},
                text=False,
            )
            hop1, _ = lookups_for(
                con,
                candidate,
                tables,
                dataset,
                names,
                exclude={column},
                hops=1,
                text=False,
                row_time=candidate.time,
            )
            columns, lookups = trim(columns, [hop1], budgets["fields"])
            branches.append(
                Branch(
                    taken.take(
                        f"{candidate.name}_via_{column}"
                        if sum(1 for c, _ in candidates if c is candidate) > 1
                        else candidate.name
                    ),
                    candidate.name,
                    f'{{t}}.{quote(column)} = {{parent}}."__child_key"',
                    candidate.time,
                    budgets["nested"],
                    columns,
                    lookups=tuple(lookups),
                    key=(column,),
                    intraday=intraday(candidate),
                )
            )
        if len(branches) + len(folded) > len(best[1]) + len(best[2]):
            best = (child_key, branches, folded)
    return best[0], tuple(best[1]), tuple(best[2])


def derive(
    con: duckdb.DuckDBPyConnection,
    tables: Mapping[str, Table],
    meta: Meta,
    *,
    nested: bool = True,
) -> Task:
    """Build the raw-structure task for one entity table from measured tables.

    ``nested=False`` stops at depth one: an ablation switch for comparing derived
    structures, not a harness option. Prose is admitted at depth one only.
    """
    entity = tables[meta.entity_table]
    if entity.pkey is None:
        raise ValueError(f"entity table {meta.entity_table!r} has no primary key")
    budgets = BUDGETS[tier(meta.training_rows)]
    root_names = Names()
    root_columns = typed(
        entity, tables, meta.dataset, root_names, exclude={entity.pkey}, root=True
    )
    hop1, hop2 = lookups_for(
        con, entity, tables, meta.dataset, root_names, row_time=entity.time
    )
    root_columns, root_lookups = trim(root_columns, [hop1, hop2], budgets["root"])

    candidates = [
        (table, column)
        for table in tables.values()
        for column, target in table.fkeys.items()
        if target == entity.name
    ]
    candidates.sort(
        key=lambda item: (-len(item[0].columns), -item[0].rows, item[0].name, item[1])
    )
    absorbed: set[str] = set()
    kept: set[str] = set()
    branch_names = Names()
    branches: list[Branch] = []
    for table, column in candidates:
        if table.name in absorbed:
            continue
        names = Names()
        columns = typed(
            table,
            tables,
            meta.dataset,
            names,
            exclude={column, *((table.time,) if table.time else ())},
        )
        hop1, hop2 = lookups_for(
            con,
            table,
            tables,
            meta.dataset,
            names,
            exclude={column},
            row_time=table.time,
        )
        aliases = Names(frozenset(RESERVED_ALIASES))
        aliases.used.update(lookup.alias for lookup in hop1 + hop2)
        siblings: list[Lookup] = []
        for sibling in tables.values():
            if (
                sibling.name == table.name
                or sibling.name in absorbed
                or sibling.name in kept
            ):
                continue
            pairs = shared_keys(sibling, table)
            if len(pairs) < 2:
                continue
            unique, coverage = alignment(con, sibling, table, pairs)
            if unique < ALIGNED[0] or coverage < ALIGNED[1]:
                continue
            alias = aliases.take(sibling.name)
            own = {s for s, _ in pairs}
            sibling_columns = typed(
                sibling,
                tables,
                meta.dataset,
                names,
                prefix=alias + "_",
                exclude=own | ({sibling.time} if sibling.time else set()),
            )
            predicate = " AND ".join(
                f"{{t}}.{quote(s)} = {{row}}.{quote(b)}" for s, b in pairs
            )
            siblings.append(
                Lookup(
                    alias,
                    sibling.name,
                    predicate,
                    sibling_columns,
                    time=sibling.time,
                    key=tuple(s for s, _ in pairs),
                    via=tuple(b for _, b in pairs),
                )
            )
            absorbed.add(sibling.name)
        child_key, children, extensions = (
            nested_for(
                con,
                table,
                tables,
                meta.dataset,
                entity=entity.name,
                absorbed=absorbed,
                budgets=budgets,
            )
            if nested
            else (None, (), ())
        )
        folded: list[Lookup] = []
        for extension, column_of in extensions:
            alias = aliases.take(extension.name)
            skip = {column_of}
            if extension.time and (
                not table.time
                or same_time(con, table, child_key, table.time, extension, on=column_of)
            ):
                skip.add(extension.time)
            folded.append(
                Lookup(
                    alias,
                    extension.name,
                    f"{{t}}.{quote(column_of)} = {{row}}.{quote(child_key)}",
                    typed(
                        extension,
                        tables,
                        meta.dataset,
                        names,
                        prefix=alias + "_",
                        exclude=skip,
                    ),
                    time=extension.time,
                    key=(column_of,),
                    via=(child_key,),
                )
            )
        columns, lookups = trim(
            columns, [hop1, siblings, folded, hop2], budgets["fields"]
        )
        multiple = sum(1 for t, _ in candidates if t is table) > 1
        branches.append(
            Branch(
                branch_names.take(
                    f"{table.name}_via_{column}" if multiple else table.name
                ),
                table.name,
                f'{{t}}.{quote(column)} = {{parent}}."__entity_key"',
                table.time,
                budgets["branch"],
                columns,
                lookups=tuple(lookups),
                nested=children,
                child_key=child_key,
                key=(column,),
                intraday=intraday(table),
            )
        )
        kept.add(table.name)

    keys: list[tuple[str, str]] = []
    for via, target in entity.fkeys.items():
        for candidate in tables.values():
            if (
                candidate.name == entity.name
                or candidate.name in absorbed
                or not belongs_solely(candidate, target, tables)
            ):
                continue
            reaching = [
                column
                for column, child_target in candidate.fkeys.items()
                if child_target == target
            ]
            for column in reaching:
                if fanout(con, candidate, (column,)) > budgets["branch"]:
                    continue
                key = f"__{snake(via)}"
                if (key, via) not in keys:
                    keys.append((key, via))
                names = Names()
                columns = typed(
                    candidate,
                    tables,
                    meta.dataset,
                    names,
                    exclude={column, *((candidate.time,) if candidate.time else ())},
                )
                hop1, _ = lookups_for(
                    con,
                    candidate,
                    tables,
                    meta.dataset,
                    names,
                    exclude={column},
                    hops=1,
                    row_time=candidate.time,
                )
                columns, lookups = trim(columns, [hop1], budgets["fields"])
                branches.append(
                    Branch(
                        branch_names.take(
                            f"{stem(via)}_{candidate.name}_via_{column}"
                            if len(reaching) > 1
                            else f"{stem(via)}_{candidate.name}"
                        ),
                        candidate.name,
                        f"{{t}}.{quote(column)} = {{parent}}.{quote(key)}",
                        candidate.time,
                        budgets["branch"],
                        columns,
                        lookups=tuple(lookups),
                        key=(column,),
                        intraday=intraday(candidate),
                    )
                )

    return Task(
        dataset=meta.dataset,
        task=meta.task,
        task_type=meta.task_type,
        entity_table=entity.name,
        entity_key=entity.pkey,
        anchor_entity_column=meta.entity_col,
        target_column=meta.target_col,
        root_time=entity.time,
        columns=root_columns,
        lookups=tuple(root_lookups),
        branches=tuple(branches),
        keys=tuple(keys),
    )
