"""Derive mature outcomes and relational features from the supplied temporal split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
from relbench.base import EntityTask, Table, TaskType

from . import derive, raw, targets
from .derive import quote
from .hurdle import Hurdle

if TYPE_CHECKING:
    from .context import Context


@dataclass(frozen=True)
class Events:
    """One direct dated relation, its count windows and numeric projections."""

    table: str
    entity: str
    clock: str
    counts: tuple[str, ...]
    means: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Cohort:
    """One declared foreign-key path connecting entities through a shared key."""

    table: str
    entity: str
    group: str
    group_table: str
    support: str
    mean: str


@dataclass
class History:
    """Frozen fitting labels, feature names and phase-specific visibility clocks."""

    labels: pa.Table
    root: str
    horizon: pd.Timedelta
    origin: pd.Timestamp
    validation: pd.Timestamp
    boundary: pd.Timestamp
    history: str
    elapsed: str
    events: list[Events]
    cohorts: list[Cohort]
    hurdle: Hurdle | None
    outcome_mean: str

    @classmethod
    def fit(
        cls,
        ctx: Context,
        task: EntityTask,
        train: Table,
        structure: raw.Task,
        phase: str,
        *,
        ordinal_hurdle: bool = True,
        number_hurdle: bool = True,
    ) -> History:
        """Derive paths from declared keys and store only harness-supplied labels."""
        root = task.entity_table
        validation = pd.Timestamp(task.dataset.val_timestamp)
        frame = train.df[[task.entity_col, task.time_col, task.target_col]].copy(
            deep=False
        )
        frame.columns = ["entity", "anchor", "value"]
        if frame.isna().any().any() or frame.duplicated(["entity", "anchor"]).any():
            raise ValueError(
                "Outcome history needs one non-null label per entity and anchor."
            )
        horizon = pd.Timedelta(task.timedelta)
        if pd.isna(horizon) or horizon <= pd.Timedelta(0):
            raise ValueError("Outcome availability requires a positive task horizon.")
        if phase not in {"inner", "outer"}:
            raise ValueError(f"Unknown outcome history split phase {phase!r}.")
        boundary = (
            validation
            if phase == "inner"
            else pd.Timestamp(task.dataset.test_timestamp)
        )
        frame["known_at"] = frame.anchor + horizon
        if (frame.known_at > boundary).any():
            raise ValueError(
                "A supplied fitting outcome matures after this split's boundary."
            )
        if structure.target_kind is None:
            raise ValueError("Outcome history requires an inferred scalar target kind.")
        value = targets.encode(pa.Array.from_pandas(frame.value), structure.target_kind)
        labels = pa.Table.from_pandas(
            frame.drop(columns="value"), preserve_index=False
        ).append_column("value", value)
        hurdle = (
            Hurdle.fit(
                value.to_numpy(zero_copy_only=False),
                target_kind=structure.target_kind,
            )
            if task.task_type == TaskType.REGRESSION
            and (
                ordinal_hurdle if structure.target_kind == "ordinal" else number_hurdle
            )
            else None
        )

        names = derive.Names()
        names.used.update(raw.model(structure)["fields"])
        history, elapsed = names.take("prior_outcomes"), names.take("anchor_elapsed")
        events, cohorts = [], []
        for table, spec in sorted(ctx.schema.items()):
            keys = spec["fkeys"]
            incoming = sorted(
                column for column, related in keys.items() if related == root
            )
            for entity in incoming:
                if spec["time_col"]:
                    prefix = f"{table}_{derive.stem(entity)}"
                    counts = tuple(
                        names.take(f"{prefix}_{window}_count")
                        for window in ("all", "horizon", "four_horizons")
                    )
                    # Use direct measurements admitted by the schema profiler.
                    numeric = sorted(
                        {
                            c.source[0]
                            for b in structure.branches
                            if b.table == table
                            for c in b.columns
                            if c.kind == "number"
                            and len(c.source) == 1
                            and c.recency is None
                        }
                    )
                    means = tuple(
                        (col, names.take(f"{prefix}_{col}_mean")) for col in numeric
                    )
                    events.append(
                        Events(table, entity, spec["time_col"], counts, means)
                    )
            entity_keys = [spec["pkey"]] if table == root else incoming
            for entity in entity_keys:
                for group, related in sorted(keys.items()):
                    if group == entity or not ctx.schema.get(related, {}).get("pkey"):
                        continue
                    prefix = f"peers_{table}_{derive.stem(entity)}_{derive.stem(group)}"
                    cohorts.append(
                        Cohort(
                            table,
                            entity,
                            group,
                            related,
                            names.take(prefix + "_count"),
                            names.take(prefix + "_mean"),
                        )
                    )
        return cls(
            labels,
            root,
            horizon,
            frame.anchor.min(),
            validation,
            boundary,
            history,
            elapsed,
            events,
            cohorts,
            hurdle,
            names.take("prior_outcomes_mean"),
        )

    def fields(self, structure: raw.Task) -> dict:
        """Bind the SQL projections to scalar and repeated model fields."""
        fields = {
            self.elapsed: {
                "type": "duration",
                "query": "as_of",
                "start": self.origin.isoformat(),
                "granularity": "days",
            },
            self.history: {
                "type": "branch",
                "length": None,
                "overflow": None,
                "fields": {
                    "value": {"type": structure.target_kind},
                    "age": {
                        "type": "duration",
                        "query": ["known_at", "../as_of"],
                        "granularity": "days",
                    },
                },
            },
        }
        fields[self.outcome_mean] = {"type": "number"}
        for events in self.events:
            fields.update({name: {"type": "number"} for name in events.counts})
            fields.update({name: {"type": "number"} for _, name in events.means})
        for cohort in self.cohorts:
            fields.update(
                {cohort.mean: {"type": "number"}, cohort.support: {"type": "number"}}
            )
        return fields

    def prepare(self, ctx: Context) -> None:
        """Build the event and outcome timelines once per database connection."""
        con = ctx.connection
        con.execute("CREATE SCHEMA relflow_outcomes")
        con.from_arrow(self.labels).to_table("relflow_outcomes.labels")
        con.execute("""CREATE TABLE relflow_outcomes.history AS
            SELECT entity, list(struct_pack(value := value, known_at := known_at)
                                ORDER BY known_at) AS outcomes
            FROM relflow_outcomes.labels GROUP BY entity""")
        for i, event in enumerate(self.events):
            aggregates = ["count(*) AS n"]
            prefixes = ["CAST(sum(n) OVER w AS BIGINT) AS n"]
            for j, (col, _) in enumerate(event.means):
                column = f"CAST(edge.{quote(col)} AS DOUBLE)"
                finite = f"CASE WHEN isfinite({column}) THEN {column} END"
                aggregates.extend(
                    [f"sum({finite}) AS s{j}", f"count({finite}) AS n{j}"]
                )
                prefixes.append(
                    f"sum(s{j}) OVER w / nullif(sum(n{j}) OVER w, 0) AS avg{j}"
                )
            entity = f"edge.{quote(event.entity)}"
            clock = f"edge.{quote(event.clock)}"
            root = ctx.schema[self.root]
            source = f"{quote(event.table)} edge"
            source += (
                f" JOIN {quote(self.root)} r ON {entity} = r.{quote(root['pkey'])}"
            )
            availability = clock
            valid = f"{clock} IS NOT NULL"
            if root["time_col"]:
                root_clock = f"r.{quote(root['time_col'])}"
                availability = f"greatest({clock}, {root_clock})"
                valid += f" AND {root_clock} IS NOT NULL"
            con.execute(f"""CREATE TABLE relflow_outcomes.events_{i} AS
                WITH ticks AS (
                    SELECT {entity} AS entity, {availability} AS t,
                           {", ".join(aggregates)}
                    FROM {source} WHERE {valid} GROUP BY 1, 2)
                SELECT entity, t, {", ".join(prefixes)} FROM ticks
                WINDOW w AS (PARTITION BY entity ORDER BY t)""")
        for i, cohort in enumerate(self.cohorts):
            self.prepare_cohort(ctx, i, cohort)

    def prepare_cohort(self, ctx: Context, i: int, path: Cohort) -> None:
        """Assign each donor's latest mature label to its cohort at the donor anchor."""
        con = ctx.connection
        root_spec, other = ctx.schema[self.root], ctx.schema[path.group_table]
        source = f"{quote(path.table)} edge"
        entity, group = f"edge.{quote(path.entity)}", f"edge.{quote(path.group)}"
        endpoints = [("edge", ctx.schema[path.table]), ("peer", other)]
        if path.table != self.root:
            root_pk = quote(root_spec["pkey"])
            source += f" JOIN {quote(self.root)} r ON r.{root_pk} = {entity}"
            endpoints.append(("r", root_spec))
        peer_pk = quote(other["pkey"])
        source += f" JOIN {quote(path.group_table)} peer ON peer.{peer_pk} = {group}"
        clocks = [
            f"{alias}.{quote(spec['time_col'])}"
            for alias, spec in endpoints
            if spec["time_col"]
        ]
        # A timeless link exists throughout the snapshot. A dated row with a
        # missing clock supplies no observable link.
        available = (
            "greatest(" + ", ".join(clocks) + ")" if clocks else "TIMESTAMP '-infinity'"
        )
        valid = " AND ".join(f"{c} IS NOT NULL" for c in clocks) or "TRUE"
        con.execute(f"""CREATE TABLE relflow_outcomes.members_{i} AS
            SELECT {entity} AS entity, {available} AS t, min({group}) AS cohort
            FROM {source} WHERE {valid} GROUP BY 1, 2""")
        con.execute(
            f"""CREATE TABLE relflow_outcomes.donors_{i} AS
                SELECT l.entity, l.known_at AS t, m.cohort,
                       CAST(l.value AS DOUBLE) AS amount
                FROM (SELECT *, least(anchor, ?) AS visible
                      FROM relflow_outcomes.labels) l
                ASOF LEFT JOIN relflow_outcomes.members_{i} m
                    ON l.entity = m.entity AND l.visible >= m.t""",
            [self.validation],
        )
        # Remove the old contribution when a donor's cohort or outcome changes.
        con.execute(f"""CREATE TABLE relflow_outcomes.totals_{i} AS
            WITH transitions AS (
                SELECT *, lag(cohort) OVER w AS old_cohort,
                       lag(amount) OVER w AS old_amount
                FROM relflow_outcomes.donors_{i}
                WINDOW w AS (PARTITION BY entity ORDER BY t)),
            deltas AS (
                SELECT cohort, t, amount AS delta, 1 AS n FROM transitions
                WHERE cohort IS NOT NULL UNION ALL
                SELECT old_cohort, t, -old_amount, -1 FROM transitions
                WHERE old_cohort IS NOT NULL),
            ticks AS (
                SELECT cohort, t, sum(delta) AS delta, sum(n) AS n
                FROM deltas GROUP BY 1, 2)
            SELECT cohort, t, sum(delta) OVER w AS total,
                   CAST(sum(n) OVER w AS BIGINT) AS n
            FROM ticks WINDOW w AS (PARTITION BY cohort ORDER BY t)""")

    def query(self, base: str) -> str:
        """Decorate each anchor with as-of counts, raw outcomes and leave-out means."""
        columns = [
            "coalesce(list_filter(h.outcomes, x -> x.known_at <= a.visible), []) "
            f"AS {quote(self.history)}",
            f"list_avg(list_transform({quote(self.history)}, "
            f"x -> CAST(x.value AS DOUBLE))) AS {quote(self.outcome_mean)}",
        ]
        joins = ["LEFT JOIN relflow_outcomes.history h ON h.entity = a.__entity_key"]
        for i, event in enumerate(self.events):
            for j, clock in enumerate(("a.visible", "a.recent", "a.older")):
                alias = f"e{i}_{j}"
                joins.append(
                    f"ASOF LEFT JOIN relflow_outcomes.events_{i} {alias} "
                    f"ON a.__entity_key = {alias}.entity AND {clock} >= {alias}.t"
                )
            n = f"coalesce(e{i}_0.n, 0)"
            for j, name in enumerate(event.counts):
                expr = n if j == 0 else f"{n} - coalesce(e{i}_{j}.n, 0)"
                columns.append(f"{expr} AS {quote(name)}")
            columns.extend(
                f"e{i}_0.avg{j} AS {quote(name)}"
                for j, (_, name) in enumerate(event.means)
            )
        for i, path in enumerate(self.cohorts):
            joins.extend(
                [
                    f"ASOF LEFT JOIN relflow_outcomes.members_{i} m{i} "
                    f"ON a.__entity_key = m{i}.entity AND a.visible >= m{i}.t",
                    f"ASOF LEFT JOIN relflow_outcomes.totals_{i} g{i} "
                    f"ON m{i}.cohort = g{i}.cohort AND a.visible >= g{i}.t",
                    f"ASOF LEFT JOIN relflow_outcomes.donors_{i} d{i} "
                    f"ON a.__entity_key = d{i}.entity AND a.visible >= d{i}.t",
                ]
            )
            own = f"m{i}.cohort = d{i}.cohort"
            count = f"(coalesce(g{i}.n, 0) - CASE WHEN {own} THEN 1 ELSE 0 END)"
            total = f"(g{i}.total - CASE WHEN {own} THEN d{i}.amount ELSE 0 END)"
            columns.extend(
                [
                    f"{count} AS {quote(path.support)}",
                    f"{total} / nullif({count}, 0) AS {quote(path.mean)}",
                ]
            )
        return f"""WITH observations AS ({base}), a AS (
            SELECT *, least(__as_of, $boundary) AS visible,
                least(__as_of - CAST($horizon AS INTERVAL), visible) AS recent,
                least(__as_of - 4 * CAST($horizon AS INTERVAL), visible) AS older
            FROM __relflow_anchors
            WHERE __row_position >= $start AND __row_position < $stop)
            SELECT o.*, {", ".join(columns)}
            FROM observations o JOIN a USING (__row_position)
            {" ".join(joins)} ORDER BY o.__row_position"""
